import boto3
import json
import os
import chromadb
import re
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()

# Cliente de Bedrock para Embeddings
bedrock_client = boto3.client(
    service_name='bedrock-runtime',
    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
)

# Usaremos Amazon Titan V2, que es excelente y muy económico para embeddings
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"

def get_embedding(text: str) -> list:
    """Convierte un bloque de texto en un vector matemático usando AWS Titan."""
    payload = {
        "inputText": text,
        "dimensions": 1024,
        "normalize": True
    }
    
    response = bedrock_client.invoke_model(
        modelId=EMBEDDING_MODEL_ID,
        body=json.dumps(payload),
        accept="application/json",
        contentType="application/json"
    )
    
    response_body = json.loads(response.get('body').read())
    return response_body.get("embedding")

def init_chromadb(db_path: str = "./chroma_db"):
    """Inicializa la conexión a la base de datos vectorial local ChromaDB."""
    # ChromaDB guardará los datos físicamente en la carpeta db_path
    client = chromadb.PersistentClient(path=db_path, settings=Settings(anonymized_telemetry=False))
    return client

def get_or_create_collection(client: chromadb.PersistentClient, collection_name: str = "enterprise_docs"):
    """Obtiene o crea la colección ('tabla') donde vivirán los 1,370 documentos."""
    return client.get_or_create_collection(name=collection_name)

def search_documents_dynamic(collection, query: str, max_tokens: int = 150_000) -> list:
    """
    Búsqueda Híbrida: Combina coincidencia exacta de palabras clave 
    (si se usan comillas) con búsqueda semántica masiva.
    """
    retrieved_docs = []
    seen_ids = set() # Para no duplicar si ambas búsquedas encuentran lo mismo
    current_chars = 0
    CHARS_PER_TOKEN = 3.5
    MAX_CHARS = max_tokens * CHARS_PER_TOKEN

    # --- 1. BÚSQUEDA EXACTA (El "Laser") ---
    # Buscamos si el usuario puso algo entre comillas dobles: ej. "Alfredo Martinez"
    palabras_exactas = re.findall(r'"([^"]*)"', query)
    
    if palabras_exactas:
        for palabra in palabras_exactas:
            try:
                # ChromaDB busca literalmente este texto en los documentos
                exact_results = collection.get(
                    where_document={"$contains": palabra}
                )
                
                if exact_results and exact_results['documents']:
                    for i in range(len(exact_results['documents'])):
                        doc_id = exact_results['ids'][i]
                        doc_text = exact_results['documents'][i]
                        doc_metadata = exact_results['metadatas'][i]
                        
                        if doc_id not in seen_ids:
                            retrieved_docs.append({
                                "id": doc_id,
                                "texto": doc_text,
                                "origen": doc_metadata.get("source", "Desconocido")
                            })
                            seen_ids.add(doc_id)
                            current_chars += len(doc_text)
            except Exception as e:
                print(f"Búsqueda exacta falló para '{palabra}': {e}")

    # --- 2. BÚSQUEDA SEMÁNTICA (La "Red de Arrastre") ---
    if current_chars < MAX_CHARS:
        query_embedding = get_embedding(query)
        
        # Subimos radicalmente de 30 a 150 resultados para aprovechar Claude Long-Context
        semantic_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=150 
        )
        
        if semantic_results and 'documents' in semantic_results and semantic_results['documents']:
            if len(semantic_results['documents'][0]) > 0:
                for i in range(len(semantic_results['documents'][0])):
                    doc_id = semantic_results['ids'][0][i]
                    doc_text = semantic_results['documents'][0][i]
                    doc_metadata = semantic_results['metadatas'][0][i]
                    
                    # Evitamos agregar documentos que ya entraron por la búsqueda exacta
                    if doc_id not in seen_ids:
                        if current_chars + len(doc_text) > MAX_CHARS:
                            break 
                            
                        retrieved_docs.append({
                            "id": doc_id,
                            "texto": doc_text,
                            "origen": doc_metadata.get("source", "Desconocido")
                        })
                        seen_ids.add(doc_id)
                        current_chars += len(doc_text)
                        
    return retrieved_docs