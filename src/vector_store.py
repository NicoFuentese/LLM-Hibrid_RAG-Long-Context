import boto3
import json
import os
import chromadb
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

def search_documents_dynamic(collection, query: str, max_tokens: int = 900_000) -> list:
    """
    Búsqueda dinámica que aprovecha el Long-Context. 
    Extrae todos los documentos relevantes hasta casi llenar la ventana de Claude.
    """
    # 1. Convertimos la pregunta del usuario a un vector
    query_embedding = get_embedding(query)
    
    # 2. Buscamos un número masivo (ej. 300 fragmentos)
    # ChromaDB usa distancia L2 (menor es más similar) o Coseno.
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=300 
    )
    
    retrieved_docs = []
    current_chars = 0
    CHARS_PER_TOKEN = 3.5
    MAX_CHARS = max_tokens * CHARS_PER_TOKEN
    
    if results['documents'] and len(results['documents'][0]) > 0:
        for i in range(len(results['documents'][0])):
            doc_text = results['documents'][0][i]
            doc_metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i] # Nivel de similitud
            
            # Filtro de Relevancia: Ignorar basura (Ajustar el umbral según el modelo de embeddings)
            # En distancia L2, valores más bajos son mejores. 
            if distance > 1.2: 
                continue 
                
            # Control del límite de tokens para no explotar Bedrock
            if current_chars + len(doc_text) > MAX_CHARS:
                break # Si el siguiente doc supera el millón de tokens, nos detenemos
                
            retrieved_docs.append({
                "texto": doc_text,
                "origen": doc_metadata.get("source", "Desconocido")
            })
            current_chars += len(doc_text)
            
    return retrieved_docs