import os
import fitz  # PyMuPDF
import time
from vector_store import init_chromadb, get_or_create_collection, get_embedding

# Configuraciones
DATA_DIR = "/app/data"
DB_PATH = "/app/chroma_db"
CHUNK_SIZE = 4000  # Caracteres por fragmento (~1000 tokens para Titan)
CHUNK_OVERLAP = 400 # Superposición para no cortar ideas por la mitad

def clean_text(text: str) -> str:
    """Limpia caracteres de control problemáticos."""
    import re
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

def extract_text_from_file(filepath: str) -> str:
    """Extrae texto de un PDF o TXT. Retorna string vacío si falla."""
    text = ""
    try:
        if filepath.lower().endswith('.pdf'):
            doc = fitz.open(filepath)
            for page in doc:
                text += page.get_text("text") + "\n"
            doc.close()
        elif filepath.lower().endswith('.txt'):
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
    except Exception as e:
        print(f"⚠️ Error leyendo {filepath}: {e}")
    return clean_text(text)

def chunk_text(text: str, chunk_size: int, overlap: int) -> list:
    """Divide el texto masivo en fragmentos procesables por el modelo de Embeddings."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def process_directory():
    """Recorre las 11 carpetas, extrae, vectoriza y guarda en ChromaDB."""
    print(f"🚀 Iniciando pipeline de ingesta en: {DATA_DIR}")
    
    # 1. Conectar a la base de datos
    client = init_chromadb(DB_PATH)
    collection = get_or_create_collection(client)
    
    archivos_procesados = 0
    fragmentos_totales = 0
    
    # 2. Recorrer recursivamente las carpetas
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.lower().endswith(('.pdf', '.txt')):
                filepath = os.path.join(root, file)
                print(f"📄 Procesando: {file}...")
                
                # Extraer texto
                full_text = extract_text_from_file(filepath)
                if not full_text.strip():
                    continue
                
                # Crear Chunks
                chunks = chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
                
                # Preparar lotes para ChromaDB
                ids = []
                documents = []
                metadatas = []
                embeddings = []
                
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{file}_chunk_{i}"
                    
                    try:
                        # Llamada a AWS Bedrock (Titan)
                        vector = get_embedding(chunk)
                        
                        ids.append(chunk_id)
                        documents.append(chunk)
                        embeddings.append(vector)
                        metadatas.append({"source": file, "chunk_index": i})
                        
                        fragmentos_totales += 1
                        
                        # Pequeña pausa para no saturar la API de AWS (Rate Limit)
                        time.sleep(0.1) 
                        
                    except Exception as e:
                        print(f"❌ Error vectorizando {chunk_id}: {e}")
                
                # 3. Guardar en ChromaDB por lotes (por archivo)
                if ids:
                    collection.add(
                        documents=documents,
                        embeddings=embeddings,
                        metadatas=metadatas,
                        ids=ids
                    )
                
                archivos_procesados += 1
                print(f"✅ Guardado: {file} ({len(ids)} fragmentos)")

    print(f"🎉 ¡Ingesta completada! {archivos_procesados} archivos, {fragmentos_totales} fragmentos en la BD.")

if __name__ == "__main__":
    process_directory()