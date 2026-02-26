import streamlit as st
from src.vector_store import init_chromadb, get_or_create_collection, search_documents_dynamic
from src.bedrock_client import invoke_claude

# --- LÍMITES DE SEGURIDAD (CORTAFUEGOS DE TOKENS) ---
# Límite máximo de Bedrock es 200k. Usamos 170k para los documentos, 
# dejando 30k libres para el System Prompt, el historial de chat y la respuesta.
MAX_CONTEXT_TOKENS = 950000
CHARS_PER_TOKEN = 3.5
MAX_CONTEXT_CHARS = int(MAX_CONTEXT_TOKENS * CHARS_PER_TOKEN)

st.set_page_config(page_title="Enterprise AI - 4.5GB", layout="wide", page_icon="🧠")

# 1. Conexión a Base de Datos
@st.cache_resource
def get_database_collection():
    client = init_chromadb("/app/chroma_db")
    return get_or_create_collection(client)

try:
    collection = get_database_collection()
except Exception as e:
    st.error(f"Error conectando a ChromaDB. Detalles: {e}")
    st.stop()

st.title("🧠 IA Corporativa - Acceso Total (5 GB)")

# 2. Inicializar estados de memoria (Rediseñado para Ventana Deslizante)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Lista ordenada de diccionarios con los documentos activos
if "active_documents" not in st.session_state:
    st.session_state.active_documents = [] 
# Set rápido para evitar duplicados (Búsqueda O(1))
if "seen_ids" not in st.session_state:
    st.session_state.seen_ids = set()

# 3. Main: Historial de Chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 4. Main: Lógica de Chat y Búsqueda Dinámica
if prompt := st.chat_input("Pregunta sobre cualquier documento, persona o proyecto..."):
    
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Buscando en 4.5 GB de datos..."):
            # Traemos un máximo de 100k tokens por turno para no saturar la ventana de golpe
            docs_recuperados = search_documents_dynamic(collection, prompt, max_tokens=100_000)
            nuevos_docs_agregados = 0
            
            if docs_recuperados:
                # A) Agregar los nuevos documentos a la memoria si no están duplicados
                for doc in docs_recuperados:
                    if doc['id'] not in st.session_state.seen_ids:
                        st.session_state.seen_ids.add(doc['id'])
                        st.session_state.active_documents.append(doc)
                        nuevos_docs_agregados += 1

                # B) POLÍTICA DE DESALOJO (FIFO) - El Cortafuegos
                # Calculamos el tamaño total actual en caracteres
                total_chars = sum(len(d['texto']) for d in st.session_state.active_documents)
                
                docs_eliminados = 0
                # Mientras nos pasemos del límite, sacamos el documento más viejo (índice 0)
                while total_chars > MAX_CONTEXT_CHARS and len(st.session_state.active_documents) > 0:
                    doc_viejo = st.session_state.active_documents.pop(0)
                    st.session_state.seen_ids.remove(doc_viejo['id']) # Lo quitamos del registro
                    total_chars -= len(doc_viejo['texto'])
                    docs_eliminados += 1
                
                if docs_eliminados > 0:
                    st.toast(f"🔄 Memoria llena: Se liberaron {docs_eliminados} fragmentos antiguos para hacer espacio.")

        # PASO C: Construir el XML final al vuelo y consultar a Claude
        with st.spinner(f"Analizando contexto ({len(st.session_state.active_documents)} fragmentos en memoria)..."):
            try:
                if len(st.session_state.active_documents) == 0:
                    xml_context = "<documentos>No se encontró información relevante.</documentos>"
                else:
                    # Construimos el XML string seguro
                    xml_context = "<documentos>\n"
                    for doc in st.session_state.active_documents:
                        xml_context += f'  <documento id="{doc["id"]}" origen="{doc["origen"]}">\n'
                        xml_context += f'    {doc["texto"]}\n'
                        xml_context += f'  </documento>\n'
                    xml_context += "</documentos>"

                respuesta, c_write, c_read = invoke_claude(
                    st.session_state.messages, 
                    xml_context
                )
                
                st.markdown(respuesta)
                st.caption(f"🔍 **Búsqueda:** {nuevos_docs_agregados} nuevos. | ⚡ **Caché:** `Escritos: {c_write}` `Leídos: {c_read}`")
                
                st.session_state.messages.append({"role": "assistant", "content": respuesta})
                
            except Exception as e:
                st.error(f"Error de conexión con AWS Bedrock: {str(e)}")

# Sidebar informativo
with st.sidebar:
    st.header("📊 Estado de la Sesión")
    
    # Calculamos tokens aproximados para mostrárselo al usuario
    current_chars_ui = sum(len(d['texto']) for d in st.session_state.active_documents)
    aprox_tokens = int(current_chars_ui / CHARS_PER_TOKEN)
    porcentaje_uso = (aprox_tokens / MAX_CONTEXT_TOKENS) * 100
    
    st.metric("Fragmentos Activos", len(st.session_state.active_documents))
    st.progress(min(porcentaje_uso / 100.0, 1.0), text=f"Memoria RAM de Claude: ~{aprox_tokens:,} tokens")
    
    if st.button("Limpiar Memoria y Chat"):
        st.session_state.messages = []
        st.session_state.active_documents = []
        st.session_state.seen_ids = set()
        st.rerun()