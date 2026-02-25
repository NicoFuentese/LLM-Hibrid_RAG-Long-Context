import streamlit as st
from src.vector_store import init_chromadb, get_or_create_collection, search_documents_dynamic
from src.bedrock_client import invoke_claude

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

#verificacion de base de datos
try:
    collection = get_database_collection()
    # NUEVO: Bloque de diagnóstico
    total_chunks = collection.count()
    st.sidebar.success(f"📦 Base de datos activa: {total_chunks} fragmentos indexados.")
except Exception as e:
    st.error(f"Error conectando a ChromaDB. Detalles: {e}")
    st.stop()

st.title("🧠 IA Corporativa - Acceso Total (4.5 GB)")
st.caption("RAG Dinámico: Haz cualquier pregunta. La IA buscará en tus carpetas en tiempo real.")

# 2. Inicializar estados de memoria
if "messages" not in st.session_state:
    st.session_state.messages = []
# doc_memory guardará los textos únicos que la IA va leyendo durante la conversación
if "chunk_ids_memory" not in st.session_state:
    st.session_state.chunk_ids_memory = set() 
if "xml_context" not in st.session_state:
    st.session_state.xml_context = "<documentos>\n"

# 3. Main: Historial de Chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 4. Main: Lógica de Chat y Búsqueda Dinámica
if prompt := st.chat_input("Pregunta sobre cualquier documento, persona o proyecto..."):
    
    # Mostrar pregunta del usuario
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        # PASO A: Buscar en ChromaDB automáticamente
        with st.spinner("Buscando en 4.5 GB de datos..."):
            # Traemos un volumen moderado (ej. 100k tokens) para ser rápidos en cada turno
            docs_recuperados = search_documents_dynamic(collection, prompt, max_tokens=100_000)
            
            nuevos_docs_agregados = 0
            
            # PASO B: Actualizar la Memoria Acumulativa (Deduplicación estricta por ID)
            if docs_recuperados:
                # Quitamos la etiqueta de cierre temporalmente para agregar más info
                st.session_state.xml_context = st.session_state.xml_context.replace("</documentos>", "")
                
                for doc in docs_recuperados:
                    chunk_id = doc.get('id')
                    texto = doc.get('texto', '')
                    origen = doc.get('origen', 'Desconocido')

                    # BARRERA DE DEDUPLICACIÓN: Si el ID ya existe, lo ignoramos por completo
                    if chunk_id not in st.session_state.chunk_ids_memory:
                        st.session_state.chunk_ids_memory.add(chunk_id)

                        st.session_state.xml_context += f'  <documento id="{chunk_id}" origen="{origen}">\n'
                        st.session_state.xml_context += f'    {texto}\n'
                        st.session_state.xml_context += f'  </documento>\n'
                        nuevos_docs_agregados += 1
                    
                st.session_state.xml_context += "</documentos>"

        # PASO C: Consultar a Claude 4.6
        with st.spinner(f"Analizando contexto (Leyendo {nuevos_docs_agregados} documentos nuevos)..."):
            try:
                # Si no hay NADA en memoria aún, le avisamos al modelo
                if len(st.session_state.chunk_ids_memory) == 0:
                    contexto_a_enviar = "<documentos>No se encontró información relevante para esta pregunta.</documentos>"
                else:
                    contexto_a_enviar = st.session_state.xml_context

                respuesta, c_write, c_read = invoke_claude(
                    st.session_state.messages, 
                    contexto_a_enviar
                )
                
                st.markdown(respuesta)
                
                # Feedback visual de operaciones
                st.caption(f"🔍 **Búsqueda:** {nuevos_docs_agregados} fragmentos nuevos encontrados. | ⚡ **Caché:** `Escritos: {c_write}` `Leídos: {c_read}`")
                
                st.session_state.messages.append({"role": "assistant", "content": respuesta})
                
            except Exception as e:
                st.error(f"Error de conexión con AWS Bedrock: {str(e)}")

# Sidebar informativo (Opcional, solo para que veas qué está pasando por debajo)
with st.sidebar:
    st.header("📊 Estado de la Sesión")
    st.metric("Documentos en Memoria", len(st.session_state.chunk_ids_memory))
    if st.button("Limpiar Memoria y Chat"):
        st.session_state.messages = []
        st.session_state.chunk_ids_memory = set()
        st.session_state.xml_context = "<documentos>\n"
        st.rerun()