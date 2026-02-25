import streamlit as st
from src.vector_store import init_chromadb, get_or_create_collection, search_documents_dynamic
from src.bedrock_client import invoke_claude

# Configuración de página
st.set_page_config(page_title="ChatBot Long-Context + RAG", layout="wide", page_icon="🏢")

# 1. Conexión a Base de Datos (Caché de Streamlit)
# Usamos @st.cache_resource para que Streamlit no abra múltiples conexiones a ChromaDB,
# evitando bloqueos de archivo (Database Locks).
@st.cache_resource
def get_database_collection():
    client = init_chromadb("/app/chroma_db")
    return get_or_create_collection(client)

try:
    collection = get_database_collection()
except Exception as e:
    st.error(f"Error conectando a ChromaDB. Asegúrate de que el volumen esté montado. Detalles: {e}")
    st.stop()

st.title("🏢 Chat Corporativo Híbrido RAG + Long Content")
st.caption("Búsqueda Vectorial Masiva + Claude 4.6 Long-Context con Prompt Caching")

# 2. Inicializar estados de memoria
if "messages" not in st.session_state:
    st.session_state.messages = []
if "xml_context" not in st.session_state:
    st.session_state.xml_context = None
if "current_topic" not in st.session_state:
    st.session_state.current_topic = None

# 3. Sidebar: Búsqueda y Anclaje de Contexto
with st.sidebar:
    st.header("🔍 1. Anclar Contexto")
    st.markdown("Busca un tema amplio. Extraeremos la información de los **4.5 GB** y la congelaremos para chatear usando caché.")
    
    tema_busqueda = st.text_input("Tema de investigación:", placeholder="Ej. Proyectos de Alfredo Martinez...")
    
    if st.button("Buscar y Extraer Contexto"):
        if tema_busqueda:
            with st.spinner("Buscando en 1,370 documentos..."):
                # Realizamos la búsqueda dinámica
                docs_recuperados = search_documents_dynamic(collection, tema_busqueda)
                
                if docs_recuperados:
                    # Construimos el XML masivo
                    xml_out = "<documentos>\n"
                    for idx, doc in enumerate(docs_recuperados, start=1):
                        origen = doc.get('origen', 'Desconocido')
                        texto = doc.get('texto', '')
                        xml_out += f'  <documento id="{idx}" origen="{origen}">\n'
                        xml_out += f'    {texto}\n'
                        xml_out += f'  </documento>\n'
                    xml_out += "</documentos>"
                    
                    # Guardamos en el estado de la sesión
                    st.session_state.xml_context = xml_out
                    st.session_state.current_topic = tema_busqueda
                    st.session_state.messages = [] # Limpiamos el chat anterior
                    
                    st.success(f"¡Éxito! {len(docs_recuperados)} fragmentos anclados a la memoria de Claude.")
                else:
                    st.warning("No se encontró información que supere el umbral de relevancia.")
        else:
            st.warning("Ingresa un tema para buscar.")

# 4. Main: Interfaz de Chat
if st.session_state.xml_context:
    st.info(f"📌 **Contexto Anclado:** Información relacionada con *'{st.session_state.current_topic}'*")
    
    # Mostrar historial de chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input de usuario
    if prompt := st.chat_input("Pregunta sobre este contexto..."):
        # Mostrar pregunta
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Llamar a Bedrock
        with st.chat_message("assistant"):
            with st.spinner("Analizando contexto anclado..."):
                try:
                    respuesta, c_write, c_read = invoke_claude(
                        st.session_state.messages, 
                        st.session_state.xml_context
                    )
                    
                    st.markdown(respuesta)
                    st.caption(f"⚡ **Tokens Cacheados:** `Escritos: {c_write}` | `Leídos (con descuento): {c_read}`")
                    
                    st.session_state.messages.append({"role": "assistant", "content": respuesta})
                except Exception as e:
                    st.error(f"Error de conexión con AWS Bedrock: {str(e)}")
else:
    st.info("👈 Comienza buscando un tema de investigación en la barra lateral para anclar los documentos relevantes.")