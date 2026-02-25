import streamlit as st
from src.document_processor import build_xml_context_parallel
from src.bedrock_client import invoke_claude

st.set_page_config(page_title="ChatBot Long-Context", layout="wide", page_icon="📚")

st.title("LLM Claude Opus 4.6 - Long Context")
st.caption("Arquitectura sin RAG: Procesamiento masivo de contexto completo con Prompt Caching en AWS Bedrock")

# Inicializar estados de memoria
if "messages" not in st.session_state:
    st.session_state.messages = []
if "xml_context" not in st.session_state:
    st.session_state.xml_context = None

# Sidebar: Carga y procesamiento
with st.sidebar:
    st.header("1. Carga de Corpus")
    uploaded_files = st.file_uploader(
        "Sube PDFs o TXTs", 
        type=['pdf', 'txt'], 
        accept_multiple_files=True
    )
    
    if st.button("Procesar y Cargar a Memoria"):
        if uploaded_files:
            # Creamos un placeholder visual para la barra de progreso
            progress_text = "Preparando procesamiento en paralelo..."
            my_bar = st.progress(0, text=progress_text)
            
            # Pasamos la barra a la función
            st.session_state.xml_context = build_xml_context_parallel(uploaded_files, progress_bar=my_bar)
            
            # Limpiamos el historial de chat si se cargan nuevos documentos
            st.session_state.messages = [] 
            
            my_bar.empty() # Borramos la barra al terminar
            st.success(f"¡{len(uploaded_files)} documentos indexados y listos en memoria!")
        else:
            st.warning("Debes subir al menos un archivo.")

# Main: Historial de Chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Main: Input de usuario y ejecución
if prompt := st.chat_input("Consulta sobre tus documentos..."):
    if not st.session_state.xml_context:
        st.error("Por favor, procesa los documentos en la barra lateral primero.")
    else:
        # 1. Imprimir y guardar pregunta del usuario
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 2. Llamada a Bedrock
        with st.chat_message("assistant"):
            with st.spinner("Analizando biblioteca documental..."):
                try:
                    respuesta, c_write, c_read = invoke_claude(
                        st.session_state.messages, 
                        st.session_state.xml_context
                    )
                    
                    st.markdown(respuesta)
                    st.caption(f"⚡ **Tokens Cacheados:** `Escritos: {c_write}` | `Leídos (con descuento): {c_read}`")
                    
                    # 3. Guardar respuesta en el historial
                    st.session_state.messages.append({"role": "assistant", "content": respuesta})
                except Exception as e:
                    st.error(f"Error de conexión con AWS Bedrock: {str(e)}")