import os
import re
import streamlit as st
from src.vector_store import init_chromadb, get_or_create_collection, search_documents_dynamic
from src.bedrock_client import invoke_claude_stream

# --- LÍMITES DE SEGURIDAD (CORTAFUEGOS DE TOKENS) ---
# Límite máximo de Bedrock es 200k. Usamos 170k para los documentos, 
# dejando 30k libres para el System Prompt, el historial de chat y la respuesta.
MAX_CONTEXT_TOKENS = 950000
CHARS_PER_TOKEN = 3.5
MAX_CONTEXT_CHARS = int(MAX_CONTEXT_TOKENS * CHARS_PER_TOKEN)

@st.cache_data
def buscar_ruta_archivo(nombre_archivo, directorio_base="/app/data"):
    """Busca recursivamente un archivo en todas las subcarpetas y retorna su ruta."""
    for root, dirs, files in os.walk(directorio_base):
        if nombre_archivo in files:
            return os.path.join(root, nombre_archivo)
    return None

st.set_page_config(page_title="Enterprise AI 1M tocken - 4.5GB", layout="wide", page_icon="🧠")

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
    
    # 1. Mostramos la pregunta del usuario
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Abrimos UN SOLO contenedor para el asistente
    with st.chat_message("assistant"):
        
        # --- PASO A y B: Búsqueda y Memoria ---
        with st.spinner("🔍 Buscando en 4.5 GB de datos..."):
            docs_recuperados = search_documents_dynamic(collection, prompt, max_tokens=100_000)
            nuevos_docs_agregados = 0
            
            if docs_recuperados:
                for doc in docs_recuperados:
                    if doc['id'] not in st.session_state.seen_ids:
                        st.session_state.seen_ids.add(doc['id'])
                        st.session_state.active_documents.append(doc)
                        nuevos_docs_agregados += 1

                # POLÍTICA DE DESALOJO (FIFO)
                total_chars = sum(len(d['texto']) for d in st.session_state.active_documents)
                docs_eliminados = 0
                while total_chars > MAX_CONTEXT_CHARS and len(st.session_state.active_documents) > 0:
                    doc_viejo = st.session_state.active_documents.pop(0)
                    st.session_state.seen_ids.remove(doc_viejo['id'])
                    total_chars -= len(doc_viejo['texto'])
                    docs_eliminados += 1
                
                if docs_eliminados > 0:
                    st.toast(f"🔄 Se liberaron {docs_eliminados} fragmentos antiguos para hacer espacio.")

        # --- PASO C: Construir XML y Consultar con Streaming ---
        if len(st.session_state.active_documents) == 0:
            xml_context = "<documentos>No se encontró información relevante.</documentos>"
        else:
            xml_context = "<documentos>\n"
            for doc in st.session_state.active_documents:
                # Nos aseguramos de limpiar la ruta para que solo quede el nombre del archivo
                nombre_limpio = os.path.basename(doc["origen"])
                xml_context += f'  <documento id="{doc["id"]}" origen="{nombre_limpio}">\n'
                xml_context += f'    {doc["texto"]}\n'
                xml_context += f'  </documento>\n'
            xml_context += "</documentos>"

        try:
            stream_generator = invoke_claude_stream(st.session_state.messages, xml_context)
            
            # Inicializamos el diccionario
            metricas_cache = {}
            
            def stream_texto_puro():
                # ELIMINAMOS la línea 'nonlocal metricas_cache'
                for chunk in stream_generator:
                    if isinstance(chunk, str):
                        yield chunk
                    elif isinstance(chunk, dict):
                        # En lugar de reasignar, ACTUALIZAMOS el diccionario existente.
                        # Esto es 100% legal en Python sin importar el scope.
                        metricas_cache.update(chunk)

            # 1. Pinta la respuesta en vivo
            respuesta_completa = st.write_stream(stream_texto_puro)
            
            # --- NUEVO: EXTRACCIÓN Y BOTONES DE DESCARGA ---
            # Buscamos todas las citas con el formato [Doc ID: ... - archivo.pdf]
            patron_citas = r'\[Doc ID: [^\]]+ - (.*?)\]'
            archivos_citados = re.findall(patron_citas, respuesta_completa)
            
            # Eliminamos duplicados por si citó el mismo PDF 3 veces
            archivos_unicos = list(set(archivos_citados))
            
            if archivos_unicos:
                # Creamos un acordeón (expander) visualmente elegante
                with st.expander("📄 Ver/Descargar documentos citados en esta respuesta"):
                    for idx, archivo in enumerate(archivos_unicos):
                        ruta_real = buscar_ruta_archivo(archivo)
                        
                        if ruta_real and os.path.exists(ruta_real):
                            with open(ruta_real, "rb") as f:
                                bytes_pdf = f.read()
                                
                                # Botón nativo de Streamlit
                                st.download_button(
                                    label=f"⬇️ Descargar: {archivo}",
                                    data=bytes_pdf,
                                    file_name=archivo,
                                    mime="application/pdf" if archivo.lower().endswith('.pdf') else "text/plain",
                                    # KEY único vital para que Streamlit no confunda los botones
                                    key=f"dl_{len(st.session_state.messages)}_{idx}" 
                                )
                        else:
                            st.warning(f"⚠️ Archivo no encontrado en el servidor: {archivo}")
            # -----------------------------------------------

            c_write = metricas_cache.get("cache_write", 0)
            c_read = metricas_cache.get("cache_read", 0)
            st.caption(f"📚 **Contexto:** {nuevos_docs_agregados} nuevos docs. | ⚡ **Caché:** `Escritos: {c_write}` `Leídos: {c_read}`")
            
            st.session_state.messages.append({"role": "assistant", "content": respuesta_completa})
            
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