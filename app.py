import os
import re
import streamlit as st
from src.vector_store import init_chromadb, get_or_create_collection, search_documents_dynamic
from src.bedrock_client import invoke_claude_stream
from src.deepseek_client import invoke_deepseek_stream

# -------------------------------------------------------------------
# 1. CONFIGURACIÓN INICIAL (Debe ser la PRIMERA llamada a Streamlit)
# -------------------------------------------------------------------
st.set_page_config(page_title="Enterprise AI - 1M Tokens", layout="wide", page_icon="🧠")

# -------------------------------------------------------------------
# 2. FUNCIONES CACHEABLES Y UTILIDADES
# -------------------------------------------------------------------
@st.cache_data
def buscar_ruta_archivo(nombre_archivo, directorio_base="/app/data"):
    """Busca recursivamente un archivo en todas las subcarpetas y retorna su ruta."""
    for root, dirs, files in os.walk(directorio_base):
        if nombre_archivo in files:
            return os.path.join(root, nombre_archivo)
    return None

@st.cache_resource
def get_database_collection():
    client = init_chromadb("/app/chroma_db")
    return get_or_create_collection(client)

# -------------------------------------------------------------------
# 3. BARRA LATERAL (SIDEBAR): Enrutamiento y Límites
# -------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Motor de IA")
    # Selector de modelo
    modelo_seleccionado = st.radio(
        "Selecciona el LLM:", 
        ["Claude Opus 4.6 (1M)", "DeepSeek V1 (64k)"]
    )

    # --- LÍMITES DINÁMICOS SEGÚN EL MODELO ---
    if modelo_seleccionado == "Claude Opus 4.6 (1M)":
        MAX_CONTEXT_TOKENS = 950_000
    else:
        # DeepSeek R1 soporta 128k. 
        # Fijamos en 100k para inyectar PDFs y dejamos 28k libres para 
        # su proceso de razonamiento profundo (<think>) y la respuesta.
        MAX_CONTEXT_TOKENS = 100_000

    CHARS_PER_TOKEN = 3.5
    MAX_CONTEXT_CHARS = int(MAX_CONTEXT_TOKENS * CHARS_PER_TOKEN)

# -------------------------------------------------------------------
# 4. INICIALIZACIÓN DE LA APLICACIÓN (Conexión y Memoria)
# -------------------------------------------------------------------
try:
    collection = get_database_collection()
except Exception as e:
    st.error(f"Error conectando a ChromaDB. Detalles: {e}")
    st.stop()

st.title(f"🧠 IA Corporativa - Acceso Total (4.5 GB)")
st.caption(f"Motor activo: **{modelo_seleccionado}**")

# Inicializar estados de memoria
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_documents" not in st.session_state:
    st.session_state.active_documents = [] 
if "seen_ids" not in st.session_state:
    st.session_state.seen_ids = set()

# -------------------------------------------------------------------
# 5. UI PRINCIPAL: Historial de Chat
# -------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------------------------------------------------
# 6. LÓGICA CORE: Chat, Búsqueda, Desalojo y Enrutamiento
# -------------------------------------------------------------------
if prompt := st.chat_input("Pregunta sobre cualquier documento, persona o proyecto..."):
    
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        
        # --- PASO A y B: Búsqueda y Política de Desalojo ---
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
                    st.toast(f"🔄 Motor cambiado o Memoria llena: Se liberaron {docs_eliminados} fragmentos.")

        # --- PASO C: Construir XML ---
        if len(st.session_state.active_documents) == 0:
            xml_context = "<documentos>No se encontró información relevante.</documentos>"
        else:
            xml_context = "<documentos>\n"
            for doc in st.session_state.active_documents:
                nombre_limpio = os.path.basename(doc["origen"])
                xml_context += f'  <documento id="{doc["id"]}" origen="{nombre_limpio}">\n'
                xml_context += f'    {doc["texto"]}\n'
                xml_context += f'  </documento>\n'
            xml_context += "</documentos>"

        # --- PASO D: Consulta al LLM (Enrutamiento Dinámico) ---
        try:
            if modelo_seleccionado == "Claude Opus 4.6 (1M)":
                stream_generator = invoke_claude_stream(st.session_state.messages, xml_context)
            else:
                stream_generator = invoke_deepseek_stream(st.session_state.messages, xml_context)
            
            metricas_cache = {}
            
            def stream_texto_puro():
                for chunk in stream_generator:
                    if isinstance(chunk, str):
                        yield chunk
                    elif isinstance(chunk, dict):
                        metricas_cache.update(chunk)

            # Pintar respuesta
            respuesta_completa = st.write_stream(stream_texto_puro)
            
            # --- EXTRACCIÓN Y BOTONES DE DESCARGA ---
            patron_citas = r'\[Doc ID: [^\]]+ - (.*?)\]'
            archivos_citados = re.findall(patron_citas, respuesta_completa)
            archivos_unicos = list(set(archivos_citados))
            
            if archivos_unicos:
                with st.expander("📄 Ver/Descargar documentos citados en esta respuesta"):
                    for idx, archivo in enumerate(archivos_unicos):
                        ruta_real = buscar_ruta_archivo(archivo)
                        
                        if ruta_real and os.path.exists(ruta_real):
                            with open(ruta_real, "rb") as f:
                                bytes_pdf = f.read()
                                st.download_button(
                                    label=f"⬇️ Descargar: {archivo}",
                                    data=bytes_pdf,
                                    file_name=archivo,
                                    mime="application/pdf" if archivo.lower().endswith('.pdf') else "text/plain",
                                    key=f"dl_{len(st.session_state.messages)}_{idx}" 
                                )
                        else:
                            st.warning(f"⚠️ Archivo no encontrado en el servidor: {archivo}")
            
            # Métricas
            c_write = metricas_cache.get("cache_write", 0)
            c_read = metricas_cache.get("cache_read", 0)
            st.caption(f"📚 **Contexto:** {nuevos_docs_agregados} nuevos docs. | ⚡ **Caché:** `Escritos: {c_write}` `Leídos: {c_read}`")
            
            st.session_state.messages.append({"role": "assistant", "content": respuesta_completa})
            
        except Exception as e:
            st.error(f"Error en la API del LLM: {str(e)}")

# -------------------------------------------------------------------
# 7. SIDEBAR: Métricas y Limpieza
# -------------------------------------------------------------------
with st.sidebar:
    st.divider()
    st.header("📊 Estado de la Sesión")
    
    current_chars_ui = sum(len(d['texto']) for d in st.session_state.active_documents)
    aprox_tokens = int(current_chars_ui / CHARS_PER_TOKEN)
    porcentaje_uso = (aprox_tokens / MAX_CONTEXT_TOKENS) * 100
    
    st.metric("Fragmentos Activos", len(st.session_state.active_documents))
    st.progress(min(porcentaje_uso / 100.0, 1.0), text=f"Memoria RAM del LLM: ~{aprox_tokens:,} tokens")
    
    if st.button("Limpiar Memoria y Chat"):
        st.session_state.messages = []
        st.session_state.active_documents = []
        st.session_state.seen_ids = set()
        st.rerun()