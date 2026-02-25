import fitz  # PyMuPDF
import re
import concurrent.futures

def clean_text(text: str) -> str:
    """Elimina caracteres de control nulos o problemáticos que puedan romper el JSON payload."""
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

def extract_text(file_bytes: bytes, filename: str) -> str:
    """Extrae texto de PDFs y TXTs de forma eficiente."""
    text = ""
    if filename.lower().endswith('.pdf'):
        # Abrir el PDF desde memoria
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page in doc:
            text += page.get_text("text") + "\n"
        doc.close()
    elif filename.lower().endswith('.txt'):
        text = file_bytes.decode('utf-8')
    
    return clean_text(text)

def extract_text_worker(file_data: tuple) -> str:
    """
    Función 'worker' que se ejecutará en paralelo.
    file_data es una tupla: (indice, nombre_archivo, bytes_del_archivo)
    """
    idx, filename, file_bytes = file_data
    text = ""
    
    try:
        if filename.lower().endswith('.pdf'):
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page in doc:
                text += page.get_text("text") + "\n"
            doc.close()
        elif filename.lower().endswith('.txt'):
            text = file_bytes.decode('utf-8')
            
        cleaned_text = clean_text(text)
        
        # Formatear directamente como XML aquí para ahorrar memoria
        xml_chunk = f'  <documento id="{idx}" nombre="{filename}">\n'
        xml_chunk += f'    {cleaned_text}\n'
        xml_chunk += f'  </documento>\n'
        
        return xml_chunk
    except Exception as e:
        # Si un archivo falla, lo reportamos dentro del XML para que el LLM lo sepa
        return f'  <documento id="{idx}" nombre="{filename}">Error procesando archivo: {str(e)}</documento>\n'

def build_xml_context_parallel(uploaded_files, progress_bar=None) -> str:
    """
    Procesa múltiples archivos en paralelo utilizando ThreadPoolExecutor.
    """
    xml_output = "<documentos>\n"
    
    # Preparamos los datos extrayendo los bytes en el hilo principal
    # st.file_uploader devuelve objetos que a veces fallan al pasar entre hilos
    tasks = []
    for idx, file in enumerate(uploaded_files, start=1):
        tasks.append((idx, file.name, file.read()))
    
    total_files = len(tasks)
    processed_files = 0
    
    # Usamos ThreadPoolExecutor para procesar en paralelo
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # map garantiza que los resultados vuelvan en el mismo orden de entrada
        results = executor.map(extract_text_worker, tasks)
        
        for xml_chunk in results:
            xml_output += xml_chunk
            processed_files += 1
            
            # Actualizamos la barra de progreso en la UI si se proporcionó
            if progress_bar:
                progress = int((processed_files / total_files) * 100)
                progress_bar.progress(progress, text=f"Procesando: {processed_files}/{total_files} archivos...")

    xml_output += "</documentos>"
    return xml_output