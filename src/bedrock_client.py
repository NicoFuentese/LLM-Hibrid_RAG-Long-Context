import boto3
import json
import os
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from botocore.exceptions import ClientError

# Cargar .env por si se ejecuta fuera de Docker
load_dotenv()

# Inicializamos el cliente. Boto3 toma las credenciales
# inyectadas por Docker desde el .env
bedrock_client = boto3.client(
    service_name='bedrock-runtime',
    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
)

# ID de AWS Bedrock
MODEL_ID = "us.anthropic.claude-opus-4-6-v1"

def is_throttling_error(exception):
    """Verifica si el error de AWS es un límite de tasa para poder reintentar."""
    if isinstance(exception, ClientError):
        error_code = exception.response.get('Error', {}).get('Code', '')
        return error_code == 'ThrottlingException'
    return False

def get_system_prompt_with_cache(xml_documents: str) -> list:
    system_instructions = """Eres un asistente corporativo experto. 
Tu tarea es responder a las preguntas basándote ÚNICAMENTE en los documentos proporcionados.
REGLA CRÍTICA: Debes citar la fuente usando el formato [Doc ID: <id> - <origen>]."""

    return [
        {"type": "text", "text": system_instructions},
        {
            "type": "text",
            "text": f"Contexto:\n{xml_documents}",
            "cache_control": {"type": "ephemeral"} 
        }
    ]

# Usamos Tenacity para reintentar hasta 4 veces si AWS nos frena por Throttling,
# esperando 2, 4, 8 segundos entre cada intento.
@retry(
    retry=retry_if_exception_type(ClientError),
    wait=wait_exponential(multiplier=2, min=2, max=10),
    stop=stop_after_attempt(4),
    reraise=True
)

def invoke_claude_stream(messages: list, xml_documents: str):
    """
    Invoca a Claude usando Streaming y retorna un generador para que Streamlit 
    pueda pintar la respuesta en tiempo real.
    """
    system_block = get_system_prompt_with_cache(xml_documents)
    
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "anthropic_beta": ["context-1m-2025-08-07"], #llave de long-context
        "max_tokens": 4096,
        "temperature": 0.1, # Muy baja para evitar alucinaciones
        "system": system_block,
        "messages": messages
    }

    # Usamos la versión de Streaming de la API de Bedrock
    response = bedrock_client.invoke_model_with_response_stream(
        modelId=MODEL_ID,
        body=json.dumps(payload),
        accept="application/json",
        contentType="application/json"
    )

    stream = response.get('body')

    # Variables para capturar las métricas de uso al final del stream
    cache_write = 0
    cache_read = 0

    # Procesamos el flujo de datos (chunks) en tiempo real
    if stream:
        for event in stream:
            chunk = event.get('chunk')
            if chunk:
                chunk_obj = json.loads(chunk.get('bytes').decode())
                
                # Extraer el texto a medida que llega
                if chunk_obj['type'] == 'content_block_delta':
                    yield chunk_obj['delta']['text']
                
                # Al final del stream, Anthropic envía las métricas de uso (y el caché)
                elif chunk_obj['type'] == 'message_start':
                    usage = chunk_obj.get('message', {}).get('usage', {})
                    cache_write = usage.get('cache_creation_input_tokens', 0)
                    cache_read = usage.get('cache_read_input_tokens', 0)
                    
    # Entregar las métricas como el último elemento del generador
    yield {"cache_write": cache_write, "cache_read": cache_read}
    