import boto3
import json
import os
from dotenv import load_dotenv

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

def get_system_prompt_with_cache(xml_documents: str) -> list:
    """
    Construye el System Prompt en formato de array para aplicar el caché
    específicamente al bloque pesado de los documentos.
    """
    system_instructions = """Eres un asistente de investigación experto. 
Tu tarea es responder a las preguntas basándote ÚNICAMENTE en los documentos adjuntos.
REGLA CRÍTICA: Debes citar obligatoriamente la fuente de cada afirmación usando el formato [Doc ID: <id> - <nombre>].
Si la respuesta no está en el texto provisto, indica claramente que no tienes esa información."""

    return [
        {
            "type": "text",
            "text": system_instructions
        },
        {
            "type": "text",
            "text": f"Contexto documental:\n{xml_documents}",
            # MANTENER EL LONG CONTEXT EN CACHE
            "cache_control": {"type": "ephemeral"} 
        }
    ]

def invoke_claude(messages: list, xml_documents: str):
    """
    Envía el historial de mensajes y el contexto cacheable a Claude vía Bedrock.
    Retorna la respuesta y las métricas de caché.
    """
    system_block = get_system_prompt_with_cache(xml_documents)
    
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "temperature": 0.1, # Muy baja para evitar alucinaciones
        "system": system_block,
        "messages": messages
    }

    response = bedrock_client.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(payload),
        accept="application/json",
        contentType="application/json"
    )
    
    response_body = json.loads(response.get('body').read())
    
    # Extraer texto de la respuesta
    assistant_text = response_body['content'][0]['text']
    
    # Extraer métricas para validar que el caché funcionó
    usage = response_body.get("usage", {})
    cache_write = usage.get("cache_creation_input_tokens", 0)
    cache_read = usage.get("cache_read_input_tokens", 0)
    
    return assistant_text, cache_write, cache_read