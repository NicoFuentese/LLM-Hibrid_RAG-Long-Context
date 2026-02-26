import boto3
import os
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

bedrock_client = boto3.client(
    service_name='bedrock-runtime',
    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
)

MODEL_ID_DEEPSEEK = "us.deepseek.r1-v1:0"

def invoke_deepseek_stream(messages: list, xml_documents: str):
    """
    Invoca a DeepSeek R1 en AWS Bedrock capturando tanto su razonamiento como su texto.
    """
    system_instructions = """Eres un asistente corporativo experto. 
Tu tarea es responder a las preguntas basándote ÚNICAMENTE en los documentos proporcionados.
REGLA CRÍTICA: Debes citar la fuente usando el formato [Doc ID: <id> - <origen>]."""

    system_prompts = [{"text": system_instructions}]
    
    formatted_messages = []
    for msg in messages:
        formatted_messages.append({
            "role": msg["role"],
            "content": [{"text": msg["content"]}]
        })
        
    if formatted_messages and formatted_messages[-1]["role"] == "user":
        pregunta_original = formatted_messages[-1]["content"][0]["text"]
        formatted_messages[-1]["content"][0]["text"] = f"Contexto Documental:\n{xml_documents}\n\nPregunta del usuario: {pregunta_original}"

    try:
        response = bedrock_client.converse_stream(
            modelId=MODEL_ID_DEEPSEEK,
            messages=formatted_messages,
            system=system_prompts,
            inferenceConfig={
                "maxTokens": 4096,
                "temperature": 0.1 
            }
        )
        
        stream = response.get('stream')
        if stream:
            for event in stream:
                if 'contentBlockDelta' in event:
                    delta = event['contentBlockDelta']['delta']
                    
                    # 1. Si Bedrock nos envía texto final, lo capturamos
                    if 'text' in delta:
                        yield delta['text']
                        
                    # 2. LA SOLUCIÓN: Si Bedrock nos envía el "pensamiento" interno
                    elif 'reasoningContent' in delta:
                        reasoning_block = delta['reasoningContent']
                        # Bedrock anida el texto bajo 'reasoningText'
                        if 'reasoningText' in reasoning_block and 'text' in reasoning_block['reasoningText']:
                            # Pintamos el proceso mental para que puedas leerlo en Streamlit
                            yield reasoning_block['reasoningText']['text']
                            
        yield {"cache_write": 0, "cache_read": 0}
        
    except ClientError as e:
        error_msg = e.response.get('Error', {}).get('Message', str(e))
        raise Exception(f"Rechazo de AWS Bedrock (DeepSeek): {error_msg}")
    except Exception as e:
        raise Exception(f"Error general en DeepSeek: {str(e)}")