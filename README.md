# LLM-Long-Context-Grounding

## Estructura del Proyecto

```powershell
notebooklm_clone/
├── .env                    # (Ya lo creaste) Credenciales estáticas de AWS
├── .gitignore              # Archivo crítico para no subir el .env
├── requirements.txt        # Dependencias del proyecto
├── Dockerfile              # Instrucciones para el contenedor
├── app.py                  # Punto de entrada de la UI (Streamlit)
└── src/                    # Lógica interna de la aplicación
    ├── __init__.py         # Archivo vacío. Indica a Python que 'src' es un módulo
    ├── document_processor.py # Extracción de PDF/TXT y formateo XML jerárquico
    └── bedrock_client.py   # Invocación a Claude 4.6 y configuración de Prompt Caching```
```

## Correr la aplicacion

En la carpeta raiz del proyecto (Sin utilizar docker-compose):

1. Contruir la imagen
```
docker build -t LLM-LONG-CONTEXT-GROUNDING .
```

2. Run al Contenedor e inyectar credenciales .env (en carpeta raiz):
```
docker run -p 8501:8501 --env-file .env LLM-LONG-CONTEXT-GROUNDING
```

3. Abrir el navegador e ir a:
```
http://localhost:8501
```

Con docker-compose:

1. Iniciar la aplicacion (contruir imagen y levantar contenedor):
```
docker compose up -d
```

2. Apagar la maquina:
```
docker compose down
```

3. Cuando hay error en el codigo, ver quien se conecta a la app, etc. Mostrar los logs
```
docker compose logs -f
```

4. Si modificamos el dockerfile debemos recontruir el contenedor o instalemos nuevas librerias:
```
docker compose build --no-cache
```

5. Correr el procesamiento RAG y almacenarlos en CHUNKS en la base de datos montada:
```
sudo docker compose exec llm-rag-long-content python src/ingest.py
```
