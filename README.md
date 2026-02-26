# LLM con metodologia RAG + Long-Context Grounding
Este proyecto es una aplicación de inteligencia artificial diseñada para procesar, buscar y chatear con bases de conocimiento masivas (ej. 4.5 GB de documentos PDF). Utiliza Amazon Bedrock como motor de inferencia, permitiendo alternar dinámicamente entre modelos de contexto masivo (Claude Opus 4.6) y modelos de razonamiento profundo (DeepSeek R1).

## Conceptos basicos
Para operar y escalar este sistema, es crucial entender las siguientes piezas arquitectónicas:

- Bases de Datos Vectoriales (ChromaDB): El corazón de la memoria a largo plazo. A diferencia de una base de datos relacional (SQL) que busca palabras exactas, una BD Vectorial almacena "Embeddings" (listas de números que representan el significado de un texto). Esto permite realizar búsquedas por proximidad matemática (similitud de coseno), encontrando respuestas aunque el usuario use sinónimos o frases distintas a las del documento original.
- Vectorización (Embeddings): Usamos el modelo Amazon Titan para convertir texto en vectores matemáticos. Textos con significado similar tienen vectores cercanos. Esto nos permite encontrar información por "concepto" y no solo por coincidencia de palabras.
- Búsqueda Híbrida Automática: El sistema detecta automáticamente si el usuario busca un concepto general (búsqueda vectorial) o un nombre propio/acrónimo (búsqueda de texto exacto), combinando ambos mundos sin fricción.
- Ventana Deslizante (Sliding Window FIFO): La memoria de la IA no es infinita. Si el usuario hace muchas preguntas, el sistema expulsa inteligentemente los documentos más antiguos del contexto para hacer espacio a los nuevos, evitando que la API de AWS colapse por exceso de tokens.
- Prompt Caching (Solo Claude): Al mantener los documentos anclados en el contexto, AWS Bedrock reconoce la información repetida y aplica un descuento masivo en el costo de lectura, permitiendo conversaciones largas sobre los mismos PDFs a una fracción del precio.
- Modelos de Razonamiento (DeepSeek R1): A diferencia de los LLMs tradicionales, DeepSeek R1 genera una cadena de pensamiento oculta (<think>) antes de responder. Requiere un límite de contexto artificialmente más bajo (ej. 100k tokens) para dejarle espacio "en blanco" para pensar.

## Metodología: RAG + Long-Context Grounding
Históricamente, los sistemas de IA usaban RAG (Retrieval-Augmented Generation) tradicional: buscaban los 3 fragmentos más relevantes de una base de datos y se los daban a leer a la IA.

Este proyecto implementa Long-Context Grounding, un enfoque de nueva generación:
- Búsqueda Masiva: En lugar de extraer 3 fragmentos, extraemos cientos de fragmentos relevantes de forma simultánea.
- Inyección de Contexto: Llenamos la ventana de memoria del LLM con hasta 100,000 - 900,000 tokens de contexto puro por interacción.
- Delegación Cognitiva: Confiamos en la capacidad analítica superior de modelos como Claude Opus y DeepSeek R1 para leer ese océano de información en segundos, cruzar datos de múltiples fuentes y entregar respuestas exactas con citas referenciadas a los documentos originales.

## Arquitectura y logica (flujo de datos)
El sistema se divide en dos fases lógicas completamente desacopladas para garantizar el rendimiento y evitar bloqueos en la interfaz de usuario:

### Fase 1: Ingesta de datos (offline)
1. Extracción y Chunking: El script (ingest.py) recorre recursivamente la carpeta local de datos. Extrae el texto de los PDFs y lo divide en "chunks" (fragmentos) superpuestos de ~4,000 caracteres.
2. Generación de Embeddings: Cada fragmento se envía a AWS Bedrock (Amazon Titan) para generar su representación vectorial.
3. Almacenamiento (Upsert): Los vectores, el texto original y la metadata (origen, ID) se guardan físicamente en la carpeta chroma_db/. El uso de la instrucción upsert garantiza que si el script se vuelve a ejecutar, no se dupliquen documentos existentes.

### Fase 2: Aplicacion (Tiempo real)
1. Recepción e Intercepción: El usuario escribe una pregunta en la UI (Streamlit). El sistema intercepta el texto y aplica Expresiones Regulares (Regex) para detectar nombres propios, códigos o acrónimos (Extracción de Entidades - NER).
2. Búsqueda Híbrida Dinámica: * Búsqueda Exacta: Si se detectaron nombres propios, se extraen los documentos que contienen textualmente esas palabras.
    - Búsqueda Semántica: La pregunta completa se vectoriza y se buscan los fragmentos más cercanos conceptualmente. Ambos resultados se fusionan eliminando duplicados.
3. Enrutamiento y Control de Memoria (Router): Los documentos recuperados se inyectan en una memoria temporal. Si la suma de caracteres excede el límite del motor seleccionado (950k para Claude, 100k para DeepSeek), se aplica la política FIFO expulsando los fragmentos más viejos.
4. Inferencia y Streaming: El bloque de texto resultante se empaqueta en XML y se envía a AWS Bedrock usando la ConverseStream API. La respuesta (incluyendo el proceso de razonamiento si se usa DeepSeek) se transmite en tiempo real a la interfaz.
5. Post-Procesamiento (Citas): El sistema escanea la respuesta final buscando el patrón [Doc ID: ...], localiza el archivo físico correspondiente en las carpetas y genera botones nativos de descarga para el usuario.

## Estructura del Proyecto

```powershell
llm-rag-long-content/
├── .env                    # Credenciales de AWS (Bedrock). NO SUBIR A GIT.
├── .gitignore              # Archivo crítico para excluir .env, carpetas de datos y caché.
├── requirements.txt        # Dependencias de Python (boto3, streamlit, chromadb, etc.)
├── Dockerfile              # Instrucciones para construir la imagen base de Python.
├── docker-compose.yml      # Orquestación, mapeo de volúmenes en vivo y red.
├── app.py                  # Punto de entrada de la interfaz de usuario (Streamlit).
├── data/                   # (Volumen) Carpeta raíz donde se alojan los 4.5 GB de PDFs originales.
├── chroma_db/              # (Volumen) Almacenamiento persistente de la base de datos vectorial.
└── src/                    # Lógica interna y conectores
    ├── __init__.py         
    ├── ingest.py           # Script ETL: Extracción, Chunking e Ingesta masiva en ChromaDB.
    ├── vector_store.py     # Lógica de Búsqueda Híbrida (Exacta/Semántica) y embeddings.
    ├── document_processor.py # Procesador de los documentos
    ├── bedrock_client.py   # Conector para Claude Opus 4.6 (Contexto 1M y Streaming).
    └── deepseek_client.py  # Conector para DeepSeek R1 (Converse API y extracción de <think>).
```
## Intalacion y ejecucion

### 1. Requisitos previos
- Docker y Docker Compose instalados.
- Cuenta en AWS con acceso a Amazon Bedrock.
- Modelos habilitados en Bedrock: Amazon Titan Embeddings v2, Anthropic Claude Opus 4.6 y DeepSeek R1.

### 2. Configuracion del entorno
Crea un archivo llamado .env en la raíz del proyecto y agrega tus credenciales:

```powershell
#claude
AWS_ACCESS_KEY_ID=tu_access_key
AWS_SECRET_ACCESS_KEY=tu_secret_key

#region
AWS_DEFAULT_REGION=us-east-1

#deepseek
DEEPSEEK_API_KEY=api-key-deepseek
MODEL_ID_DEEPSEEK = model-usado-deepseek
```

### 3. Preparacion de datos
Coloca tus documentos (PDFs, TXT, etc.) dentro de la carpeta data/. El sistema soporta múltiples subcarpetas.

### 4. Contruccion  despliegue de la infraestructura
Inicia el contenedor de Docker en modo "detached" (segundo plano). Esto levantará la aplicación web de Streamlit y mapeará los volúmenes de tu disco duro.

```powershell
docker compose up -d --build
```

Levantar el contenedor y correr el servicio:
```
docker compose up -d
```

Para dar de baja el contenedor/servicio:
```
docker compose down
```

Si se requiere revisar los logs del contenedor:
```
docker compose logs -f
```

### 5. Fase de ingesta (Solo se hace una vez o al agregar nuevos PDFs)
Ejecuta el pipeline de procesamiento masivo. Este script leerá tus PDFs, los dividirá en fragmentos, calculará sus vectores matemáticos y los guardará en la base de datos local (ChromaDB). Gracias a la función "upsert", puedes ejecutarlo varias veces sin duplicar información.

```powershell
docker compose exec llm-rag-long-content python src/ingest.py
```

### 6. Uso de aplicacion
Abre tu navegador web y visita:

```powershell
- Local URL: http://localhost:8501
- Network URL: http://172.18.0.2:8501
- External URL: http://96.0.49.111:8501
```
