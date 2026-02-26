# LLM con metodologia RAG + Long-Context Grounding

## Descripcion
Una solución de Inteligencia Artificial corporativa diseñada para procesar, buscar y conversar con repositorios masivos de conocimiento (probado con 4.5 GB de documentos). Combina la precisión de la búsqueda matemática con la capacidad analítica de los modelos fundacionales más avanzados del mundo (AWS Bedrock).

## Que hace el software?
En las organizaciones, miles de horas se pierden buscando cláusulas en contratos, verificando normativas en manuales o cruzando datos entre decenas de reportes PDF. Este software transforma su repositorio estático de documentos en un experto corporativo interactivo.

En lugar de usar palabras clave en un buscador tradicional que arroja cientos de PDFs que un humano debe leer, este sistema:
1. Lee y entiende su pregunta en lenguaje natural.
2. Extrae instantáneamente la información relevante de miles de páginas.
3. Analiza los datos cruzados utilizando razonamiento profundo.
4. Responde redactando una solución clara y citando la fuente exacta (con un botón para descargar el documento original).

Resultado: Decisiones más rápidas, mitigación de riesgos operativos y democratización del conocimiento técnico interno.

## Conceptos basicos
Para operar y escalar este sistema, es crucial entender las siguientes piezas arquitectónicas:

* **Bases de Datos Vectoriales (ChromaDB)**: El corazón de la memoria a largo plazo. A diferencia de una base de datos relacional (SQL) que busca palabras exactas, una BD Vectorial almacena "Embeddings" (listas de números que representan el significado de un texto). Esto permite realizar búsquedas por proximidad matemática (similitud de coseno), encontrando respuestas aunque el usuario use sinónimos o frases distintas a las del documento original.
* **Vectorización (Embeddings)**: Usamos el modelo Amazon Titan para convertir texto en vectores matemáticos. Textos con significado similar tienen vectores cercanos. Esto nos permite encontrar información por "concepto" y no solo por coincidencia de palabras.
* **Búsqueda Híbrida Automática**: El sistema detecta automáticamente si el usuario busca un concepto general (búsqueda vectorial) o un nombre propio/acrónimo (búsqueda de texto exacto), combinando ambos mundos sin fricción.
* **Ventana Deslizante (Sliding Window FIFO)**: La memoria de la IA no es infinita. Si el usuario hace muchas preguntas, el sistema expulsa inteligentemente los documentos más antiguos del contexto para hacer espacio a los nuevos, evitando que la API de AWS colapse por exceso de tokens.
* **Prompt Caching (Solo Claude)**: Al mantener los documentos anclados en el contexto, AWS Bedrock reconoce la información repetida y aplica un descuento masivo en el costo de lectura, permitiendo conversaciones largas sobre los mismos PDFs a una fracción del precio.
* **Modelos de Razonamiento (DeepSeek R1)**: A diferencia de los LLMs tradicionales, DeepSeek R1 genera una cadena de pensamiento oculta (<think>) antes de responder. Requiere un límite de contexto artificialmente más bajo (ej. 100k tokens) para dejarle espacio "en blanco" para pensar.

## Metodología: RAG + Long-Context Grounding
Este proyecto implementa Long-Context Grounding, un enfoque de nueva generación:
* **Búsqueda Masiva**: En lugar de extraer 3 fragmentos, extraemos cientos de fragmentos relevantes de forma simultánea.
* **Inyección de Contexto**: Llenamos la ventana de memoria del LLM con hasta 100,000 - 900,000 tokens de contexto puro por interacción.
* **Delegación Cognitiva**: Confiamos en la capacidad analítica superior de modelos como Claude Opus y DeepSeek R1 para leer ese océano de información en segundos, cruzar datos de múltiples fuentes y entregar respuestas exactas con citas referenciadas a los documentos originales.

Nuestro sistema moderniza el flujo tradicional de IA:

* **El Método Antiguo (RAG Clásico)**: El sistema buscaba los 3 párrafos más parecidos a su pregunta y le pedía a la IA que respondiera solo con eso. Si la respuesta requería cruzar 50 documentos, fallaba.
* **Metodología Hibrida propuesta**: Extraemos cientos de fragmentos relevantes simultáneamente y aprovechamos el Long-Context para inyectar hasta 900.000 tokens directos al cerebro de la IA. Modelos como DeepSeek R1 o Claude Opus leen este océano de datos, razonan sobre él y entregan una respuesta holística y milimétricamente citada.

## Arquitectura y logica (flujo de datos)
El sistema se divide en dos fases lógicas completamente desacopladas para garantizar el rendimiento y evitar bloqueos en la interfaz de usuario:

### Fase 1: Ingesta de datos (offline)
1. **Extracción y Chunking**: El script (ingest.py) recorre recursivamente la carpeta local de datos. Extrae el texto de los PDFs y lo divide en "chunks" (fragmentos) superpuestos de ~4,000 caracteres.
2. **Generación de Embeddings**: Cada fragmento se envía a AWS Bedrock (Amazon Titan) para generar su representación vectorial.
3. **Almacenamiento (Upsert)**: Los vectores, el texto original y la metadata (origen, ID) se guardan físicamente en la carpeta chroma_db/. El uso de la instrucción upsert garantiza que si el script se vuelve a ejecutar, no se dupliquen documentos existentes.

### Fase 2: Aplicacion (Tiempo real)
1. **Recepción e Intercepción**: El usuario escribe una pregunta en la UI (Streamlit). El sistema intercepta el texto y aplica Expresiones Regulares (Regex) para detectar nombres propios, códigos o acrónimos (Extracción de Entidades - NER).
2. **Búsqueda Híbrida Dinámica**: * Búsqueda Exacta: Si se detectaron nombres propios, se extraen los documentos que contienen textualmente esas palabras.
    - Búsqueda Semántica: La pregunta completa se vectoriza y se buscan los fragmentos más cercanos conceptualmente. Ambos resultados se fusionan eliminando duplicados.
3. **Enrutamiento y Control de Memoria (Router)**: Los documentos recuperados se inyectan en una memoria temporal. Si la suma de caracteres excede el límite del motor seleccionado (950k para Claude, 100k para DeepSeek), se aplica la política FIFO expulsando los fragmentos más viejos.
4. **Inferencia y Streaming**: El bloque de texto resultante se empaqueta en XML y se envía a AWS Bedrock usando la ConverseStream API. La respuesta (incluyendo el proceso de razonamiento si se usa DeepSeek) se transmite en tiempo real a la interfaz.
5. **Post-Procesamiento (Citas)**: El sistema escanea la respuesta final buscando el patrón [Doc ID: ...], localiza el archivo físico correspondiente en las carpetas y genera botones nativos de descarga para el usuario.

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

### Requisitos previos y versiones del sistema utilizados
Para garantizar la estabilidad y replicabilidad en producción, el entorno requiere las siguientes especificaciones:

* **Sistema Operativo Base:** Ubuntu 24.04.3 LTS (GNU/Linux 6.14.0-1015-aws x86_64).
* Docker Engine (>= 24.0), Docker Compose (>= 2.20).
* Python 3.11 o superior (ejecutándose dentro del contenedor).
* **Core Libraries (Python):** `streamlit==1.32.0`, `chromadb==0.4.24`, `boto3>=1.34.0`, `tenacity>=8.2.3`, `PyMuPDF`, `python-dotenv`.
* **Modelos AWS Bedrock Requeridos:**
    * *Amazon Titan Text Embeddings v2* (`amazon.titan-embed-text-v2:0`)
    * *Anthropic Claude Opus* (`us.anthropic.claude-opus-4-6-v1`)
    * *DeepSeek R1* (`us.deepseek.r1-v1:0`)

### 1. Configuracion del entorno y credenciales [Perfil DevOps / Administrador TI]
Crea un archivo llamado .env en la raíz del proyecto y agrega tus credenciales de servicio:

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

### 2. Despliegue de la infraestructura [Perfil: DevOps / Administrador TI]
El ciclo de vida de la aplicación se gestiona enteramente a través de Docker Compose.

Iniciar el contenedor de Docker. Construye la imagen (si hay cambios nuevos) y levanta el servidor web en segundo plano ("detached mode"):

```powershell
docker compose up -d --build
```

Para ver en tiempo real qué está sucediendo dentro de la aplicación o diagnosticar errores de conexión con AWS se pueden monitorear los logs:
```powershell
docker compose logs -f
```

### 3. Preparacion de datos [Perfil: Gestor de Conocimiento / Admin]
Coloca tus documentos (PDFs, TXT, etc.) dentro de la carpeta data/. El sistema soporta múltiples subcarpetas para mantener su orden organizacional.

### 4. Ejecucion de ingesta [Perfil: DevOps / Administrador TI]
*Nota: Este paso procesa la data. Solo se ejecuta la primera vez o cuando se añaden nuevos archivos a la carpeta data/.*
Ejecute el pipeline ETL para vectorizar la información. Este script leerá tus PDFs, los dividirá en fragmentos, calculará sus vectores matemáticos y los guardará en la base de datos local (ChromaDB). Gracias a la función "upsert", puedes ejecutarlo varias veces sin duplicar información.

```powershell
docker compose exec llm-rag-long-content python src/ingest.py
```

### 5. Levantar servicio [Perfil: DevOps / Administrador TI]

Levantar el contenedor y correr el servicio:
```
docker compose up -d
```

### Comandos utiles para el servicio [Perfil: DevOps / Administrador TI]

Para dar de baja el contenedor/servicio:
```
docker compose down
```

Si se necesita forzar una reconstrucción total del contenedor desde cero (ignorando la caché):
```
docker compose down
docker compose build --no-cache
docker compose up -d
```

### 6. Uso de aplicacion [Perfil: Usuario Final / Negocio]

1. Abre tu navegador web y visita:
```powershell
- Local URL: http://localhost:8501
- Network URL: http://172.18.0.2:8501
- External URL: http://96.0.49.111:8501
```

2. Seleccione su motor de IA en la barra lateral según su necesidad (Razonamiento profundo vs. Lectura masiva rápida).
3. Haga preguntas naturales a la plataforma.
4. Haga clic en los botones dinámicos inferiores para visualizar y descargar los documentos fuente originales.

## Casos de uso y Escalabilidad

### ¿Dónde se aplica este software?
* **Legal y Compliance**: Auditoría de contratos, búsqueda de cláusulas específicas en licitaciones pasadas.
* **Ingeniería y Operaciones**: Consultas instantáneas sobre manuales de mantenimiento, normativas de seguridad (ej. procedimientos eléctricos) y fichas técnicas.
* **Recursos Humanos**: Asistente de políticas internas, reglamentos y beneficios corporativos.

### Escalabilidad a futuro
La arquitectura actual (basada en Docker Compose y ChromaDB local) es perfecta para equipos medianos e implementaciones departamentales. El código está diseñado de forma modular para escalar a nivel corporativo global:
1. **Capa de Datos**: ChromaDB embebido puede ser reemplazado por Amazon OpenSearch Serverless o Pinecone para manejar Terabytes de información.
2. **Capa de Cómputo**: Los contenedores pueden migrarse de un servidor único a AWS ECS (Elastic Container Service) o Kubernetes para balancear la carga de miles de usuarios simultáneos.
3. **Capa de Autenticación**: Se puede integrar Single Sign-On (SSO / Azure AD) en la UI de Streamlit.

## Glosario
Para facilitar la comprensión técnica, definimos los conceptos clave utilizados en este sistema:

**Conceptos de IA:**
* **LLM (Large Language Model):** El "cerebro" de la IA. Es el motor que entiende el lenguaje, razona y redacta las respuestas.
* **RAG (Retrieval-Augmented Generation):** Técnica que consiste en buscar información en una base de datos privada y entregársela a la IA para que responda basándose *solo* en esos datos.
* **Vector/Embedding:** Una traducción matemática de un texto. Permite a la computadora entender que "colaborador" y "empleado" significan lo mismo, aunque se escriban diferente.
* **Long-Context:** La capacidad de los nuevos LLMs de mantener en su "memoria a corto plazo" cientos de miles de tokens en una sola interacción.

**Tecnologías de la Plataforma:**
* **Docker / Docker Compose:** Plataforma que empaqueta la aplicación y todas sus dependencias en "contenedores" aislados. Garantiza que el software funcione exactamente igual en el servidor de producción que en la computadora del desarrollador.
* **Streamlit:** Framework de Python que convierte nuestros scripts de procesamiento de datos en una aplicación web interactiva (la interfaz gráfica) de forma rápida y segura.
* **ChromaDB:** El motor de base de datos vectorial de código abierto que utilizamos para guardar y buscar los vectores matemáticos de forma local y privada.
* **AWS Bedrock:** El servicio cloud de Amazon que nos da acceso seguro a los modelos de inteligencia artificial sin exponer nuestros datos corporativos al internet público.
* **Claude Opus 4.6:** LLM desarrollado por Anthropic (disponible vía AWS). Optimizado para tareas de altísima complejidad y lectura de contextos masivos (hasta 1 millón de tokens).
* **DeepSeek R1:** LLM especializado en razonamiento profundo. Destaca por su capacidad de "pensar" y planificar su respuesta antes de escribirla.