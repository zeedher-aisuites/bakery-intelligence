
# Experimento 001 — Conexión con Meta Model API

## Objetivo

Validar que Bakery Intelligence puede autenticarse en Meta Model API,
consultar los modelos disponibles y realizar una inferencia de texto
con Muse Spark 1.3.

## Fecha

26 de octubre de 2026.

## Entorno

- Sistema operativo: Windows
- Python: 3.12
- Gestor de dependencias: uv
- SDK: openai (Python)
- API: Meta Model API
- Modelo: muse-spark-1.3
- Endpoint: /v1/responses

## Configuración

Prompt utilizado:

"Reply with exactly: Bakery Intelligence API OK"

Credenciales: variable de entorno MODEL_API_KEY.

## Resultados observados

Consulta del catálogo:
- HTTP 200
- 9 identificadores de modelos disponibles

Inferencia:
- Estado: completed
- Respuesta: Bakery Intelligence API OK
- Tokens de entrada: 16
- Tokens de salida: 182
- Tokens de razonamiento: 168
- Tokens totales: 198

## Observaciones

Las primeras pruebas, con límites de salida de 40 y 256 tokens,
no generaron texto visible.

La ejecución final terminó correctamente y produjo la respuesta esperada.

El modelo utiliza parte de su presupuesto de salida en razonamiento.

## Conclusión

Se validó la conexión y una inferencia de texto con Muse Spark 1.3.

La cuenta permite utilizar la API y registra consumo de créditos.

Pendiente:
- Validar inferencia con imágenes.
- Validar procesamiento de video.
- Medir latencia y costo por solicitud.

## Reproducción

Desde la raíz del repositorio:

uv sync --locked

uv run python experiments/001_meta_api/check_connection.py

Requiere MODEL_API_KEY en el archivo .env.
