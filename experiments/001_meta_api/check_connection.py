
import os
import sys

import httpx
from dotenv import load_dotenv
from openai import OpenAI, APIStatusError, APIConnectionError

BASE_URL = "https://api.meta.ai/v1"
MODEL = "muse-spark-1.3"


def main():
    load_dotenv()

    api_key = os.getenv("MODEL_API_KEY")

    if not api_key:
        print("ERROR: No se encontró MODEL_API_KEY en .env")
        sys.exit(1)

    print("=== Experimento 001: Meta Model API ===")

    # Prueba 1: consultar los modelos disponibles
    print("\n[1/2] Consultando modelos disponibles...")

    try:
        response = httpx.get(
            f"{BASE_URL}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=20.0,
        )

        print(f"HTTP status: {response.status_code}")
        response.raise_for_status()

        data = response.json()

        for model in data.get("data", []):
            print(" -", model.get("id", "Sin identificador"))

    except httpx.HTTPStatusError as exc:
        print("Error HTTP:", exc.response.status_code)
        print("Detalle:", exc.response.text[:1000])

    except httpx.RequestError as exc:
        print("Error de conexión:", str(exc))

    # Prueba 2: inferencia mínima
    print("\n[2/2] Probando inferencia con Muse Spark...")

    try:
        client = OpenAI(
            base_url=BASE_URL,
            api_key=api_key,
            timeout=30.0,
            max_retries=0,
        )

        result = client.responses.create(
            model=MODEL,
            input="Reply with exactly: Bakery Intelligence API OK",
            max_output_tokens=256,
        )

        print("\n=== DIAGNÓSTICO ===")
        print("Estado:", result.status)
        print("Detalles de incompleto:", result.incomplete_details)
        print("Contenido:", repr(result.output_text))
        print("Objetos de salida:", result.output)
        print("Uso:", result.usage.model_dump() if result.usage else None)

        print("Modelo:", result.model)
        print("Respuesta:", result.output_text)

        if result.usage:
            print("Uso:", result.usage.model_dump())

        print("\nRESULTADO: La API respondió correctamente.")

    except APIStatusError as exc:
        print("HTTP status:", exc.status_code)
        print("Detalle:", str(exc)[:1000])
        print("\nLa API rechazó la solicitud.")
        print("Revisa si el error corresponde a permisos, modelo o facturación.")

    except APIConnectionError as exc:
        print("Error de conexión:", str(exc))

    except Exception as exc:
        print("Error inesperado:", type(exc).__name__, str(exc)[:500])


if __name__ == "__main__":
    main()
