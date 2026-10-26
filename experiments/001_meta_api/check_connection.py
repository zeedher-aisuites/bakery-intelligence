
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

        model_ids = [
            model.get("id")
            for model in data.get("data", [])
        ]

        if MODEL not in model_ids:
            print(f"ERROR: El modelo {MODEL} no está disponible.")
            sys.exit(1)

        for model in data.get("data", []):
            print(" -", model.get("id", "Sin identificador"))

    except httpx.HTTPStatusError as exc:
        print("Error HTTP:", exc.response.status_code)
        print("Detalle:", exc.response.text[:1000])
        sys.exit(1)

    except httpx.RequestError as exc:
        print("Error de conexión:", str(exc))
        sys.exit(1)

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
            max_output_tokens=1024,
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

        expected_text = "Bakery Intelligence API OK"

        if result.status != "completed":
            raise RuntimeError(
                f"Generación incompleta: {result.status}"
            )

        if result.output_text.strip() != expected_text:
            raise RuntimeError(
                f"Respuesta inesperada: {result.output_text!r}"
            )

        print("\nRESULTADO: EXPERIMENTO APROBADO")

    except APIStatusError as exc:
        print("HTTP status:", exc.status_code)
        print("Detalle:", str(exc)[:1000])
        print("\nLa API rechazó la solicitud.")
        print("Revisa si el error corresponde a permisos, modelo o facturación.")
        sys.exit(1)

    except APIConnectionError as exc:
        print("Error de conexión:", str(exc))
        sys.exit(1)

    except Exception as exc:
        print("Error inesperado:", type(exc).__name__, str(exc)[:500])
        sys.exit(1)

if __name__ == "__main__":
    main()
