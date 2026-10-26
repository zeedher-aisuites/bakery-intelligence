
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
        print("ERROR: MODEL_API_KEY was not found in .env")
        sys.exit(1)

    print("=== Experiment 001: Meta Model API ===")

    # Test 1: list available models
    print("\n[1/2] Listing available models...")

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
            print(f"ERROR: Model {MODEL} is not available.")
            sys.exit(1)

        for model in data.get("data", []):
            print(" -", model.get("id", "Missing identifier"))

    except httpx.HTTPStatusError as exc:
        print("HTTP error:", exc.response.status_code)
        print("Details:", exc.response.text[:1000])
        sys.exit(1)

    except httpx.RequestError as exc:
        print("Connection error:", str(exc))
        sys.exit(1)

    # Test 2: minimal inference
    print("\n[2/2] Testing inference with Muse Spark...")

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

        print("\n=== DIAGNOSTICS ===")
        print("Status:", result.status)
        print("Incomplete details:", result.incomplete_details)
        print("Content:", repr(result.output_text))
        print("Output objects:", result.output)

        print("Model:", result.model)
        print("Response:", result.output_text)

        if result.usage:
            print("Usage:", result.usage.model_dump())

        expected_text = "Bakery Intelligence API OK"

        if result.status != "completed":
            raise RuntimeError(
                f"Incomplete generation: {result.status}"
            )

        if result.output_text.strip() != expected_text:
            raise RuntimeError(
                f"Unexpected response: {result.output_text!r}"
            )

        print("\nRESULT: EXPERIMENT PASSED")

    except APIStatusError as exc:
        print("HTTP status:", exc.status_code)
        print("Details:", str(exc)[:1000])
        print("\nThe API rejected the request.")
        print("Check whether the error is related to permissions, model access, or billing.")
        sys.exit(1)

    except APIConnectionError as exc:
        print("Connection error:", str(exc))
        sys.exit(1)

    except Exception as exc:
        print("Unexpected error:", type(exc).__name__, str(exc)[:500])
        sys.exit(1)


if __name__ == "__main__":
    main()
