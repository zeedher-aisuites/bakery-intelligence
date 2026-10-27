
import base64
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, APIStatusError, APIConnectionError


ROOT = Path(__file__).resolve().parents[2]
BASE_URL = "https://inference.baseten.co/v1"

IMAGE_PATH = (
    ROOT
    / "data/processed/experiment_003/global_1fps/frame_00018750.jpg"
)

RESULTS_DIR = Path(__file__).parent / "results"

EXPECTED_READING = "1756"


def main():
    load_dotenv(ROOT / ".env")

    api_key = os.getenv("BASETEN_API_KEY")
    model = os.getenv(
        "BASETEN_MODEL",
        "meta-models/muse-spark-open-1026",
    )

    if not api_key:
        print("ERROR: BASETEN_API_KEY was not found.")
        sys.exit(1)

    if not IMAGE_PATH.is_file():
        print(f"ERROR: Image not found: {IMAGE_PATH}")
        sys.exit(1)

    print("=== Experiment 004: Baseten Vision API ===")
    print(f"Model: {model}")
    print(f"Image: {IMAGE_PATH.name}")

    image_base64 = base64.b64encode(
        IMAGE_PATH.read_bytes()
    ).decode("ascii")

    image_url = f"data:image/jpeg;base64,{image_base64}"

    client = OpenAI(
        base_url=BASE_URL,
        api_key=api_key,
        timeout=120.0,
        max_retries=0,
    )

    prompt = (
        "Analyze this image from a cooking video. "
        "Focus on the digital kitchen scale. "
        "Report the exact number visible on its display. "
        "Do not infer ingredient identity, weight units, "
        "or quantities that are not directly visible. "
        "If the display is unreadable, explicitly say so. "
        "Reply with only the number, or UNREADABLE."
    )

    print("\nSending image to Baseten...")
    start = time.perf_counter()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                            },
                        },
                    ],
                }
            ],
            max_tokens=4096,
        )

        latency = round(
            time.perf_counter() - start,
            3,
        )

        choice = response.choices[0]
        answer = (choice.message.content or "").strip()
        finish_reason = choice.finish_reason

        usage = (
            response.usage.model_dump()
            if response.usage
            else None
        )

        # Exact comparison against a preliminary manual observation.
        reading_matches = answer == EXPECTED_READING

        result = {
            "timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "provider": "baseten",
            "model": model,
            "image": str(IMAGE_PATH.relative_to(ROOT)),
            "expected_reading_preliminary": EXPECTED_READING,
            "response": answer,
            "finish_reason": finish_reason,
            "reading_matches_preliminary": reading_matches,
            "latency_seconds": latency,
            "usage": usage,
        }

        RESULTS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = RESULTS_DIR / "vision_result.json"

        output_path.write_text(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print("\n=== DIAGNOSTICS ===")
        print("Finish reason:", finish_reason)
        print("Response:", repr(answer))
        print("Latency:", latency, "seconds")
        print("Usage:", usage)
        print("Preliminary expected reading:", EXPECTED_READING)
        print("Exact match:", reading_matches)
        print("Result saved to:", output_path)

        if finish_reason != "stop":
            print("\nRESULT: INCOMPLETE GENERATION")
            sys.exit(1)

        if not answer:
            print("\nRESULT: EMPTY RESPONSE")
            sys.exit(1)

        if answer == "UNREADABLE":
            print("\nRESULT: IMAGE ACCEPTED; READING UNREADABLE")
            return

        if reading_matches:
            print("\nRESULT: VISION READING MATCHED PRELIMINARY LABEL")
        else:
            print("\nRESULT: VISION RESPONSE REQUIRES REVIEW")

    except APIStatusError as exc:
        print("\nRESULT: API REQUEST FAILED")
        print("HTTP status:", exc.status_code)
        print("Error type:", type(exc).__name__)
        print("Details:", str(exc)[:1000])
        sys.exit(1)

    except APIConnectionError as exc:
        print("\nRESULT: CONNECTION FAILED")
        print("Details:", str(exc))
        sys.exit(1)

    except Exception as exc:
        print("\nRESULT: UNEXPECTED ERROR")
        print("Error:", type(exc).__name__, str(exc)[:500])
        sys.exit(1)


if __name__ == "__main__":
    main()