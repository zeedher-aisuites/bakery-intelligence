
import base64
import csv
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

EXPERIMENT_DIR = Path(__file__).resolve().parent
GROUND_TRUTH = EXPERIMENT_DIR / "ground_truth.csv"
IMAGES_DIR = EXPERIMENT_DIR / "images"
RESULTS_DIR = EXPERIMENT_DIR / "results"

MODEL = "muse-spark-1.3"
BASE_URL = "https://api.meta.ai/v1"

# Estimated USD prices per 1 million tokens.
# Update these values when the provider changes its pricing.
INPUT_PRICE_PER_MILLION = 1.25
OUTPUT_PRICE_PER_MILLION = 4.25
CACHED_INPUT_PRICE_PER_MILLION = 0.15

PROMPT = (
    "Read only the numeric weight displayed on the digital kitchen scale. "
    "Reply with the number and unit, for example: 83.6 g. "
    "If the display cannot be read confidently, reply exactly: unreadable. "
    "Do not estimate the weight from the ingredient or container. "
    "Do not explain your answer."
)


def parse_weight(text):
    if "unreadable" in text.lower():
        return None, None

    match = re.search(
        r"(?<!\w)(\d+(?:[.,]\d+)?)\s*(kg|g)\b",
        text.lower(),
    )

    if not match:
        return None, None

    value = float(match.group(1).replace(",", "."))
    unit = match.group(2)

    if unit == "kg":
        value *= 1000
        unit = "g"

    return value, unit


def read_ground_truth():
    with GROUND_TRUTH.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        return list(csv.DictReader(file))

def calculate_cost(usage):
    if usage is None:
        return None

    input_tokens = usage.input_tokens or 0
    output_tokens = usage.output_tokens or 0

    details = usage.input_tokens_details
    cached_tokens = (
        (details.cached_tokens or 0)
        if details is not None
        else 0
    )

    regular_input_tokens = max(input_tokens - cached_tokens, 0)

    cost = (
        regular_input_tokens * INPUT_PRICE_PER_MILLION
        + cached_tokens * CACHED_INPUT_PRICE_PER_MILLION
        + output_tokens * OUTPUT_PRICE_PER_MILLION
    ) / 1_000_000

    return round(cost, 8)


def evaluate_image(client, row):
    image_path = IMAGES_DIR / row["image_filename"]

    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    encoded_image = base64.b64encode(
        image_path.read_bytes()
    ).decode("ascii")

    start = time.perf_counter()

    response = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": PROMPT},
                    {
                        "type": "input_image",
                        "image_url": (
                            f"data:image/png;base64,{encoded_image}"
                        ),
                    },
                ],
            }
        ],
        max_output_tokens=1024,
    )

    latency = round(time.perf_counter() - start, 3)

    if response.status != "completed":
        raise RuntimeError(
            f"Model response status: {response.status}"
        )

    raw_response = response.output_text or ""
    predicted_value, predicted_unit = parse_weight(raw_response)

    expected_value = float(row["weight_value"])
    expected_unit = row["weight_unit"].lower()

    correct = (
        predicted_value is not None
        and predicted_unit == expected_unit
        and abs(predicted_value - expected_value) < 0.001
    )

    usage = response.usage

    estimated_cost = calculate_cost(usage)

    return {
        "image_filename": row["image_filename"],
        "ingredient": row["ingredient"],
        "expected_value": expected_value,
        "expected_unit": expected_unit,
        "raw_response": raw_response,
        "predicted_value": predicted_value,
        "predicted_unit": predicted_unit,
        "correct": correct,
        "estimated_cost_usd": estimated_cost,
        "latency_seconds": latency,
        "input_tokens": usage.input_tokens if usage else None,
        "output_tokens": usage.output_tokens if usage else None,
        "total_tokens": usage.total_tokens if usage else None,
        "status": response.status,
    }


def save_results(results, summary):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%S%fZ")

    payload = {
        "model": MODEL,
        "timestamp_utc": timestamp.isoformat(),
        "prompt": PROMPT,
        "pricing_usd_per_million_tokens": {
            "input": INPUT_PRICE_PER_MILLION,
            "cached_input": CACHED_INPUT_PRICE_PER_MILLION,
            "output": OUTPUT_PRICE_PER_MILLION,
        },
        "summary": summary,
        "results": results,
    }

    output_path = RESULTS_DIR / f"evaluation_{run_id}.json"
    latest_path = RESULTS_DIR / "evaluation.json"

    serialized = json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
    )

    output_path.write_text(serialized, encoding="utf-8")
    latest_path.write_text(serialized, encoding="utf-8")

    print(f"\nResults saved to: {output_path}")
    print(f"Latest results: {latest_path}")


def main():
    load_dotenv()

    api_key = os.getenv("MODEL_API_KEY")

    if not api_key:
        raise RuntimeError("MODEL_API_KEY is missing.")

    client = OpenAI(
        base_url=BASE_URL,
        api_key=api_key,
        timeout=60.0,
        max_retries=0,
    )

    rows = read_ground_truth()
    results = []

    print(f"Model: {MODEL}")
    print(f"Images: {len(rows)}\n")

    try:
        for index, row in enumerate(rows, start=1):
            print(
                f"[{index}/{len(rows)}] "
                f"{row['image_filename']}",
                flush=True,
            )

            try:
                result = evaluate_image(client, row)
                results.append(result)

                print(
                    f"  Expected: {result['expected_value']} "
                    f"{result['expected_unit']}"
                )
                print(f"  Predicted: {result['raw_response']}")
                print(f"  Correct: {result['correct']}")
                print(
                    f"  Latency: {result['latency_seconds']} s\n",
                    flush=True,
                )
                print(
                    f"  Estimated cost: "
                    f"${result['estimated_cost_usd']:.6f} USD"
                    if result["estimated_cost_usd"] is not None
                    else "  Estimated cost: unavailable",
                    flush=True,
                )

            except Exception as error:
                print(
                    f"  Error: {type(error).__name__}: {error}\n",
                    flush=True,
                )
                results.append({
                    "image_filename": row["image_filename"],
                    "expected_value": float(row["weight_value"]),
                    "expected_unit": row["weight_unit"],
                    "correct": False,
                    "error": str(error),
                })

    finally:
        successful = [
            result for result in results
            if "error" not in result
        ]

        correct_count = sum(
            result["correct"] for result in successful
        )

        costs = [
            result["estimated_cost_usd"]
            for result in successful
            if result["estimated_cost_usd"] is not None
        ]

        total_cost = sum(costs)

        summary = {
            "total_images": len(rows),
            "processed": len(results),
            "successful_requests": len(successful),
            "correct_predictions": correct_count,
            "accuracy_percent": round(
                100 * correct_count / len(rows), 2
            ) if rows else 0,
            "total_tokens": sum(
                result["total_tokens"] or 0
                for result in successful
            ),
            "total_estimated_cost_usd": round(total_cost, 8),

            "average_cost_per_inference_usd": (
                round(total_cost / len(costs), 8)
                if costs else None
            ),

            "cost_per_correct_prediction_usd": (
                round(total_cost / correct_count, 8)
                if correct_count and len(costs) == len(successful)
                else None
            ),
        }

        print("\n--- Evaluation Summary ---")
        print(json.dumps(summary, indent=2))

        save_results(results, summary)


if __name__ == "__main__":
    main()
