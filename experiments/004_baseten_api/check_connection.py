
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

api_key = os.getenv("BASETEN_API_KEY")
model = os.getenv("BASETEN_MODEL", "meta-models/muse-spark-open-1026")

if not api_key:
    raise RuntimeError("BASETEN_API_KEY is missing.")

client = OpenAI(
    base_url="https://inference.baseten.co/v1",
    api_key=api_key,
    timeout=60.0,
    max_retries=0,
)

print("Provider: Baseten")
print(f"Model: {model}")
print("Sending text-only request...")

start = time.perf_counter()

try:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    "Reply with exactly this sentence: "
                    "Bakery Intelligence API connection successful."
                ),
            }
        ],
        max_tokens=1024,
    )
except Exception as exc:
    print(f"Request failed: {type(exc).__name__}: {exc}")
    raise SystemExit(1)

latency = round(time.perf_counter() - start, 3)
answer = response.choices[0].message.content or ""
finish_reason = response.choices[0].finish_reason
print(f"Finish reason: {finish_reason}")

expected = "Bakery Intelligence API connection successful."

if finish_reason != "stop":
    raise RuntimeError(
        f"Incomplete generation: finish_reason={finish_reason}"
    )

if answer.strip() != expected:
    raise RuntimeError(
        f"Unexpected response: {answer!r}"
    )

print("RESULT: EXPERIMENT PASSED")

usage = response.usage.model_dump() if response.usage else None

result = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "provider": "baseten",
    "model": model,
    "latency_seconds": latency,
    "response": answer,
    "finish_reason": finish_reason,
    "usage": usage,
}

results_dir = Path(__file__).parent / "results"
results_dir.mkdir(parents=True, exist_ok=True)

output_path = results_dir / "connection_result.json"
output_path.write_text(
    json.dumps(result, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(f"Response: {answer}")
print(f"Latency: {latency} seconds")
print(f"Token usage: {usage}")
print(f"Result saved to: {output_path}")
