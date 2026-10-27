import base64
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

BASE_URL = "https://api.meta.ai/v1"
MODEL = "muse-spark-1.3"

ROOT = Path(__file__).resolve().parents[2]

IMAGE_PATH = (
    ROOT
    / "data/processed/experiment_003/global_1fps/frame_00018750.jpg"
)


def main():
    load_dotenv()

    api_key = os.getenv("MODEL_API_KEY")
    if not api_key:
        raise RuntimeError("MODEL_API_KEY is missing.")

    image_bytes = IMAGE_PATH.read_bytes()
    encoded_image = base64.b64encode(image_bytes).decode("ascii")

    client = OpenAI(
        base_url=BASE_URL,
        api_key=api_key,
        timeout=60.0,
        max_retries=0,
    )

    response = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Analyze this image from a cooking video. "
                            "Focus on the digital kitchen scale. "
                            "Report the exact number visible on its display. "
                            "Do not infer ingredient identity, weight units, "
                            "or quantities that are not directly visible. "
                            "If the display is unreadable, explicitly say so. "
                            "Reply with only the number, or UNREADABLE."
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": (
                            f"data:image/jpeg;base64,{encoded_image}"
                        ),
                    },
                ],
            }
        ],
        max_output_tokens=4096,
    )

    print("Status:", response.status)
    print("Model:", response.model)
    print("Response:", response.output_text)

    if response.usage:
        print("Usage:", response.usage.model_dump())

    if response.status != "completed":
        raise RuntimeError(
            f"Inference did not complete: {response.status}"
        )


if __name__ == "__main__":
    main()