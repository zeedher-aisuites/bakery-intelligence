import base64
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

BASE_URL = "https://api.meta.ai/v1"
MODEL = "muse-spark-1.3"

IMAGE_PATH = (
    Path(__file__).resolve().parent
    / "images"
    / "scale_001.png"
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
                            "Read the digital kitchen scale display. "
                            "Return the weight and unit shown. "
                            "If the display is unreadable, say so. "
                            "Do not guess."
                        ),
                    },
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