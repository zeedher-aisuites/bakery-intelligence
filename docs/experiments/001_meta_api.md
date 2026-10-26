# Experiment 001 — Meta Model API Connectivity

## Objective

Validate that Bakery Intelligence can authenticate with the Meta Model API,
list available models, and perform text inference using Muse Spark 1.3.

## Date

October 26, 2026.

## Environment

- Operating system: Windows
- Python: 3.12
- Dependency manager: uv
- SDK: openai (Python)
- API: Meta Model API
- Model: `muse-spark-1.3`
- Endpoint: `/v1/responses`

## Configuration

Prompt used:

```text
Reply with exactly: Bakery Intelligence API OK
```

Credentials: `MODEL_API_KEY` environment variable.

## Observed Results

Model catalog request:
- HTTP status: 200
- Available model identifiers: 9

Initial successful inference:
- Status: `completed`
- Response: `Bakery Intelligence API OK`
- Input tokens: 16
- Output tokens: 182
- Reasoning tokens: 168
- Total tokens: 198

Subsequent successful validation:
- Status: `completed`
- Response: `Bakery Intelligence API OK`
- Input tokens: 16
- Output tokens: 277
- Reasoning tokens: 263
- Total tokens: 293
- Process exit code: `0`

## Observations

Early tests with output limits of 40 and 256 tokens produced no visible text.
A subsequent run completed successfully and returned the expected response.

The model uses part of its output-token budget for reasoning. Token usage
varies between requests, even when the prompt is unchanged.

The validation script checks that the requested model appears in the model
catalog, the response status is `completed`, and the returned text matches
the expected value. Failures result in a nonzero exit code.

## Conclusion

Authentication, model discovery, and text inference with Muse Spark 1.3
were successfully validated. The account can access the API and usage
is reflected in the available promotional credits.

Next steps:
- Validate image inference.
- Validate video processing.
- Measure latency and cost per request.
- Add mocked tests for failure scenarios without using API credits.

## Reproduction

From the repository root:

```powershell
uv sync --locked
uv run python experiments/001_meta_api/check_connection.py
$LASTEXITCODE
```

Requires `MODEL_API_KEY` in the local `.env` file. Do not commit `.env`.
