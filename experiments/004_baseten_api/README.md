# Experiment 004 — Baseten API Integration and Muse Code Configuration

## Objective

Evaluate the feasibility of using Baseten as an alternative inference provider for Bakery Intelligence and establish reproducible local development workflows with Muse Code.

The experiment covers:

- Text inference connectivity through Baseten.
- Vision inference on a real cooking-video frame.
- A preliminary comparison with Meta Model API.
- Local Muse Code initialization using either provider.

This experiment evaluates API integration and selected inference behavior. It does not evaluate end-to-end recipe reconstruction.

## Environment

- Python 3.12
- `uv` dependency management
- OpenAI Python SDK
- Windows PowerShell
- Muse Code 1.4.13

### Inference endpoints

| Provider | Model | API interface |
|---|---|---|
| Baseten | `meta-models/muse-spark-open-1026` | Chat Completions |
| Meta | `muse-spark-1.3` | Responses API |

The two configurations use different model identifiers and API interfaces. The experiment is not a controlled comparison of identical model deployments.

## Methodology

### 1. Baseten text connectivity

The connection script sends a deterministic instruction requesting an exact response.

The test checks the completion status and returned text, then records response metadata, latency, and token usage.

### 2. Baseten vision inference

A frame extracted from the EPIC-KITCHENS-100 development video `P04_104` is submitted to the Baseten model.

Local input:

`data/processed/experiment_003/global_1fps/frame_00018750.jpg`

The model is instructed to report only the number visible on the digital kitchen scale, or `UNREADABLE` when the display cannot be read.

The result is compared against a preliminary reading of `1756`.

This preliminary label is not an independently adjudicated reference annotation.

### 3. Meta vision inference

The same frame and visual-reading instruction are used with Meta Model API and `muse-spark-1.3`.

The Meta script provides a comparison point, but its output is not currently persisted in a structured experimental result file.

### 4. Muse Code provider configuration

Two PowerShell launchers were configured and interactively tested:

- `scripts/muse-meta.ps1`
- `scripts/muse-baseten.ps1`

The Meta launcher reads `MODEL_API_KEY` from the local `.env` file.

The Baseten launcher reads `BASETEN_API_KEY`, temporarily configures Muse Code to use the Baseten endpoint, and restores the original Muse Code settings and process environment when the session exits normally.

## Observed results

### Baseten text inference

The stored result in `results/connection_result.json` records:

| Measurement | Observed value |
|---|---|
| Completion | `stop` |
| Response | `Bakery Intelligence API connection successful.` |
| Latency | 1.313 seconds |
| Prompt tokens | 157 |
| Completion tokens | 243 |
| Total tokens | 400 |

**Verified outcome:** the configured Baseten endpoint completed the text request and returned the expected response.

### Baseten vision inference

The stored result in `results/vision_result.json` records:

| Measurement | Observed value |
|---|---|
| Input | `frame_00018750.jpg` |
| Preliminary expected reading | `1756` |
| Model response | `1356` |
| Completion | `stop` |
| Exact preliminary-label match | `false` |
| Latency | 7.197 seconds |
| Prompt tokens | 2,901 |
| Completion tokens | 554 |
| Total tokens | 3,455 |

**Verified outcome:** the Baseten endpoint accepted and processed the image request but returned a different number from the preliminary reference.

### Meta vision observation

During interactive testing, the Meta model returned `1756` for the same frame.

This observation was reported during development. Unlike the Baseten result, the Meta script does not currently persist an equivalent JSON record containing response metadata and latency.

It is therefore retained as a preliminary comparative observation, not a fully instrumented benchmark result.

### Muse Code integration

Interactive launch tests confirmed that:

- Muse Code initialized with the configured Baseten model.
- Muse Code initialized with the configured Meta model.
- Each provider completed a basic text-response test.
- The Baseten launcher restored the original Muse Code settings after normal exit.
- The temporary process-level API key was cleared after the validated session.

These tests establish basic development-environment connectivity. They do not establish full compatibility of coding tools, shell execution, file modification, or agent workflows with both providers.

## Reproduction

Install dependencies from the repository root:

```powershell
uv sync --locked
```

Configure the local `.env` file with the appropriate credentials:

```dotenv
MODEL_API_KEY=<meta-api-key>
BASETEN_API_KEY=<baseten-api-key>
```

An optional `BASETEN_MODEL` variable may override the default inference model used by the experiment scripts.

Do not commit `.env` or API credentials.

### Baseten text test

```powershell
uv run python experiments/004_baseten_api/check_connection.py
```

### Baseten vision test

```powershell
uv run python experiments/004_baseten_api/check_vision.py
```

### Meta vision test

```powershell
uv run python experiments/004_baseten_api/check_meta_vision.py
```

The vision tests require the referenced frame to exist locally. Source video files and generated frames are excluded from the repository.

### Muse Code launchers

From the repository root:

```powershell
.\scripts\muse-meta.ps1
```

Or:

```powershell
.\scripts\muse-baseten.ps1
```

The Baseten launcher temporarily modifies the user-level Muse Code settings file. It should not be run concurrently with other Muse Code sessions sharing that configuration. Restoration after forced process termination or system interruption has not been established.

## Experimental limitations

- One real-world frame was used for the documented vision comparison.
- The expected reading is preliminary rather than independently adjudicated ground truth.
- Different provider endpoints, model identifiers, and API interfaces were used.
- No repeated-trial accuracy distribution was measured.
- No statistical significance or general model ranking can be inferred.
- Token usage is recorded, but complete USD inference costs were not measured in these results.
- The Meta vision result lacks a persisted output artifact equivalent to the Baseten JSON record.
- Full Muse Code agent functionality through Baseten has not been validated.
- Recipe reconstruction, temporal event reasoning, ingredient attribution, and tare-aware quantity recovery were not evaluated by this experiment.

## Conclusion

Experiment 004 established basic Baseten text and image inference connectivity and demonstrated that Muse Code can be initialized with either Meta or Baseten in the local development environment.

The single-frame vision result exposes a measurement discrepancy that warrants further investigation. It does not establish a general accuracy advantage for either provider.

The integration provides a reusable experimental foundation for Bakery Intelligence while preserving the distinction between verified API behavior, preliminary model observations, and untested downstream capabilities.