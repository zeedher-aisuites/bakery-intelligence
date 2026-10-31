"""UNVERIFIED DRAFT — Modal Volumes v2 smoke test.

DO NOT EXECUTE until Step 0 of PROCEDURE.md is complete: the Modal
Volumes API calls below were written without access to the official
documentation and may not match the current v2 surface. Verify every
call against https://modal.com/docs/guide/volumes first, correct
this file, and record the documentation revision.

Intended flow (mirrors PROCEDURE.md Steps 1-4):

1. Look up the dedicated test volume ``bakery-spike-008``.
2. Upload a small local test file.
3. From an isolated worker, read the file and return its SHA-256.
4. From the same worker, write a small JSON artifact to the volume.
"""

from __future__ import annotations

VOLUME_NAME = "bakery-spike-008"
TEST_OBJECT = "spike008/source.bin"
ARTIFACT_OBJECT = "spike008/smoke_result.json"

# Draft only: import modal lazily so offline test collection never
# requires the dependency or credentials.
#
# import hashlib
# import json
# import modal
#
# app = modal.App("bakery-spike-008")
# volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
#
#
# @app.function(volumes={"/data": volume})
# def verify_source() -> dict:
#     ...read /data/spike008/source.bin, hash it, write artifact...
#
#
# @app.local_entrypoint()
# def main() -> None:
#     ...upload local file, call verify_source.remote(), download artifact...


def main() -> int:
    """Refuse to run: this draft is unverified by design."""
    raise SystemExit(
        "UNVERIFIED DRAFT: complete PROCEDURE.md Step 0 (verify the "
        "official Modal Volumes v2 API) before executing.")


if __name__ == "__main__":
    main()
