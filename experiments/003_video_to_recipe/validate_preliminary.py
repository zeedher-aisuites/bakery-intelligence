"""Validate the preliminary human-review record (not verified ground truth)."""
import argparse
import json
from pathlib import Path


def validate(data: dict) -> list[str]:
    errors = []
    if data.get('record_type') != 'preliminary_human_review':
        errors.append('record_type must be preliminary_human_review')
    if data.get('annotation_status') != 'preliminary_not_frame_verified':
        errors.append('annotation_status must preserve preliminary status')
    if data.get('scale_unit') is not None:
        errors.append('scale_unit must remain null until visually verified')
    duration = data.get('duration_seconds')
    if not isinstance(duration, (int, float)) or duration <= 0:
        errors.append('duration_seconds must be positive')
        duration = 0
    ids = set()
    for segment in data.get('segments', []):
        for observation in segment.get('observations', []):
            oid = observation.get('observation_id')
            if not isinstance(oid, str) or oid in ids:
                errors.append(f'duplicate or missing observation_id: {oid}')
            ids.add(oid)
            timestamp = observation.get('approximate_timestamp_seconds')
            if timestamp is not None and (not isinstance(timestamp, (int, float)) or not 0 <= timestamp <= duration):
                errors.append(f'invalid timestamp for {oid}: {timestamp}')
            evidence = observation.get('evidence', {})
            if evidence.get('verification_status') == 'not_frame_verified':
                if evidence.get('frame_index') is not None or evidence.get('image_ref') is not None:
                    errors.append(f'unverified observation has claimed frame evidence: {oid}')
    candidate_ids = [event.get('candidate_id') for event in data.get('candidate_events', [])]
    if len(candidate_ids) != len(set(candidate_ids)):
        errors.append('duplicate candidate IDs')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    args = parser.parse_args()
    data = json.loads(args.path.read_text(encoding='utf-8'))
    errors = validate(data)
    if errors:
        for error in errors:
            print(f'ERROR: {error}')
        raise SystemExit(1)
    print(f"PASS: {len(data['segments'])} segments, {sum(len(s['observations']) for s in data['segments'])} preliminary observations, {len(data['candidate_events'])} candidate events")


if __name__ == '__main__':
    main()
