"""POST the public eval JSON to a live endpoint and score the predictions."""
import argparse
import json
import time
from typing import Dict, Tuple

import httpx


def load_eval(path: str) -> Tuple[dict, Dict[Tuple[str, str], bool]]:
    with open(path) as f:
        raw = json.load(f)

    truth = {}
    for t in raw['truth']:
        v = t['is_relevant_to_current']
        if isinstance(v, str):
            v = v.strip().lower() == 'true'
        elif not isinstance(v, bool):
            v = bool(v)
        truth[(t['case_id'], t['study_id'])] = v

    request_body = {
        'challenge_id': raw.get('challenge_id'),
        'schema_version': raw.get('schema_version'),
        'cases': raw['cases'],
    }
    return request_body, truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', required=True, help='https://your-endpoint/predict')
    ap.add_argument('--input', required=True, help='public eval JSON path')
    ap.add_argument('--timeout', type=float, default=300.0)
    args = ap.parse_args()

    print(f"Loading eval from {args.input}...")
    body, truth = load_eval(args.input)
    print(f"  Cases: {len(body['cases'])}")
    print(f"  Total priors: {sum(len(c['prior_studies']) for c in body['cases'])}")
    print(f"  Truth records: {len(truth)}")

    print(f"\nPOSTing to {args.url}...")
    t0 = time.time()
    with httpx.Client(timeout=args.timeout) as client:
        r = client.post(args.url, json=body)
    elapsed = time.time() - t0

    if r.status_code != 200:
        print(f"FAIL: HTTP {r.status_code}")
        print(r.text[:500])
        return

    response = r.json()
    preds = response.get('predictions', [])
    print(f"  HTTP 200 in {elapsed:.1f}s")
    print(f"  Predictions returned: {len(preds)}")

    # Score
    correct = 0
    incorrect = 0
    skipped = 0
    pred_keys = set()
    for p in preds:
        key = (p['case_id'], p['study_id'])
        pred_keys.add(key)
        if key not in truth:
            continue
        if p['predicted_is_relevant'] == truth[key]:
            correct += 1
        else:
            incorrect += 1

    skipped = len(truth) - len(truth.keys() & pred_keys)
    total = correct + incorrect + skipped
    acc = correct / total if total else 0.0

    print(f"\nResults:")
    print(f"  Correct:   {correct}")
    print(f"  Incorrect: {incorrect}")
    print(f"  Skipped:   {skipped} (count as incorrect)")
    print(f"  Accuracy:  {acc*100:.2f}%")

    # Gates
    if acc < 0.92:
        print(f"\nFAIL: accuracy {acc*100:.2f}% < 92% expected")
    elif skipped > 0:
        print(f"\nFAIL: {skipped} predictions were skipped")
    else:
        print(f"\nPASS: accuracy {acc*100:.2f}% with no skips")


if __name__ == '__main__':
    main()
