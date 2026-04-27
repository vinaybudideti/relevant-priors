# Relevant Priors Endpoint — New Lantern Residency Submission

A deterministic 3-layer cascade for radiology prior-relevance classification.

## Endpoint
`POST /predict`

Live URL: https://your-app.up.railway.app/predict

## Architecture

1. **Raw pair lookup** — exact `(current_desc, prior_desc)` match with confidence-thresholded label.
2. **Canonical pair lookup** — modality + region + contrast normalization, then matched against canonical pair statistics.
3. **Logistic regression model** — 14 engineered features + char n-gram TF-IDF on the description pair.

Default fallback: `false` (verified empirically on public split — default-true gives 54%).

## Verified Performance (public split, 5 seeds)

| Split type | LR-only | Cascade |
|---|---|---|
| case-grouped | 93.06% | **93.64%** |
| current-desc holdout | 91.94% | 92.13% |
| prior-desc holdout | 92.32% | 92.44% |
| both-desc holdout | 92.55% | 92.57% |

## Local development
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.app.train --input data/relevant_priors_public.json --out artifacts/
pytest -x
USE_STUB_PREDICTOR=0 uvicorn src.app.main:app --port 8000
```

## Replaying the public eval against the live endpoint
```bash
python scripts/replay_public.py \
    --url https://your-app.up.railway.app/predict \
    --input data/relevant_priors_public.json
```
