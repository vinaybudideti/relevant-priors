# Relevant Priors Endpoint

A production-grade FastAPI service that classifies whether each prior radiology study is **relevant** to a current study for the reading radiologist. Built for the **New Lantern Residency** challenge.

**Live endpoint**: `https://relevant-priors-production-8914.up.railway.app/predict`

> **TL;DR** — POST cases (current study + prior studies) → get one `predicted_is_relevant: bool` per prior. Single batched request handles the full public eval (27,614 priors) in **~0.9s local / 3.8s over Railway** at **95.50% accuracy**, with **zero skipped predictions**. No LLM. No database. No Redis. Just a deterministic 3-layer cascade + scikit-learn.

---

## Table of Contents

1. [Highlights](#highlights)
2. [The Problem](#the-problem)
3. [Architecture](#architecture)
4. [Verified Performance](#verified-performance)
5. [Project Structure](#project-structure)
6. [Quickstart](#quickstart)
7. [API Reference](#api-reference)
8. [Configuration](#configuration)
9. [Testing](#testing)
10. [Training the Model](#training-the-model)
11. [Local Docker](#local-docker)
12. [Deploying to Railway](#deploying-to-railway)
13. [Replay & Scoring Tool](#replay--scoring-tool)
14. [Production Considerations](#production-considerations)
15. [Future Improvements](#future-improvements)
16. [Tech Stack](#tech-stack)
17. [License](#license)

---

## Highlights

| Metric                                      | Value                              |
| ------------------------------------------- | ---------------------------------- |
| Accuracy on full public replay (train+evaluate on full public)        | **95.50%**           |
| Accuracy on case-grouped 5-seed cross-validation                      | **93.64%**           |
| Total predictions returned                  | 27,614 / 27,614 (zero skips)       |
| End-to-end inference time (full eval)       | **~0.9 s**                         |
| Container start-up to ready                 | **~3 s**                           |
| Image size                                  | ~270 MB (python:3.11-slim base)    |
| External dependencies at runtime            | **None** (no DB, no Redis, no LLM) |
| Test count                                  | 30 (27 unit + 3 cascade-integration; cascade tests skip until artifacts are trained)  |
| Lines of production code                    | ~590 across 10 modules             |

---

## The Problem

Given a radiology *case* — a current imaging study and a list of N prior studies for the same patient — decide which priors are clinically relevant for interpreting the current study. The endpoint must:

- Return **exactly one prediction per prior** (zero skips, ever).
- Tolerate evaluator schema drift (extra fields, malformed dates).
- Never log PHI (`patient_name`, `patient_id`, `mrn`).
- Default to `false` when uncertain — pure default-true achieves only 23% accuracy on the public split (relevant rate is 23.78%); even raw-pair lookup with default-true on misses only reaches 54%, which is decisively worse than default-false.

The challenge constraints rule out per-call LLM inference for cost and latency reasons.

---

## Architecture

The predictor is a **3-layer cascade** with a hard floor at `false`. Each layer is verified empirically against case-grouped 5-seed cross-validation before being included.

```
┌──────────────────────────────────────────────────────────────┐
│ POST /predict                                                │
│                                                              │
│   for each (case, prior):                                    │
│                                                              │
│   ┌───────────────────────────────────────────────┐          │
│   │ 1. RAW PAIR LOOKUP                            │          │
│   │    exact (current_desc, prior_desc) match     │ ──hit──► │
│   │    n ≥ 3 and (p ≥ 0.85 or p ≤ 0.15) → decide  │          │
│   └────────────────────┬──────────────────────────┘          │
│                        │ miss                                │
│   ┌────────────────────▼──────────────────────────┐          │
│   │ 2. CANONICAL PAIR LOOKUP                      │          │
│   │    normalize(modality | region | contrast)    │ ──hit──► │
│   │    n ≥ 10 and (p ≥ 0.90 or p ≤ 0.10) → decide │          │
│   └────────────────────┬──────────────────────────┘          │
│                        │ miss                                │
│   ┌────────────────────▼──────────────────────────┐          │
│   │ 3. LOGISTIC REGRESSION                        │          │
│   │    14 engineered features +                   │ ──hit──► │
│   │    char-wb (3,4)-gram TF-IDF on desc pair     │          │
│   │    threshold = 0.5                            │          │
│   └────────────────────┬──────────────────────────┘          │
│                        │ error                               │
│                        ▼                                     │
│                   DEFAULT_FALLBACK = false                   │
└──────────────────────────────────────────────────────────────┘
```

### Why this architecture?

The pair `(current_description, prior_description)` is **near-deterministic** of the label on the public split: 98.21% of unique pairs have 100% consistent labels, putting the oracle accuracy at 98.84%. That makes a memorization-friendly cascade with normalization fallbacks the right shape. The LR model only fires on rows where neither lookup is confident enough — typically novel descriptions where pair statistics are too sparse to override the model.

### The 14 engineered features

`same_string`, `same_modality`, `same_region`, `cross_mod_same_region`, `same_mod_diff_region`, `jaccard` (token overlap on normalized text), `gap_log` (log-days between studies), `gap_le_30`, `gap_gt_5y`, `n_priors_log`, `contrast_match`, `both_breast`, `both_chest`, `cur_chest_prior_abd`.

---

## Verified Performance

All numbers are mean across **5 deterministic seeds** on the public 996-case split.

| Split type             | LR-only | Cascade   | Cascade lift |
| ---------------------- | ------- | --------- | ------------ |
| case-grouped           | 93.06%  | **93.64%**| +0.58 pp     |
| current-desc holdout   | 91.94%  | 92.13%    | +0.19 pp     |
| prior-desc holdout     | 92.32%  | 92.44%    | +0.12 pp     |
| both-desc holdout      | 92.55%  | 92.57%    | +0.03 pp     |

The cascade lift is small but **positive on every split** (it's not just adding noise). On the public-as-train, public-as-test scenario used by `scripts/replay_public.py`, the same code achieves **95.50%** because the pair-lookup tables have full coverage of the input.

### Reading the headline numbers

This README quotes two distinct accuracy numbers that mean different things:

- **95.50%** is the full-public replay metric. The model is trained on the full 996-case public split, then the same split is replayed against the live endpoint to verify the contract end-to-end. This is what the live URL produces and what the Highlights table reports.
- **93.64%** is the case-grouped 5-seed cross-validation metric. The model is trained on 80% of cases and evaluated on the held-out 20%, repeated across five seeds. This is the more conservative estimate of generalization to unseen cases from the same distribution.

The 1.86pp gap between the two is expected: pair-statistic lookups have full coverage when train and test overlap, which lifts the cascade above its true cross-validation accuracy. Neither number is a projection of private-split accuracy, which is unknown until evaluation.

### Experiments and Findings

The architecture was locked through 7 phases of empirical verification before any production code was written. After v1 deployment, four candidate improvements were measured in isolation against the v1 baseline using a 5-seed × 4-split harness with strict per-fold cross-validation. All four methods were rejected by a strict pre-declared acceptance gate, but two model-layer methods (M3 GBT ensemble and M4 sample weighting) revealed a structural property of the cascade: case_grouped accuracy is capped at the override layer, while drift-split accuracy is capped at the model layer. The cascade absorbs model-layer improvements on case_grouped specifically. Full methodology, per-method results, performance tables, deployment incident report, and decision rationale are in [experiments.md](experiments.md).

---

## Project Structure

```
relevant-priors/
│
├── src/                              All production Python code lives here.
│   └── app/                          Single application package.
│       ├── __init__.py               (empty marker)
│       ├── main.py                   FastAPI app + /predict + /healthz + /readyz.
│       ├── schemas.py                Pydantic request/response models (lenient).
│       ├── predictor.py              Predictor protocol + AllFalsePredictor stub.
│       ├── cascade.py                CascadePredictor — the real 3-layer predictor.
│       ├── normalize.py              Description normalizer + canonical key builder.
│       ├── features.py               14-dimensional engineered feature vector.
│       ├── train.py                  Trains LR + builds pair stats → writes artifacts/.
│       ├── anti_skip.py              Anti-skip invariant guard.
│       └── logging_setup.py          PHI-safe structured JSON logging.
│
├── scripts/                          CLI utilities (NOT imported by the app).
│   ├── __init__.py                   (empty marker — makes scripts/ importable)
│   └── replay_public.py              POST public eval to a live URL, score accuracy.
│
├── tests/                            pytest tests, one file per module.
│   ├── __init__.py                   (empty marker)
│   ├── test_normalize.py             Unit tests for normalize.py
│   ├── test_features.py              Unit tests for features.py
│   ├── test_schemas.py               Unit tests for schemas.py
│   ├── test_anti_skip.py             Unit tests for anti_skip.py
│   ├── test_predictor.py             Unit tests for AllFalsePredictor
│   ├── test_cascade.py               Unit tests for CascadePredictor (auto-skip if no artifacts)
│   └── test_contract.py              FastAPI endpoint contract tests
│
├── data/                             Input data (read-only at runtime).
│   └── relevant_priors_public.json   Public eval JSON — 996 cases, 27,614 truth labels.
│
├── artifacts/                        GENERATED by train.py — gitignored.
│   ├── lr_model.joblib               Pickled scikit-learn LogisticRegression.
│   ├── tfidf_vectorizer.joblib       Pickled char-wb TfidfVectorizer.
│   ├── raw_pair_stats.json           Raw-pair lookup table.
│   └── canonical_pair_stats.json     Canonical-pair lookup table.
│
├── Dockerfile                        Single-stage build that trains artifacts inside the image via RUN python -m src.app.train.
├── railway.toml                      Railway deploy config (DOCKERFILE builder).
├── requirements.txt                  Python dependencies, version-pinned.
├── .gitignore                        Excludes artifacts/, .venv/, __pycache__/, etc.
├── .dockerignore                     Excludes .venv, .git, tests/ from image (NOT artifacts/).
├── README.md                         You are here.
└── experiments.md                    Full experimental write-up (5-seed measurements).
```

### Module dependency graph (acyclic)

```
normalize.py        →  no internal deps
features.py         →  imports normalize
schemas.py          →  no internal deps
predictor.py        →  no internal deps
anti_skip.py        →  no internal deps (duck-typed)
logging_setup.py    →  no internal deps
cascade.py          →  imports normalize, features
train.py            →  imports normalize, features
main.py             →  imports schemas, predictor, anti_skip, logging_setup
                       (lazy-imports cascade only when USE_STUB_PREDICTOR=0)
```

---

## Quickstart

Five commands to a running cascade endpoint:

```bash
git clone https://github.com/vinaybudideti/relevant-priors.git
cd relevant-priors
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train the model artifacts (≈10 s)
python -m src.app.train --input data/relevant_priors_public.json --out artifacts/

# Run the cascade endpoint
USE_STUB_PREDICTOR=0 ARTIFACTS_DIR=artifacts uvicorn src.app.main:app --port 8000
```

In a second terminal:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "cases": [{
      "case_id": "c1",
      "current_study": {
        "study_id": "s1",
        "study_description": "MRI BRAIN",
        "study_date": "2024-01-01"
      },
      "prior_studies": [
        {
          "study_id": "p1",
          "study_description": "CT HEAD",
          "study_date": "2023-01-01"
        }
      ]
    }]
  }'
```

Response:

```json
{
  "predictions": [
    {
      "case_id": "c1",
      "study_id": "p1",
      "predicted_is_relevant": true
    }
  ]
}
```

---

## API Reference

### `GET /healthz`

Liveness probe. Returns `200 {"status": "ok"}` if the process is up. Used by Railway for healthchecks.

### `GET /readyz`

Readiness probe. Returns `200 {"status": "ready"}` once startup is complete (predictor loaded, artifacts read). Returns `503 {"status": "starting"}` during startup.

### `POST /predict`

Run the cascade against a batch of cases.

#### Request body

| Field             | Type           | Required | Notes                                          |
| ----------------- | -------------- | -------- | ---------------------------------------------- |
| `challenge_id`    | string \| null | no       | Echoed in logs.                                |
| `schema_version`  | int \| null    | no       | Echoed in logs.                                |
| `generated_at`    | string \| null | no       | Ignored.                                       |
| `cases`           | `Case[]`       | yes      | List of cases to score.                        |

`Case`:

| Field             | Type           | Required | Notes                                                    |
| ----------------- | -------------- | -------- | -------------------------------------------------------- |
| `case_id`         | string         | yes      | Echoed back in every prediction.                         |
| `patient_id`      | string \| null | no       | **Never logged** (PHI).                                  |
| `patient_name`    | string \| null | no       | **Never logged** (PHI).                                  |
| `current_study`   | `Study`        | yes      | The current/index study.                                 |
| `prior_studies`   | `Study[]`      | no       | List of priors. May be empty.                            |

`Study`:

| Field                 | Type   | Required | Notes                                                                |
| --------------------- | ------ | -------- | -------------------------------------------------------------------- |
| `study_id`            | string | yes      | Echoed back in the prediction. Need not be unique within a case.     |
| `study_description`   | string | yes      | Free-text radiology description (e.g. `"MRI BRAIN W/O CONTRAST"`).   |
| `study_date`          | string | yes      | Stored as **string**, not `date` — malformed values are accepted.    |

The schema uses `extra='ignore'` at every level, so unknown fields from the evaluator are silently dropped instead of producing 422.

#### Response body

```jsonc
{
  "predictions": [
    { "case_id": "c1", "study_id": "p1", "predicted_is_relevant": true  },
    { "case_id": "c1", "study_id": "p2", "predicted_is_relevant": false }
  ]
}
```

**Invariant**: `len(predictions) == sum(len(case.prior_studies) for case in cases)`. Always. Even if the predictor crashes mid-batch, the anti-skip guard fills missing entries with `false` so the contract holds.

#### Sample curl

A multi-prior, multi-case example with malformed data the endpoint must tolerate:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "cases": [
      {
        "case_id": "c1",
        "patient_name": "Doe, John",
        "current_study": {
          "study_id": "cur",
          "study_description": "MR BRAIN WITHOUT CONTRAST",
          "study_date": "2024-01-01",
          "extra_evaluator_field": "ignored"
        },
        "prior_studies": [
          {"study_id": "p1", "study_description": "CT HEAD WITHOUT CONTRAST", "study_date": "2023-01-01"},
          {"study_id": "p2", "study_description": "XRAY KNEE",                "study_date": "not-a-date"},
          {"study_id": "p3", "study_description": "MR BRAIN WITHOUT CONTRAST","study_date": "2023-06-01"}
        ]
      }
    ]
  }'
```

---

## Configuration

All configuration is via environment variables. There is **no** `config.py` or YAML — that's intentional.

| Env var                | Default       | Description                                                                                              |
| ---------------------- | ------------- | -------------------------------------------------------------------------------------------------------- |
| `USE_STUB_PREDICTOR`   | `1`           | When `1`, returns all-False (76% baseline). Set to `0` to load the cascade.                              |
| `ARTIFACTS_DIR`        | `artifacts`   | Directory containing `lr_model.joblib`, `tfidf_vectorizer.joblib`, and the two pair-stats JSON files.    |
| `PORT`                 | `8000`        | Port to listen on. Railway sets this automatically.                                                      |
| `PYTHONPATH`           | repo root     | Must include the repo root so `from src.app.X import Y` resolves. Set automatically in Docker.           |

If artifacts can't be loaded, the service **falls back to the all-False stub** and continues serving — it doesn't crash on startup. This is a deliberate availability/correctness trade-off.

---

## Testing

The full suite covers normalization, feature engineering, Pydantic schemas, the anti-skip invariant, the stub predictor, the cascade predictor, and end-to-end FastAPI contract tests using `TestClient`.

```bash
# From the repo root, with the venv active:
pytest tests/ -x -v
```

Expected output, cold clone (before training artifacts):

```
============================== 27 passed, 3 skipped in ~0.5s ==============================
```

After running `python -m src.app.train --input data/relevant_priors_public.json --out artifacts/`:

```
============================== 30 passed in ~0.7s ==============================
```

Test layout:

| File                          | What it covers                                                                                                                 |
| ----------------------------- | -----------------------------------------------------------------------------------------------------------------------------  |
| `tests/test_normalize.py`     | 7 tests — abbrev expansion, modality/region/contrast extraction, canonical-key determinism                                     |
| `tests/test_features.py`      | 3 tests — vector length, same-string flag, malformed-date safety                                                               |
| `tests/test_schemas.py`       | 3 tests — minimal payload, extra-field tolerance, bad date acceptance                                                          |
| `tests/test_anti_skip.py`     | 3 tests — complete / missing / extra prediction sets                                                                           |
| `tests/test_predictor.py`     | 2 tests — `AllFalsePredictor` correctness                                                                                      |
| `tests/test_cascade.py`       | 3 tests — auto-skip when `artifacts/` doesn't exist                                                                            |
| `tests/test_contract.py`      | 9 end-to-end tests via `TestClient` — health, readiness, batch sizing, edge cases (empty priors, duplicate `study_id`, 1000-prior payload, malformed dates) |

---

## Training the Model

`src/app/train.py` is a one-shot script that takes the public eval JSON and writes four artifact files:

```bash
python -m src.app.train \
  --input data/relevant_priors_public.json \
  --out artifacts/
```

Sample output:

```
Training on 27614 priors...
  LR feature count: 3544
  Raw pair stats:   12462
  Canon pair stats: 2850
  Saved to: artifacts
```

What it produces:

| File                                  | Size     | Purpose                                                                                      |
| ------------------------------------- | -------- | -------------------------------------------------------------------------------------------- |
| `artifacts/lr_model.joblib`           | ~30 KB   | Pickled `LogisticRegression(max_iter=2000, solver='liblinear')`                              |
| `artifacts/tfidf_vectorizer.joblib`   | ~120 KB | Pickled `TfidfVectorizer(analyzer='char_wb', ngram_range=(3,4), min_df=3, max_features=8000)` |
| `artifacts/raw_pair_stats.json`       | ~900 KB  | `{"current_desc|||prior_desc": {"n": 5, "p": 1.0}, ...}`                                     |
| `artifacts/canonical_pair_stats.json` | ~160 KB | `{"MR\|HEAD\|WITHOUT|||CT\|HEAD\|WITHOUT": {"n": 12, "p": 0.92}, ...}`                        |

`artifacts/` is **gitignored**. The Docker build trains artifacts inside the image at build time via `RUN python -m src.app.train`, which guarantees artifacts and the runtime sklearn version share the same Python environment.

---

## Local Docker

Build a self-contained image with the cascade baked in:

```bash
# 1. Make sure artifacts/ exists (run training first if not)
ls artifacts/

# 2. Build
docker build -t priors-cascade .

# 3. Run
docker run -p 8000:8000 priors-cascade

# 4. Smoke test (in another terminal)
curl http://localhost:8000/healthz
curl http://localhost:8000/readyz
```

The Dockerfile sets `USE_STUB_PREDICTOR=0` and `ARTIFACTS_DIR=/app/artifacts` by default, so the container runs the real cascade out of the box.

---

## Deploying to Railway

The repo is Railway-ready. After pushing to GitHub:

1. Create a new project at https://railway.app and connect this repository.
2. Railway auto-detects `railway.toml` and uses the `Dockerfile` builder.
3. Railway sets `$PORT` automatically — the Dockerfile honors it via `--port ${PORT:-8000}`.
4. Healthcheck path is `/healthz` with a 60 s timeout (configured in `railway.toml`).
5. Restart policy is `ON_FAILURE` with up to 5 retries.

The current deployment is live at `https://relevant-priors-production-8914.up.railway.app` — verify with:

```bash
curl https://relevant-priors-production-8914.up.railway.app/healthz
curl https://relevant-priors-production-8914.up.railway.app/readyz
```

---

## Replay & Scoring Tool

`scripts/replay_public.py` POSTs the entire public eval to a live URL and scores accuracy:

```bash
python scripts/replay_public.py \
  --url http://localhost:8000/predict \
  --input data/relevant_priors_public.json
```

Sample output (against the local cascade endpoint):

```
Loading eval from data/relevant_priors_public.json...
  Cases: 996
  Total priors: 27614
  Truth records: 27614

POSTing to http://localhost:8000/predict...
  HTTP 200 in 0.9s
  Predictions returned: 27614

Results:
  Correct:   26372
  Incorrect: 1242
  Skipped:   0 (count as incorrect)
  Accuracy:  95.50%

PASS: accuracy 95.50% with no skips
```

Gates: the script reports `FAIL` if accuracy < 92% or any predictions are skipped. Both gates pass against the cascade.

---

## Production Considerations

The non-obvious decisions baked into this submission:

- **PHI handling.** `patient_name` and `patient_id` are accepted by the schema (so the evaluator can include them), but are filtered out at the *log formatter* level via `PHI_BLACKLIST = {'patient_name', 'patient_id', 'mrn'}`. They never appear in any structured log line. Verified by post-run grep on every uvicorn log generated during development.

- **Anti-skip invariant.** Every `(case_id, study_id)` is initialized with `predicted_is_relevant = false` before the predictor is invoked. Predictor failures preserve those defaults rather than dropping rows. A final `assert_no_skips(...)` runs before the response is built; if it fails, a repair path reconstructs the response from the `by_key` map. The defaults-plus-assert-and-repair pattern means skipped predictions have not been observed across the full 27,614-prior public replay or any tested failure mode. The invariant is enforced at three layers: initialization, predict-error fallback, and final assert/repair.

- **Schema flexibility.** Pydantic models use `model_config = ConfigDict(extra='ignore')` everywhere. Evaluator-side schema additions never produce 422.

- **Date handling.** `study_date` is typed as `str`, not Pydantic's `date`. Malformed dates (`""`, `"not-a-date"`, `"01/12/2024"`) flow through and are converted to `gap = 0` inside the feature extractor. One bad row never rejects the whole request.

- **Single batched inference.** The LR model runs **once per request**, on all model-fallback rows at once via `predict_proba`. There's no per-item Python loop into scikit-learn.

- **Failure-contained startup path.** If `USE_STUB_PREDICTOR=1` or the `artifacts/` directory does not exist at startup, the service falls back to `AllFalsePredictor` (76% baseline) and starts serving. If artifact files exist but fail to load (e.g., corrupted joblib), startup currently fails — corrupt-artifact fallback is documented as future work in this README.

- **Lazy import of the cascade.** `main.py` imports `cascade.py` *inside* the `lifespan` block, only when `USE_STUB_PREDICTOR=0`. This makes module-level imports fast and lets the stub mode boot without joblib/sklearn touching disk.

- **Env-driven paths.** No paths are hardcoded. `ARTIFACTS_DIR` and `PORT` are env vars; Railway sets `PORT` automatically.

- **Locked thresholds.** The cascade thresholds (`RAW_N_MIN=3`, `RAW_P_HI=0.85`, `CAN_N_MIN=10`, `CAN_P_HI=0.90`) were tuned empirically across 5 seeds and 4 split types. They are marked `# DO NOT MODIFY` in `cascade.py`.

---

## Future Improvements

In priority order, based on what the experiments phase measured:

### 1. Resolve the M3 question with private-split evidence
The M3 LR + GBT ensemble was rejected by the strict acceptance gate (case_grouped lift +0.02 pp) but lifted every drift split (+0.10 to +0.16 pp on cascade; +0.23 to +0.27 pp on the LR/ensemble layer alone). If private-split feedback indicates drift weakness, M3 integration becomes empirically motivated. The integration path is straightforward: `train.py` adds a `HistGradientBoostingClassifier` on the engineered features, `cascade.py` averages probabilities at the model layer.

### 2. Re-design the acceptance gate for model-layer methods
The gate (case_grouped lift ≥ +0.10 pp) embeds an assumption that the cascade is symmetric across split types. The experiments phase falsified this. A revised gate for model-layer methods: drift-split mean lift ≥ +0.10 pp AND case_grouped does not regress. Documented as future work to avoid post-hoc rationalization for this submission.

### 3. Mammography canonical key refinement at higher data volume
M1 was rejected because the refined canonical key fragmented the pair-statistics table on the 996-case public split. With a larger training corpus, the n ≥ 10 override threshold may be reachable for refined keys. Worth revisiting if the project moves to a larger labeled dataset.

### 4. Adversarial drift testing
Current drift simulation tests vocabulary drift only. A more rigorous evaluation would simulate temporal drift, modality-balance drift, and prior-list-length drift.

### 5. Bounded LLM disambiguation for the residual uncertainty band
For predictions where the cascade probability is in [0.40, 0.60], a batched-per-case LLM call could capture cross-modality semantic edges that rules and the linear/GBT models miss. Deliberately not built in v1 because of operational risk and unmeasured behavior.

---

## Tech Stack

| Layer            | Choice                                | Why                                                                  |
| ---------------- | ------------------------------------- | -------------------------------------------------------------------- |
| Language         | Python 3.11                           | Stable, well-supported by every dep below.                           |
| Web framework    | FastAPI 0.110+                        | Async-friendly, Pydantic-native, OpenAPI for free.                   |
| Server           | uvicorn (with `[standard]` extras)    | uvloop + httptools for fast HTTP path.                               |
| Schemas          | Pydantic v2                           | `extra='ignore'`, fast validation, mature ecosystem.                 |
| ML               | scikit-learn 1.3+                     | LogisticRegression + TfidfVectorizer; CPU-only, ships in slim image. |
| Numerics         | numpy 1.24+ / scipy 1.11+ / pandas 2.0+ | Standard scientific Python; pinned to majors that play nice.       |
| Persistence      | joblib + JSON                         | Joblib for sklearn estimators, JSON for human-readable lookup tables.|
| HTTP client      | httpx 0.25+                           | For `scripts/replay_public.py`. Same async story as FastAPI.         |
| Tests            | pytest 8                              | Industry default. `TestClient` from FastAPI for contract tests.      |
| Container        | python:3.11-slim                      | Small base; gcc/g++ added only at build time for sklearn wheels.     |
| Platform         | Railway (Docker builder)              | Single-config deploy; healthcheck + restart policy in `railway.toml`.|

**Explicitly NOT used (and why):**
- **No LLM at runtime.** Cost, latency, and determinism. Future bounded-LLM work is a 0.40–0.60 confidence-band-only addition.
- **No database.** State at runtime is the four artifact files, loaded once at startup. No mutation.
- **No Redis / cache layer.** Inference is already <1 ms per request after warmup; caching adds complexity without meaningful benefit.
- **No custom middleware, no `config.py`, no `utils.py`, no `helpers.py`.** Every utility lives in a named module. This is enforced architecture.

---

## License

This is a residency-challenge submission. All rights reserved by the author.
