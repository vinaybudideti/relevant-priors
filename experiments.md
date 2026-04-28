# Relevant Priors — Experiments and Findings

## Summary

This submission implements a deterministic 3-layer cascade for the relevant-prior classification task: raw exact-pair lookup, then a normalized canonical-key lookup, then a logistic regression model trained on engineered features and char n-grams. The service is live at `https://relevant-priors-production-8914.up.railway.app/predict`. A full public-eval replay against the live endpoint reports 26,379 correct / 1,235 incorrect / 0 skipped over 27,614 priors (95.53% accuracy, 3.8s round-trip). The work was sequenced as a 7-phase verification before architecture lock, followed by an isolated 4-method exploration with a strict pre-declared acceptance gate. Private-split performance is unknown until evaluation.

## Architecture

The predictor runs three layers in order. Layer 1 looks up the exact `(current_desc, prior_desc)` pair in train statistics and overrides only when evidence is strong (`n ≥ 3` and `p ≥ 0.85` or `p ≤ 0.15`, plus a strong-coverage path at `n ≥ 5`). Layer 2 normalizes each description into a `modality | region | contrast` canonical key and looks up that pair, overriding only when `n ≥ 10` and `p ≥ 0.90` or `p ≤ 0.10`. Layer 3 is logistic regression on 14 engineered features plus char-wb (3,4)-gram TF-IDF, threshold 0.5. The default fallback is `false`. Thresholds were chosen by maximizing override LIFT — the gain over what the LR model alone would have decided on the same rows — not override accuracy alone, because a high override accuracy can coexist with zero lift if the model would have made the same call.

## Approach

The architecture was not designed up-front. It came out of a 7-phase verification process where each phase had pass/fail gates that had to clear before the next phase started:

1. **Data integrity** — case_count, truth_count, label coercion, schema field checks.
2. **Baseline measurement** — measure always_false, always_true, raw-pair lookup with both fallbacks. Establish what we have to beat.
3. **Normalization lift** — measure raw vs canonical lookup on case-grouped and the description-holdout splits. Decide whether normalization helps drift specifically.
4. **Model layer evaluation** — train LR on engineered features and TF-IDF, measure on case-grouped and drift splits.
5. **Cascade composition** — combine the three layers; tune override thresholds by lift, not accuracy.
6. **Error attribution** — per-category error breakdown to identify what the v1 cascade gets wrong.
7. **Architecture lock** — record thresholds and feature definitions; freeze them before deployment.

Only then did production code enter `src/app/`. The four-method exploration described later happened *after* deployment, in an isolated `experiments/` tree, with the production code untouched throughout.

## What worked

**Pair-statistics lookup with confidence thresholding.** Raw exact-pair lookup with default-false achieved 0.8974 ± 0.0067 on case-grouped 5-seed holdout (Phase 2 source). The lookup is treated as a high-confidence override only when statistical evidence is strong (`n ≥ 3, p ≥ 0.85` or `≤ 0.15`).

**Canonical normalization for vocabulary drift.** A `modality | region | contrast` canonical key recovered most of the coverage that exact-pair lookup loses on out-of-vocabulary descriptions:

- Raw lookup coverage on case-grouped: 47.45%.
- Canonical lookup coverage on case-grouped: 89.91%.
- Canonical lift over raw on `curr_desc_holdout`: **+13.15 pp**.
- Canonical lift over raw on `prior_desc_holdout`: **+13.19 pp**.

**Logistic regression as the workhorse.** Fourteen engineered features (modality match, region match, cross-modality same-region, jaccard of normalized tokens, date gap features in days/log/buckets, prior-list length, contrast match, three region-pair indicators) combined with char-wb (3,4)-gram TF-IDF (`min_df=3, max_features=8000`). Under the per-fold-TF-IDF harness (post-M0 fix), LR-only reaches **0.9306 ± 0.0040** on case_grouped, **0.9197 ± 0.0144** on curr_desc_holdout, **0.9233 ± 0.0081** on prior_desc_holdout (v1_baseline/results.json source).

**Tighter override thresholds.** Earlier prototyping used looser thresholds (`n ≥ 5, p ≥ 0.80`). Override accuracy at loose thresholds was approximately 0.95 across splits, but override LIFT was essentially 0 pp on `both_desc_holdout` — overrides were firing on rows where the LR would have made the same decision anyway. Tightening to `n ≥ 10, p ≥ 0.90` raises canonical-override accuracy on case_grouped to **0.9679 ± 0.0044** while keeping override lift positive on every split.

## What failed (or did not help)

**Default-true fallback.** Tested empirically across 5 case-grouped seeds: 0.5428 ± 0.0273. Default-false baseline: 0.7700 ± 0.0103. Decisively worse. Locked default-false in production.

**Raw-string lookup on description-holdout splits.** Raw exact-pair lookup cannot cover pairs whose held-out description never appeared in training. Coverage is 0% by construction on the description-holdout splits, so normalization and the model layer carry those cases.

**Over-aggressive canonical override.** At loose thresholds (`n ≥ 5, p ≥ 0.80`), override accuracy was about 0.95 but override LIFT collapsed toward zero on the drift splits. Tighter thresholds (`n ≥ 10, p ≥ 0.90`) restore positive lift everywhere.

## Verified Performance Tables

### Phase 2 baselines, 5-seed mean ± std on case-grouped

| Method | Case-grouped accuracy |
|---|---|
| always_false | 0.7700 ± 0.0103 |
| always_true | 0.2300 ± 0.0103 |
| raw_pair_fb_false | 0.8974 ± 0.0067 |
| raw_pair_fb_true | 0.5428 ± 0.0273 |

### Phase 3 normalization lift

| Split | Raw | Canonical | Lift |
|---|---|---|---|
| case-grouped | 0.8974 | 0.9239 | +2.65 pp |
| curr_desc_holdout | 0.7643 | 0.8958 | +13.15 pp |
| prior_desc_holdout | 0.7635 | 0.8954 | +13.19 pp |

### Phase 5 cascade with tight thresholds, 5 seeds × 4 splits (post-fix harness)

| Split | Cascade | LR-only | Cascade lift | Canonical override |
|---|---|---|---|---|
| case_grouped | 0.9364 ± 0.0029 | 0.9306 ± 0.0040 | +0.58 pp | 0.9679 ± 0.0044 |
| curr_desc_holdout | 0.9214 ± 0.0142 | 0.9197 ± 0.0144 | +0.17 pp | 0.9654 ± 0.0180 |
| prior_desc_holdout | 0.9246 ± 0.0083 | 0.9233 ± 0.0081 | +0.13 pp | 0.9689 ± 0.0095 |
| both_desc_holdout | 0.9256 ± 0.0064 | 0.9255 ± 0.0046 | +0.01 pp | 0.9614 ± 0.0058 |

Numbers source: `experiments/v1_baseline/results.json`. The cascade lift on `both_desc_holdout` is essentially zero — both descriptions held out simultaneously means neither the raw nor the canonical layer has anything to override against, and the cascade collapses to LR-only.

## Production Deployment

**Anti-skip invariant.** Every prior is initialized with default-false at parse time, updated in place by the predictor, and re-checked by `assert_no_skips` before the response is built; if any check fails, a repair path reconstructs the response from the by-key map. The full public replay reported **0 skipped predictions across 27,614 priors**.

**PHI handling.** `patient_name`, `patient_id`, and `mrn` are never logged. The structured logger filters these fields at the formatter level via a deny-list. Verified by grep across all uvicorn log files generated during a full public-eval replay: 0 matches for any of those three field names.

**Schema flexibility.** Pydantic input models use `extra='ignore'`, so evaluator-side schema additions are accepted without 422 errors. `study_date` is typed as `str` rather than Pydantic's `date`, so malformed dates flow through without rejecting the surrounding request.

**Inference batching.** The model is called once per request for all model-fallback rows, not per item. The full public payload of 27,614 priors round-trips in 3.8 seconds against the live endpoint, well under the 360-second evaluator timeout.

**Failure-contained startup.** If `USE_STUB_PREDICTOR=1` or the artifacts directory is missing at startup, the predictor falls back to `AllFalsePredictor` so the endpoint still serves contract-valid responses. Predictor initialization mode is recorded in the startup log line `predictor_init kind=...` so artifact problems are detectable rather than silent. Request-time predictor exceptions are caught and the rows retain their default-false initialization rather than being dropped.

## Deployment Incident Report

Three issues were caught and fixed during deployment to Railway. Recorded because they reveal real production tradeoffs.

**Issue 1 — `apt-get` build layer was unnecessary and fragile.** The original Dockerfile installed `gcc` and `g++` to support compiling Python wheels. The `apt-get update` step failed during a Railway build with `context canceled`. scikit-learn, scipy, and numpy ship pre-built wheels for `linux/amd64` on Python 3.11; the apt layer was not needed and was removed.

**Issue 2 — Artifact version mismatch caused a silent failure.** Local training ran on Python 3.13 with sklearn 1.8.0, but the Railway container runs Python 3.11. Loading sklearn pickles across minor sklearn versions is unreliable. Resolution: artifacts are now built inside the Docker image during `docker build` via a `RUN python -m src.app.train ...` step, so the runtime and the artifacts share an environment.

**Issue 3 — Port hardcoding broke the Railway healthcheck.** Railway injects its own `$PORT` and runs the healthcheck against it. Hardcoding `--port 8000` in an exec-form `CMD` made uvicorn bind to a port the healthcheck wasn't dialing. Resolution: shell-form `CMD` with `${PORT:-8000}` to expand the variable correctly, plus an explicit `PORT=8000` in Railway Variables.

After all three fixes, the live URL replays the full public eval JSON with **26,379 correct, 1,235 incorrect, 0 skipped** — identical to local results — at 3.8 seconds round-trip on a 27,614-prior request.

## Methods Comparison Phase

After v1 deployment, four candidate improvements were built and measured in isolation against the v1 baseline. Each method has its own folder under `experiments/`, its own unit tests, and its own runner. Production code under `src/app/` was never touched by any method. The acceptance gate was set before any method ran.

### Acceptance gate

Five conditions, all of which had to pass for a method to ship:

- case-grouped cascade lift ≥ +0.10 pp vs v1 baseline,
- no split regresses by more than −0.20 pp,
- canonical override accuracy ≥ 0.95 on every split,
- production pytest still passes (`src/app/` untouched),
- method unit tests pass.

The first condition rejected the model-layer methods. Whether it was the right gate for that method class is discussed in the structural finding section.

### Methodology fix during the experiments phase

While building Method 3, a signature mismatch surfaced between the harness contract and the method runner contract. The original harness pre-computed TF-IDF on the full corpus and shared it across folds, leaking test-fold vocabulary into train. The harness was rewritten to fit TF-IDF on each train fold individually, eliminating the leak. The v1 baseline was re-measured under the corrected harness; numbers shifted by less than 0.05 pp on every split — char-wb (3,4)-grams over a small radiology vocabulary are robust to which split a description ends up in. The fix matters for clean methodology rather than for repointing conclusions.

### Method 1 — Mammography canonical key refinement (skipped)

**Mechanism.** Extend the canonical key from 3 parts to 5 for mammography descriptions only: `MAMMOGRAPHY|BREAST|UNK|<subtype>|<laterality>` where `subtype ∈ {SCREEN, DIAG, TOMO, UNK}` and `laterality ∈ {LEFT, RIGHT, BIL, UNK}`. Non-mammography descriptions return the v1 key unchanged.

| split | v1 cascade | M1 cascade | Δ pp |
|---|---|---|---|
| case_grouped | 0.9364 | 0.9363 | −0.02 |
| curr_desc_holdout | 0.9214 | 0.9215 | +0.01 |
| prior_desc_holdout | 0.9246 | 0.9242 | −0.03 |
| both_desc_holdout | 0.9256 | 0.9257 | +0.01 |

**Why rejected.** The refined key fragmented the canonical pair-stat table. Many mammography bins fell below `n ≥ 10`, so the canonical layer fired less often and made more mistakes when it did fire. Override accuracy on `both_desc_holdout` dropped from 0.9614 (v1) to 0.9548 (M1). Adding key dimensions is the wrong direction when the canonical layer is volume-limited.

### Method 2 — XR-implicit detection (skipped)

**Mechanism.** Detect view-pattern markers (`\b\d+\s*VIEWS?\b`, `\bAP\s*[/&]\s*LAT(ERAL|RL)?\b`, `\b[123]V\b`, etc.) and infer `modality=XRAY` when no other modality token is present. Region detection unchanged.

| split | v1 cascade | M2 cascade | Δ pp |
|---|---|---|---|
| case_grouped | 0.9364 | 0.9363 | −0.01 |
| curr_desc_holdout | 0.9214 | 0.9212 | −0.02 |
| prior_desc_holdout | 0.9246 | 0.9248 | +0.02 |
| both_desc_holdout | 0.9256 | 0.9256 | +0.00 |

**Why rejected.** Volume diagnostic showed 53 of 827 unique descriptions had their canonical key changed (1,963 priors as `current_desc` and 3,112 priors as `prior_desc`). Volume was sufficient — the change was reaching real rows. The LR layer was already classifying these correctly via char n-gram TF-IDF (which captures view-pattern signal directly through 3-4 character windows like `IEW`, `2 V`, `LAT`) and engineered region features. Promoting them from `UNK|<region>|UNK` to `XRAY|<region>|UNK` did not change predictions on most rows. The one positive signal — `prior_desc_holdout` canonical override 0.9689 → 0.9713 — is in the right direction but too small to clear the gate.

### Method 3 — GBT ensemble (skipped — most informative result)

**Mechanism.** Train a `HistGradientBoostingClassifier` (`max_iter=200, max_depth=6, lr=0.1`) on the same 14 engineered features (GBT can't consume the sparse TF-IDF directly without expensive conversion). Average the LR and GBT probabilities at the model layer with equal 0.5 weights. Cascade thresholds and canonical_key unchanged.

| split | v1 cascade | M3 cascade | M3 lr_only (ensemble) | Δ cascade | Δ lr_only |
|---|---|---|---|---|---|
| case_grouped | 0.9364 | 0.9366 | 0.9307 | +0.02 | +0.01 |
| curr_desc_holdout | 0.9214 | 0.9230 | 0.9224 | **+0.16** | **+0.27** |
| prior_desc_holdout | 0.9246 | 0.9257 | 0.9258 | **+0.11** | **+0.25** |
| both_desc_holdout | 0.9256 | 0.9266 | 0.9278 | **+0.10** | **+0.23** |

The headline number is the LR-only column on drift splits. M3 lifts the model-layer accuracy by +0.27, +0.25, and +0.23 pp on the three drift splits. The cascade dampens these gains to +0.16, +0.11, and +0.10 pp respectively, because the cascade's deterministic override layers absorb part of the model-layer improvement when their own coverage is high. On `case_grouped`, where override coverage is highest, the dampening is nearly total — cascade lift is only +0.02 pp.

### Method 4 — Rare-pair sample weighting (skipped)

**Mechanism.** LR trained with `sample_weight = 1 / log1p(canonical_pair_count_in_train)`. Same model class as v1; only the per-row training weight changes. High-frequency pairs (already handled by override) get weight ≈ 0.2; rare pairs (where the model has to do real work) get weight ≈ 1.0.

| split | v1 cascade | M4 cascade | M4 lr_only | Δ cascade | Δ lr_only |
|---|---|---|---|---|---|
| case_grouped | 0.9364 | 0.9364 | 0.9305 | −0.01 | −0.01 |
| curr_desc_holdout | 0.9214 | 0.9218 | 0.9207 | +0.05 | +0.10 |
| prior_desc_holdout | 0.9246 | 0.9250 | 0.9245 | +0.04 | +0.12 |
| both_desc_holdout | 0.9256 | 0.9257 | 0.9257 | +0.01 | +0.02 |

Reproduced Method 3's pattern at roughly half the magnitude on cascade and roughly one-third on LR-only. Same direction (drift help, case-grouped flat), smaller intervention. Sample weighting refocuses the same model; GBT adds a different model class.

### The structural finding

Two normalizer-only methods (M1, M2) and two model-layer methods (M3, M4) produced characteristically different patterns:

- **Normalizer methods.** Every split changed by less than ±0.05 pp. Effectively zero lift in any direction.
- **Model-layer methods.** `case_grouped` barely moved (−0.01 to +0.02 pp), but every drift split improved (+0.04 to +0.16 pp on cascade; +0.10 to +0.27 pp on the LR/ensemble layer alone).

The pattern is consistent and explained by the cascade structure. On `case_grouped`, raw-pair and canonical-pair lookups have high coverage when train and test descriptions overlap, so the override layers fire on most rows. Improvements at the model layer barely surface because the model only handles the residual minority. On drift splits, the override layers have low coverage by construction — the held-out descriptions are unseen — and the model layer carries most of the inference load. That's where model-layer improvements show up.

Two independent model-layer interventions producing the same fingerprint at different magnitudes is not noise. It is a structural property of this cascade design: case_grouped performance is mostly override-limited, while drift-split performance is more model-limited. The cascade's gating decisions cap how much a model-layer improvement can surface on case_grouped, which is why M3 and M4 produce flat case_grouped lift despite improving the LR/ensemble layer significantly on drift splits.

This finding has consequences for future gate design. The strict gate (case-grouped lift ≥ +0.10 pp) was set before data arrived and turned out to embed an assumption — that the cascade is symmetric across split types — which the experiments phase falsified. A revised gate that targets drift-split mean lift while requiring `case_grouped` not to regress would have admitted M3. The redesign is documented as future work; the gate was not changed retroactively for this submission.

## Decision

All four methods were skipped by the strict gate. The v1 cascade ships as deployed at the live endpoint URL.

Because the acceptance gate was pre-declared before any method ran, I kept the production decision tied to it rather than relaxing it after seeing the results. The structural finding from M3 and M4 is documented for future gate design rather than used to override this gate.

If the private score lands materially below the drift-split range observed in validation (especially below approximately 92.5%), M3 integration becomes worth reconsidering as a follow-up submission.

## Limitations

**Label noise on the public split.** Pair-determinism analysis: 98.21% of unique `(current_desc, prior_desc)` pairs in the public split have 100% consistent labels, but a small minority do not — the same pair was sometimes labeled True and sometimes False by the underlying labeling process. Oracle accuracy (always pick majority label per pair, applied to the full public split as both train and test) tops out at 98.84%. A small fraction of public is not recoverable from description-pair features alone by any pair-based classifier on this data alone.

**Drift simulation covers vocabulary only.** The four split types test vocabulary drift only. They do not simulate temporal drift, hospital drift, modality-balance shifts, or adversarial inputs. Private-split performance depends on which kinds of drift the private data exhibits, and we have no advance signal.

**Feature extraction tolerates schema malformations but doesn't fix them.** Pydantic accepts extra fields and malformed dates, and `feature_vector` defaults the date-gap feature to `0` if parsing fails instead of rejecting the request. This preserves the endpoint contract and avoids skipped predictions, but it can reduce accuracy when date gap would otherwise have been informative.

**LR + GBT inference time is roughly 2× LR alone.** If M3 were integrated, the ensemble would roughly double the model-layer compute. With current request sizes (3.8 s for 27,614 priors), there is large headroom under the 360 s timeout. At larger request sizes the headroom shrinks.

## Future Work

1. **Resolve the M3 question with private-split evidence.** If private score lands materially below v1's measured drift-split range, M3 integration becomes empirically motivated. The integration path is small: `train.py` adds a `HistGradientBoostingClassifier` on the engineered features; `cascade.py` averages probas at the model layer; the live endpoint redeploys via a feature branch.

2. **Re-design the acceptance gate for model-layer methods.** Replace `case_grouped lift ≥ +0.10 pp` with `drift-split mean lift ≥ +0.10 pp AND case_grouped does not regress`. Future model-layer methods should be evaluated against the layer where their improvement actually surfaces.

3. **Mammography canonical key refinement at higher data volume.** M1 was rejected because the refined key fragmented the pair table on the 996-case public split. With a larger labeled corpus, the refined keys may reach the `n ≥ 10` override threshold. Worth revisiting if the project moves to a larger dataset.

4. **Adversarial drift testing.** Vocabulary drift is one axis; temporal, modality-balance, and prior-list-length drift are others. Each tests a different generalization mode. Per the prior-art error attribution: mammography accounted for approximately 12% of cascade errors (228 of 1896 across 5 seeds, estimated lift floor +0.76 pp); XR-implicit accounted for approximately 6.6% (estimated lift floor +0.42 pp); vascular/Doppler/Echo/DXA together approximately 5%. Targeted refinements at each axis are tractable.

5. **Bounded LLM disambiguation in the residual uncertainty band.** For predictions where the cascade's final probability is in [0.40, 0.60], a batched-per-case LLM call could capture cross-modality semantic edges that the rules and linear/GBT models miss. Deliberately not built in v1 because of operational risk. Worth measuring if private results suggest the model layer is the bottleneck.

## Reproducing These Numbers

```bash
git clone https://github.com/vinaybudideti/relevant-priors.git
cd relevant-priors
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train artifacts inside the local venv
python -m src.app.train --input data/relevant_priors_public.json --out artifacts/

# Run all production tests (should report 30 passed)
pytest tests/ -x

# Run server locally
USE_STUB_PREDICTOR=0 ARTIFACTS_DIR=artifacts uvicorn src.app.main:app --port 8000

# Verify accuracy against local server
python scripts/replay_public.py \
    --url http://localhost:8000/predict \
    --input data/relevant_priors_public.json

# Verify accuracy against the live endpoint
python scripts/replay_public.py \
    --url https://relevant-priors-production-8914.up.railway.app/predict \
    --input data/relevant_priors_public.json
```

## Reproducing The Experiments Phase

The four-method comparison lives under `experiments/` in the working repository, separate from the production code under `src/app/`. The `experiments/` tree is not included in the submission zip because the production endpoint does not depend on it; the per-method numbers cited above were captured into this write-up directly. To rerun the methods, the `experiments/` directory must be obtained separately. Each method runner writes its own `results.json`, and `experiments/DECISION.md` summarizes across methods.

```bash
python -m experiments.v1_baseline.run_v1_baseline
python -m experiments.method_1_mammography.run_method_1
python -m experiments.method_2_xr_implicit.run_method_2
python -m experiments.method_3_gbt_ensemble.run_method_3
python -m experiments.method_4_sample_weighting.run_method_4
```
