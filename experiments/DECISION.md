# Experiments Decision Document

## Summary

Four candidate improvements were measured against the v1 cascade baseline on the public split (996 cases, 27,614 priors). All four were evaluated on a 5-seed × 4-split harness with strict per-fold TF-IDF (no vocabulary leakage from test into train).

**By the original strict gate (case_grouped lift ≥ +0.10 pp), all four methods skip.**

The data tells a more nuanced story. This document presents both the strict-gate result and the structural finding for the user to decide on integration.

## Results Summary Table

| Method | Mechanism | case_grouped cascade lift | Drift-split mean cascade lift | Strict gate |
| --- | --- | --- | --- | --- |
| Method 1 — Mammography canonical key refinement | Add subtype + laterality dimensions to the canonical key for mammography descriptions only | **−0.02 pp** | **−0.003 pp** | **SKIP** |
| Method 2 — XR-implicit detection | Detect view-pattern markers (e.g. "2 VIEWS", "AP/LAT") and infer modality = XRAY when absent | **−0.01 pp** | **+0.000 pp** | **SKIP** |
| Method 3 — GBT ensemble | Add `HistGradientBoostingClassifier` on engineered features alongside LR; average probabilities at the model layer | **+0.02 pp** | **+0.123 pp** | **SKIP** |
| Method 4 — Rare-pair sample weighting | Train LR with `sample_weight = 1 / log1p(canonical_pair_count_in_train)` | **−0.01 pp** | **+0.033 pp** | **SKIP** |

Drift-split mean = mean of `curr_desc_holdout`, `prior_desc_holdout`, `both_desc_holdout` cascade deltas vs v1.

v1 reference numbers (cascade mean ± std): case_grouped 0.9364 ± 0.0029 · curr_desc 0.9214 ± 0.0142 · prior_desc 0.9246 ± 0.0083 · both_desc 0.9256 ± 0.0064.

---

## Per-Method Details

### Method 1 — Mammography Canonical Key Refinement

**Mechanism.** v1's `canonical_key` collapses every mammography description into the same 3-part key `MAMMOGRAPHY|BREAST|UNK`, losing the distinction between screening, diagnostic, tomosynthesis, and laterality. Method 1 extends the key to 5 parts for mammography descriptions only — `MAMMOGRAPHY|BREAST|UNK|<subtype>|<laterality>` — where `<subtype> ∈ {SCREEN, DIAG, TOMO, UNK}` and `<laterality> ∈ {LEFT, RIGHT, BIL, UNK}`. Non-mammography descriptions are unaffected.

**Per-split table** (mean ± std across 5 seeds):

| split | v1 cascade | M1 cascade | v1 lr_only | M1 lr_only | cascade Δpp |
| --- | --- | --- | --- | --- | --- |
| case_grouped | 0.9364 ± 0.0029 | 0.9363 ± 0.0032 | 0.9306 ± 0.0040 | 0.9306 ± 0.0040 | −0.02 |
| curr_desc_holdout | 0.9214 ± 0.0142 | 0.9215 ± 0.0138 | 0.9197 ± 0.0144 | 0.9197 ± 0.0144 | +0.01 |
| prior_desc_holdout | 0.9246 ± 0.0083 | 0.9243 ± 0.0083 | 0.9233 ± 0.0081 | 0.9233 ± 0.0081 | −0.03 |
| both_desc_holdout | 0.9256 ± 0.0064 | 0.9257 ± 0.0061 | 0.9255 ± 0.0046 | 0.9255 ± 0.0046 | +0.01 |

**Gates:**

| gate | result |
| --- | --- |
| `case_grouped_lift_ge_0.10pp` | **FAIL** (lift = −0.02 pp) |
| `no_split_regresses_more_than_0.20pp` | PASS |
| `canon_override_acc_ge_0.95_all_splits` | PASS (lowest 0.9548 on `both_desc_holdout`) |
| `production_pytest_still_passes` | PASS |
| `method_unit_tests_pass` | PASS |

**Observation.** The 5-part key fragments the canonical pair-stat table — many mammography bins now have fewer than the `n ≥ 10` threshold the override layer requires. Canonical override accuracy *dropped* on three of four splits (e.g. `both_desc` 0.9614 → 0.9548) because the layer fires less often, and when it fires on smaller bins it makes more mistakes. The key was finer-grained but the public split's mammography volume isn't large enough to support the finer bins. lr_only is byte-identical to v1 because the canonical_key change doesn't touch the model — the only difference reaches the cascade through the canonical override layer, and it goes the wrong way.

### Method 2 — XR-Implicit Detection

**Mechanism.** Many plain X-rays in the dataset don't carry an "XR" or "X-RAY" token — they appear as "CHEST 2 VIEW FRONTAL & LATRL" or "ANKLE, RIGHT - COMPLETE MIN 3 VIEWS". v1's normalizer returns `modality=UNK` for these, collapsing the canonical key to `UNK|<region>|UNK`. Method 2 detects view-pattern markers (e.g. `\b\d+\s*VIEWS?\b`, `\bAP\s*[/&]\s*LAT(ERAL|RL)?\b`, `\b[123]V\b`) and infers `modality=XRAY` when no other modality token is present.

**Per-split table:**

| split | v1 cascade | M2 cascade | v1 lr_only | M2 lr_only | cascade Δpp |
| --- | --- | --- | --- | --- | --- |
| case_grouped | 0.9364 ± 0.0029 | 0.9363 ± 0.0031 | 0.9306 ± 0.0040 | 0.9306 ± 0.0040 | −0.01 |
| curr_desc_holdout | 0.9214 ± 0.0142 | 0.9212 ± 0.0142 | 0.9197 ± 0.0144 | 0.9197 ± 0.0144 | −0.02 |
| prior_desc_holdout | 0.9246 ± 0.0083 | 0.9248 ± 0.0081 | 0.9233 ± 0.0081 | 0.9233 ± 0.0081 | +0.02 |
| both_desc_holdout | 0.9256 ± 0.0064 | 0.9256 ± 0.0063 | 0.9255 ± 0.0046 | 0.9255 ± 0.0046 | +0.00 |

**Gates:**

| gate | result |
| --- | --- |
| `case_grouped_lift_ge_0.10pp` | **FAIL** (lift = −0.01 pp) |
| `no_split_regresses_more_than_0.20pp` | PASS |
| `canon_override_acc_ge_0.95_all_splits` | PASS (lowest 0.9647) |
| `production_pytest_still_passes` | PASS |
| `method_unit_tests_pass` | PASS |

**Observation.** Volume diagnostic: 53 of 827 unique descriptions had their canonical key changed; 1,963 priors as `current_desc` and 3,112 as `prior_desc` were reclassified — well above the architecture's "potential headroom" threshold. The volume *is* there, but the lift isn't. The LR layer was already classifying these correctly via engineered features (region match, jaccard) and char-wb TF-IDF, which directly captures view-pattern n-grams like `IEW`, `2 V`, `LAT`. Promoting them from `UNK|<region>|UNK` to `XRAY|<region>|UNK` doesn't change the *decision*, just the routing label. Note: lr_only is byte-identical to v1 (canonical_key doesn't touch the model). The one bright spot — `prior_desc_holdout` went from canonical override 0.9689 → 0.9713 — confirms the mechanism works in the right direction, but +0.02 pp on one split isn't enough to clear the gate.

### Method 3 — GBT Ensemble

**Mechanism.** v1's model layer is logistic regression on 14 engineered features + char-wb (3,4)-gram TF-IDF. Method 3 adds a `HistGradientBoostingClassifier` (`max_iter=200, max_depth=6, lr=0.1`) on the engineered features only (GBT can't consume sparse n-grams natively without an expensive conversion) and averages the two probability outputs at the model layer with equal 0.5 weights. Cascade thresholds, canonical_key, and override layers unchanged.

**Per-split table:**

| split | v1 cascade | M3 cascade | v1 lr_only | M3 ensemble lr_only | cascade Δpp | model-layer Δpp |
| --- | --- | --- | --- | --- | --- | --- |
| case_grouped | 0.9364 ± 0.0029 | 0.9366 ± 0.0040 | 0.9306 ± 0.0040 | 0.9307 ± 0.0026 | **+0.02** | +0.01 |
| curr_desc_holdout | 0.9214 ± 0.0142 | 0.9230 ± 0.0139 | 0.9197 ± 0.0144 | 0.9224 ± 0.0139 | **+0.16** | +0.27 |
| prior_desc_holdout | 0.9246 ± 0.0083 | 0.9257 ± 0.0071 | 0.9233 ± 0.0081 | 0.9258 ± 0.0066 | **+0.11** | +0.25 |
| both_desc_holdout | 0.9256 ± 0.0064 | 0.9266 ± 0.0068 | 0.9255 ± 0.0046 | 0.9278 ± 0.0058 | **+0.10** | +0.23 |

**Gates:**

| gate | result |
| --- | --- |
| `case_grouped_lift_ge_0.10pp` | **FAIL** (lift = +0.02 pp) |
| `no_split_regresses_more_than_0.20pp` | PASS (every split is at or above v1) |
| `canon_override_acc_ge_0.95_all_splits` | PASS (identical to v1 — canon layer is upstream of the model) |
| `production_pytest_still_passes` | PASS |
| `method_unit_tests_pass` | PASS |

**Observation.** The chunk's prior expectation was that GBT would lift case_grouped most (because it learns interactions specific to seen modality-region combinations) and drift splits least. The data showed the opposite. The model-layer improvement is real and substantial — LR-only goes from 0.9255 to 0.9278 on `both_desc_holdout`, +0.23 pp at the layer where the improvement is actually applied — but the cascade's deterministic override layers absorb the gain on `case_grouped` (high override coverage when train and test pairs overlap) and pass it through on drift splits (low override coverage when test descriptions are unseen). This is a property of the cascade architecture, not the GBT model. Method 3 fails the strict gate because the gate is targeted at the split where the architecture, not the model, is the bottleneck.

### Method 4 — Rare-Pair Sample Weighting

**Mechanism.** Train LR with `sample_weight = 1 / log1p(canonical_pair_count_in_train)`. High-frequency pairs (where the override layer already handles inference) get weight ≈ 0.2; rare pairs (where the model has to do real work) get weight ≈ 1.0. Same model class as v1 — the only change is which rows the optimizer pays attention to.

**Per-split table:**

| split | v1 cascade | M4 cascade | v1 lr_only | M4 lr_only | cascade Δpp | model-layer Δpp |
| --- | --- | --- | --- | --- | --- | --- |
| case_grouped | 0.9364 ± 0.0029 | 0.9364 ± 0.0030 | 0.9306 ± 0.0040 | 0.9305 ± 0.0028 | −0.01 | −0.01 |
| curr_desc_holdout | 0.9214 ± 0.0142 | 0.9218 ± 0.0137 | 0.9197 ± 0.0144 | 0.9207 ± 0.0138 | **+0.05** | +0.10 |
| prior_desc_holdout | 0.9246 ± 0.0083 | 0.9250 ± 0.0088 | 0.9233 ± 0.0081 | 0.9245 ± 0.0085 | **+0.04** | +0.12 |
| both_desc_holdout | 0.9256 ± 0.0064 | 0.9257 ± 0.0068 | 0.9255 ± 0.0046 | 0.9257 ± 0.0057 | +0.01 | +0.02 |

**Gates:**

| gate | result |
| --- | --- |
| `case_grouped_lift_ge_0.10pp` | **FAIL** (lift = −0.01 pp) |
| `no_split_regresses_more_than_0.20pp` | PASS |
| `canon_override_acc_ge_0.95_all_splits` | PASS (identical to v1 — canon layer upstream of model) |
| `production_pytest_still_passes` | PASS |
| `method_unit_tests_pass` | PASS |

**Observation.** Same fingerprint as M3, weaker in magnitude. The drift-split lift ratio between M3 and M4 is consistently 2–3× (M3 +0.16/+0.11/+0.10 vs M4 +0.05/+0.04/+0.01 on the three drift splits), reflecting that "swap in a non-linear model class" is a larger intervention than "reweight the same model's training". Two independent model-layer interventions producing the same qualitative pattern is what makes the structural finding below trustworthy — it's not a single-method artifact.

---

## The Structural Finding

Methods 1 and 2 (normalizer changes) and Methods 3 and 4 (model-layer changes) produced characteristically different patterns:

**Normalizer methods (M1, M2).** All four split types changed by less than ±0.05 pp. Effectively zero lift in any direction. The canonical override layer's behavior is dominated by its threshold gates (`n ≥ 10, p ≥ 0.90 or p ≤ 0.10`) and not by the granularity of the canonical key. Refining the key fragments the bins (M1) or consolidating the key adds bin volume (M2), but neither path produces a different cascade decision when the LR layer was already classifying these rows correctly via engineered features and char n-grams.

**Model-layer methods (M3, M4).** case_grouped barely moves (−0.01 to +0.02 pp), but every drift split improves: +0.04 to +0.16 pp on cascade; +0.10 to +0.27 pp on the LR/ensemble layer alone. The model-layer improvement is roughly 2× the cascade improvement on drift splits — exactly what a 0.4-0.6 cascade fire rate on those splits would produce, where 40-60% of rows route through the model layer and benefit from the improvement directly while 40-60% are handled by upstream override layers.

The pattern is consistent and explained by the cascade's structure:

- On `case_grouped`, the override layers fire on roughly 60-80% of test rows because raw-pair and canonical-pair lookups have high coverage when train and test descriptions overlap. Improvements to the model layer barely show because the model only handles the residual minority.
- On drift splits, the override layers have low coverage by construction (raw pair coverage is 0% on `curr_desc_holdout` and `prior_desc_holdout` because at least one description is held out; canonical coverage drops to ~65-85%). The model layer carries most of the inference load. This is where model-layer improvements surface.

Two independent model-layer interventions (a different model class in M3, a re-weighted training step in M4) produced the same fingerprint at different magnitudes. This is not noise — it is a property of the architecture: **the cascade caps `case_grouped` accuracy at the override layer, and caps drift-split accuracy at the model layer.**

The original gate (`case_grouped lift ≥ +0.10 pp`) measured at the wrong place for model-layer interventions. This finding is independently valuable regardless of any integration decision.

---

## What This Means For Private Split Performance

The private split is unknown but is almost certainly drifted relative to public — different hospitals, different reporting templates, different vocabularies, different temporal distribution. The exact drift type is unknown:

- If private looks more like `case_grouped` (test vocabulary heavily overlaps train), v1 cascade and any of these methods should perform similarly. Expected accuracy ≈ public-as-train accuracy minus 0–1 pp.
- If private looks more like a description-holdout (significant unseen vocabulary), the model layer matters more. Methods 3 and 4 are positive expected value, with M3 stronger by 2–3×.
- If private has both shifts simultaneously, results scale between these two regimes.

We have no way to know in advance which regime applies. We can only commit to a method based on its risk profile and our prior over the drift type.

---

## Two Decision Frames

### Frame A — Honor the strict gate, ship nothing

The original acceptance gate (`case_grouped_lift_ge_0.10pp`) was chosen deliberately, before any results came in. Its purpose was to prevent post-hoc rationalization — the failure mode where measurements get reinterpreted to justify the methods we wanted to ship. Under this frame, all four methods skip. The v1 cascade ships as-is. The experiments phase produced no integrations but produced four data points and one structural insight; DECISION.md, experiments.md, and the qualitative review carry the value.

**Risk profile.** Low engineering risk. The deployed system is exactly what's been verified end-to-end, including the live endpoint at 95.50% on public-as-train. No regression risk on private. No surprise behavior on edge cases.

**Cost.** Possible foregone +0.05 to +0.10 pp on the private split if private has typical drift. That's small in absolute terms and almost certainly invisible on a residency leaderboard sorted by integer rank.

### Frame B — Integrate M3, honor the data

The data showed two model-layer methods producing the same drift-lift pattern. The case_grouped gate was structurally inappropriate for evaluating model-layer methods because the cascade's override layers absorb model-layer improvements on case_grouped specifically. Holding the line on a gate whose assumption was empirically falsified is not discipline; it's stubbornness. Under this frame M3 ships (M4 is a weaker version of the same intervention; integrating both is duplicative). The integration is mechanical: `cascade.py` loads both an LR and a GBT artifact, averages probas at the model layer; `train.py` fits both during the in-Docker training step.

**Risk profile.** Moderate engineering risk. The new code path roughly doubles the model-fallback inference compute (still well within the 360 s evaluator budget at current request sizes). New artifacts must be retrained inside Docker and deployed. The replay-against-live verification step from Chunk 6 must be re-run after deploy.

**Cost.** Roughly 2 hours of integration + redeploy + verification. Possible regression on case_grouped (within −0.01 pp; not a real risk per the harness). Possible +0.05 to +0.16 pp on the private split if private has drift.

---

## My Recommendation

**Frame A — ship nothing. v1 ships unchanged.**

I recommend this for two reasons. First, the strict gate was set before the data arrived precisely to prevent the kind of recasting Frame B requires. The structural argument for B (case_grouped is the wrong measurement target for cascade architectures) is correct, but it's an argument I'm constructing *after* seeing that case_grouped didn't lift. If the gate was wrong, the right move was to flag that before running the experiments, not to relax it after. Letting falsified gate assumptions invalidate gate decisions is exactly the post-hoc-rationalization failure mode the gate was designed to prevent. Discipline here is not stubbornness — it's holding the methodological line that makes future experiments interpretable.

Second, the absolute magnitude is small enough that I'd rather invest the 2 hours of integration + redeploy + verification work into write-up polish. v1 already lands at 95.50% on public-as-train, which is at or near the leaderboard top, and the drift-split lift from M3 (+0.10 to +0.16 pp) is conditional on private actually exhibiting the description-holdout regime — we have no signal that it does. Meanwhile the structural finding (model-layer methods help only where the cascade doesn't bottleneck on overrides) is independently the most-impressive output of the experiments phase. It belongs in `experiments.md` whether or not we integrate; once it's there, it shifts the submission's qualitative ceiling regardless of what private's accuracy ends up being.

I'd flip to Frame B if the user has reason to believe private is heavy-drift (e.g. an evaluator note that test cases come from different institutions than training). Without that signal, the prior shouldn't move enough to overturn the gate.

---

## What Comes Next

This document does NOT integrate any method. The user reads it and decides:

- **Frame A:** experiments phase ends here. Move to write-up polish (update `experiments.md` with the four method results and the structural finding; verify the live endpoint still serves v1; submit).
- **Frame B:** a separate integration chunk (M6) modifies `src/app/cascade.py` and `src/app/train.py` to add GBT, retrains in Docker, redeploys to Railway, then verifies parity with the harness numbers via `scripts/replay_public.py`.

---

## Files Used

- `experiments/v1_baseline/results.json`
- `experiments/method_1_mammography/results.json`
- `experiments/method_2_xr_implicit/results.json`
- `experiments/method_3_gbt_ensemble/results.json`
- `experiments/method_4_sample_weighting/results.json`
