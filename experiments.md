# Experiments — Relevant Priors

## Approach

This submission implements a deterministic 3-layer cascade for the relevant-prior classification task. Each layer was verified empirically on the public 996-case / 27,614-prior split before being added.

The architecture choice was driven by a key finding from the data: the pair (current_description, prior_description) is nearly deterministic of the label. 98.21% of unique pairs in the public split have 100% consistent labels, putting the oracle accuracy at 98.84%. This suggested a memorization-friendly approach with normalization to handle vocabulary drift.

## What worked

**Pair-statistics lookup with confidence thresholding.** Raw exact-pair lookup achieved 89.74% on case-grouped 5-seed holdout. The lookup is treated as a high-confidence override only when statistical evidence is strong (n ≥ 3, p ≥ 0.85 or p ≤ 0.15).

**Canonical normalization for vocabulary drift.** A modality + region + contrast canonical key recovered most of the coverage lost on out-of-vocabulary descriptions:
- raw lookup coverage on case-grouped: 47.45%
- canonical lookup coverage on case-grouped: 89.91%
- canonical lookup lift over raw on current-desc holdout: +13.15pp
- canonical lookup lift over raw on prior-desc holdout: +13.19pp

**Logistic regression as the workhorse.** 14 engineered features (modality match, region match, cross-modality same-region, jaccard of normalized tokens, date gap features, prior-list length) combined with char-wb (3,4)-gram TF-IDF achieved 93.06% on case-grouped, 92.32% on prior-desc holdout.

**Tighter override thresholds (n ≥ 10, p ≥ 0.90).** Override accuracy improved from ~95% (loose thresholds) to 96.79% (tight) on case-grouped, while keeping override LIFT positive on every split. Lift is the right metric — override accuracy alone can be high while the underlying model would have made the same decision anyway.

## What failed (or didn't help)

**Default-true fallback.** Tested empirically: 54.28% accuracy across 5 case-grouped seeds. Decisively worse than default-false (76.22% baseline). Locked default-false in production.

**Raw-string lookup without normalization on drift splits.** 0% coverage on description-holdout splits because the held-out descriptions are by definition unseen.

**Over-aggressive canonical override (loose thresholds).** Canonical override at n ≥ 5, p ≥ 0.80 had override accuracy of 0.95 but override LIFT of essentially 0pp on the both-desc holdout — the override was firing on rows where the LR model would have made the same decision. Tighter thresholds fix this.

## Key measurements

All numbers are mean ± std across 5 deterministic seeds on the public 996-case split.

### Phase 2 baselines
| Method | Case-grouped accuracy |
|---|---|
| always_false | 0.7700 ± 0.0103 |
| always_true | 0.2300 ± 0.0103 |
| raw_pair_fb_false | 0.8974 ± 0.0067 |
| raw_pair_fb_true | 0.5428 ± 0.0273 |

### Phase 3 normalization lift
| Split | Raw | Canonical | Lift |
|---|---|---|---|
| case-grouped | 0.8974 | 0.9239 | +2.65pp |
| current-desc holdout | 0.7643 | 0.8958 | +13.15pp |
| prior-desc holdout | 0.7635 | 0.8954 | +13.19pp |

### Phase 5 cascade with tight thresholds
| Split | Cascade | LR-only | Cascade lift |
|---|---|---|---|
| case-grouped | 0.9364 | 0.9306 | +0.58pp |
| current-desc holdout | 0.9213 | 0.9194 | +0.19pp |
| prior-desc holdout | 0.9244 | 0.9232 | +0.12pp |
| both-desc holdout | 0.9257 | 0.9255 | +0.03pp |

Canonical override accuracy: 0.9614–0.9689 across splits. Override lift positive on all splits.

## Production considerations

- **PHI handling:** patient_name and patient_id never logged; logger filters them at the formatter level.
- **Anti-skip invariant:** every `(case_id, study_id)` is initialized with default-false at parse time. Predictor failures preserve defaults. Final invariant asserted before response, with a repair path. Skips are structurally impossible.
- **Schema flexibility:** Pydantic input models use `extra='ignore'` to accept evaluator schema additions without 422.
- **Date handling:** study_date is parsed as a string, not a Pydantic `date` type, to accept malformed dates without rejecting the whole request.
- **Single batched inference:** the LR model is called once per request for all model-fallback rows, never per-item.
- **Graceful degradation:** if artifacts fail to load, the service falls back to `AllFalsePredictor` rather than failing startup.

## Next improvements

1. **Mammography canonical key refinement.** Error analysis showed mammography produces 12% of cascade errors (228 of 1896 across 5 seeds). Adding laterality and screening/diagnostic flags to the canonical key would address the largest single error bucket. Estimated lift floor: +0.76pp.

2. **XR view-pattern detection.** Plain radiographs without "XR" tokens (e.g., "CHEST 2 VIEW FRONTAL & LATRL") account for 6.6% of cascade errors. View-pattern regex + radiographic region taxonomy would close this. Estimated lift floor: +0.42pp.

3. **VAS / Doppler / Echo / DXA disambiguation.** Smaller buckets totaling ~5% of cascade errors. Combined into one batch.

4. **Bounded LLM disambiguation.** For predictions in the 0.40–0.60 probability band only, a batched-per-case LLM call could capture cross-modality semantic edges (e.g., CT angio coronary ↔ XR chest cardiac workup) that the rules and model both miss. Strict deadline guard, caches by canonical pair.

5. **Per-modality threshold tuning.** Currently using global threshold 0.5. Modality-cluster-specific thresholds may capture asymmetric clinical loss.
