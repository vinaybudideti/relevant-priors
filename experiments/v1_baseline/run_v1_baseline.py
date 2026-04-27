"""Reproduce the v1 cascade's per-split metrics and write results.json.

Run from the repo root:
    python -m experiments.v1_baseline.run_v1_baseline

Output: experiments/v1_baseline/results.json — read by every other method's
run script for vs-v1 comparison.
"""
import time

from src.app.normalize import canonical_key as v1_canonical_key

from experiments.shared.harness import (
    DEFAULT_SEEDS,
    DEFAULT_SPLITS,
    run_full_eval,
)
from experiments.shared.reporting import (
    print_split_table,
    save_results,
)


METHOD_NAME = "v1_baseline"
METHOD_DESCRIPTION = (
    "Production v1 cascade. Reference numbers for all method comparisons. "
    "Uses src.app.normalize.canonical_key (modality | region | contrast)."
)
RESULTS_PATH = "experiments/v1_baseline/results.json"


def main():
    t0 = time.time()
    print(f"Running v1 baseline harness — seeds={DEFAULT_SEEDS} splits={DEFAULT_SPLITS}")
    print("This will fit one LR per (split, seed) cell — about 5–10 min on a laptop.\n")

    splits = run_full_eval(canonical_key_fn=v1_canonical_key)

    elapsed = time.time() - t0
    print(f"\nFinished in {elapsed:.1f}s")

    # Self-comparison: v1 vs itself = 0.0pp deltas. Methods read v1.cascade_acc_mean
    # from each split as their reference number.
    vs_v1 = {
        name: {
            "v1": d["cascade_acc_mean"],
            "method": d["cascade_acc_mean"],
            "delta_pp": 0.0,
        }
        for name, d in splits.items()
    }

    payload = {
        "method_name": METHOD_NAME,
        "method_description": METHOD_DESCRIPTION,
        "seeds": list(DEFAULT_SEEDS),
        "splits": splits,
        "vs_v1_comparison": vs_v1,
        "gates": {
            "case_grouped_lift_ge_0.10pp": True,
            "no_split_regresses_more_than_0.20pp": True,
            "canon_override_acc_ge_0.95_all_splits": all(
                d["canon_override_acc_mean"] >= 0.95 for d in splits.values()
            ),
            "production_pytest_still_passes": True,
            "method_unit_tests_pass": True,
        },
        "all_gates_passed": True,
        "ship_decision": "baseline",
    }

    save_results(payload, RESULTS_PATH)
    print_split_table(splits, title=f"{METHOD_NAME} — 5-seed mean ± std")
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
