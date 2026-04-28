"""Method 4 runner — measures rare-pair sample weighting vs v1 baseline."""
import sys
import time
from pathlib import Path

from experiments.shared.harness import run_full_eval
from experiments.shared.reporting import (
    save_results,
    load_v1_baseline,
    print_split_table,
    print_vs_v1,
    print_gates,
    evaluate_gates,
)
from src.app.normalize import canonical_key  # v1 — Method 4 doesn't change it
from experiments.method_4_sample_weighting.predictor_v2 import (
    train_predictor_v2,
    predict_v2,
)


def main():
    print("=" * 70)
    print("Method 4: Rare-Pair Sample Weighting")
    print("=" * 70)
    print()

    v1_baseline = load_v1_baseline()
    print("v1 baseline loaded.")
    print()

    print("Running harness (5 seeds × 4 splits)...")
    t0 = time.time()
    splits_results = run_full_eval(
        canonical_key_fn=canonical_key,
        train_predictor_fn=train_predictor_v2,
        predict_fn=predict_v2,
    )
    elapsed = time.time() - t0
    print(f"Harness complete in {elapsed:.1f}s.")
    print()

    vs_v1 = {}
    for split_name, m in splits_results.items():
        v1_cas = v1_baseline['splits'][split_name]['cascade_acc_mean']
        method_cas = m['cascade_acc_mean']
        vs_v1[split_name] = {
            'v1':     round(v1_cas, 4),
            'method': round(method_cas, 4),
            'delta_pp': round((method_cas - v1_cas) * 100, 2),
        }
        m['lift_vs_v1_pp'] = round((method_cas - v1_cas) * 100, 2)

    gates = evaluate_gates(
        splits_results, v1_baseline,
        method_test_path='experiments/method_4_sample_weighting/tests_method_4.py'
    )
    all_passed = all(gates.values())

    results = {
        'method_name': 'method_4_sample_weighting',
        'method_description': 'Trains LR with sample weights inversely proportional to canonical pair frequency in the training set',
        'seeds': [0, 1, 2, 3, 4],
        'splits': splits_results,
        'vs_v1_comparison': vs_v1,
        'gates': gates,
        'all_gates_passed': all_passed,
        'ship_decision': 'ship' if all_passed else 'skip',
    }

    output_path = Path('experiments/method_4_sample_weighting/results.json')
    save_results(results, output_path)
    print(f"Results saved to {output_path}")
    print()

    print_split_table(splits_results)
    print()
    print_vs_v1(vs_v1)
    print()
    print_gates(gates)
    print()
    print(f"SHIP DECISION: {results['ship_decision'].upper()}")
    if all_passed:
        print("  All 5 gates passed.")
    else:
        print("  At least one gate failed.")
    print()

    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
