"""Method 2 runner — measures XR-implicit detection vs v1 baseline."""
import json
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
from experiments.method_2_xr_implicit.normalize_v2 import canonical_key


def main():
    print("=" * 70)
    print("Method 2: XR-Implicit Detection")
    print("=" * 70)
    print()

    # Load v1 baseline
    v1_baseline = load_v1_baseline()
    print("v1 baseline loaded.")
    print()

    # Run harness
    print("Running harness (5 seeds × 4 splits)...")
    t0 = time.time()
    splits_results = run_full_eval(canonical_key_fn=canonical_key)
    elapsed = time.time() - t0
    print(f"Harness complete in {elapsed:.1f}s.")
    print()

    # Build vs_v1_comparison
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

    # Evaluate gates
    gates = evaluate_gates(
        splits_results, v1_baseline,
        method_test_path='experiments/method_2_xr_implicit/tests_method_2.py'
    )
    all_passed = all(gates.values())

    # Build results.json
    results = {
        'method_name': 'method_2_xr_implicit',
        'method_description': 'Detects view-pattern markers and infers modality=XRAY for descriptions without explicit XR tokens',
        'seeds': [0, 1, 2, 3, 4],
        'splits': splits_results,
        'vs_v1_comparison': vs_v1,
        'gates': gates,
        'all_gates_passed': all_passed,
        'ship_decision': 'ship' if all_passed else 'skip',
    }

    # Save
    output_path = Path('experiments/method_2_xr_implicit/results.json')
    save_results(results, output_path)
    print(f"Results saved to {output_path}")
    print()

    # Display
    print_split_table(splits_results)
    print()
    print_vs_v1(vs_v1)
    print()
    print_gates(gates)
    print()
    print(f"SHIP DECISION: {results['ship_decision'].upper()}")
    if all_passed:
        print("  All 5 gates passed. Method 2 is a candidate for integration.")
    else:
        print("  At least one gate failed. Method 2 will NOT be shipped.")
    print()

    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
