"""Shared reporting helpers: print comparison tables, save results.json, evaluate gates."""
import json
import subprocess
from pathlib import Path
from typing import Optional


def save_results(payload: dict, path) -> None:
    """Write payload as pretty JSON to path. Creates parent dirs if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)


def load_v1_baseline(repo_root: str = ".") -> Optional[dict]:
    """Read the v1 baseline results, or return None if it doesn't exist yet."""
    p = Path(repo_root) / "experiments" / "v1_baseline" / "results.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def print_split_table(splits: dict, title: str = "Per-split results") -> None:
    """One-row-per-split summary table — used by every run script."""
    print(f"\n=== {title} ===")
    print(f"{'split':<22s} {'cascade':<18s} {'lr_only':<18s} {'canon_override':<18s}")
    print("-" * 80)
    for name, d in splits.items():
        cas = f"{d['cascade_acc_mean']:.4f} ± {d['cascade_acc_std']:.4f}"
        lr  = f"{d['lr_only_acc_mean']:.4f} ± {d['lr_only_acc_std']:.4f}"
        can = f"{d['canon_override_acc_mean']:.4f} ± {d['canon_override_acc_std']:.4f}"
        print(f"{name:<22s} {cas:<18s} {lr:<18s} {can:<18s}")


def print_vs_v1(vs_v1: dict, title: str = "vs v1 baseline") -> None:
    print(f"\n=== {title} ===")
    print(f"{'split':<22s} {'v1':<10s} {'method':<10s} {'delta_pp':<10s}")
    print("-" * 60)
    for name, d in vs_v1.items():
        print(f"{name:<22s} {d['v1']:.4f}    {d['method']:.4f}    {d['delta_pp']:+.2f}")


def print_gates(gates: dict) -> None:
    """Print pass/fail for each gate. Caller prints ship_decision separately."""
    print("=" * 60)
    print("GATE RESULTS")
    print("=" * 60)
    for name, passed in gates.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")


def evaluate_gates(splits_results: dict, v1_baseline: dict,
                   method_test_path: str) -> dict:
    """Evaluate the 5 acceptance gates. Returns dict matching results.json `gates` schema.

    Runs pytest internally for two of the gates:
      - production_pytest_still_passes: `pytest tests/ -x` from repo root
      - method_unit_tests_pass: `pytest <method_test_path> -x`

    Subprocess inherits cwd from the caller — runners are expected to be invoked
    from the repo root via `python -m experiments.method_N_<name>.run_method_N`.
    """
    # Gate 1: case_grouped lift >= +0.10pp
    cas_v1 = v1_baseline["splits"]["case_grouped"]["cascade_acc_mean"]
    cas_method = splits_results["case_grouped"]["cascade_acc_mean"]
    case_grouped_lift_pp = (cas_method - cas_v1) * 100
    g1 = case_grouped_lift_pp >= 0.10

    # Gate 2: no split regresses by more than -0.20pp
    g2 = True
    for split_name, m in splits_results.items():
        v1_cas = v1_baseline["splits"][split_name]["cascade_acc_mean"]
        delta_pp = (m["cascade_acc_mean"] - v1_cas) * 100
        if delta_pp < -0.20:
            g2 = False
            break

    # Gate 3: canonical override accuracy >= 0.95 on every split
    g3 = all(m["canon_override_acc_mean"] >= 0.95 for m in splits_results.values())

    # Gate 4: production pytest still passes
    prod_result = subprocess.run(
        ["pytest", "tests/", "-x", "--tb=no", "-q"],
        capture_output=True, text=True,
    )
    g4 = (prod_result.returncode == 0)

    # Gate 5: method unit tests pass
    method_result = subprocess.run(
        ["pytest", method_test_path, "-x", "--tb=no", "-q"],
        capture_output=True, text=True,
    )
    g5 = (method_result.returncode == 0)

    return {
        "case_grouped_lift_ge_0.10pp": g1,
        "no_split_regresses_more_than_0.20pp": g2,
        "canon_override_acc_ge_0.95_all_splits": g3,
        "production_pytest_still_passes": g4,
        "method_unit_tests_pass": g5,
    }
