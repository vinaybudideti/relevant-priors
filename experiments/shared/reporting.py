"""Shared reporting helpers: print comparison tables, save results.json."""
import json
from pathlib import Path
from typing import Optional


def save_results(path: str, payload: dict) -> None:
    """Write `payload` to `path` as pretty JSON. Creates parent dirs if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=False)


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


def print_gates(gates: dict, decision: str) -> None:
    print(f"\n=== Gates ===")
    for gate, passed in gates.items():
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] {gate}")
    print(f"\nship_decision: {decision}")


def evaluate_gates(
    splits: dict,
    vs_v1: dict,
    method_unit_tests_pass: bool,
    production_pytest_passes: bool,
    case_grouped_lift_min_pp: float = 0.10,
    no_split_regress_max_pp: float = 0.20,
    canon_override_acc_min: float = 0.95,
) -> dict:
    """Apply the §8 acceptance gate. Returns the gates dict for results.json."""
    case_lift_pp = vs_v1["case_grouped"]["delta_pp"] if "case_grouped" in vs_v1 else 0.0
    worst_regress_pp = min((d["delta_pp"] for d in vs_v1.values()), default=0.0)

    canon_ok = all(
        d["canon_override_acc_mean"] >= canon_override_acc_min
        for d in splits.values()
    )

    return {
        f"case_grouped_lift_ge_{case_grouped_lift_min_pp:.2f}pp":
            case_lift_pp >= case_grouped_lift_min_pp,
        f"no_split_regresses_more_than_{no_split_regress_max_pp:.2f}pp":
            worst_regress_pp >= -no_split_regress_max_pp,
        f"canon_override_acc_ge_{canon_override_acc_min:.2f}_all_splits": canon_ok,
        "production_pytest_still_passes": production_pytest_passes,
        "method_unit_tests_pass": method_unit_tests_pass,
    }
