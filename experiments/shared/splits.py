"""Deterministic 80/20 splits used by the harness.

The four split types and their seed-determinism logic match the notebook
(cells 13, 20, 31). All shuffles use Python's `random.Random(seed)` so
results are bit-identical across machines.
"""
import random

import pandas as pd


def case_grouped_split(df: pd.DataFrame, seed: int):
    """80/20 split at the case_id level. Test holds 20% of unique cases."""
    rng = random.Random(seed)
    cases = list(df["case_id"].unique())
    rng.shuffle(cases)
    n_test = int(len(cases) * 0.2)
    test_cases = set(cases[:n_test])
    train_df = df[~df["case_id"].isin(test_cases)].copy()
    test_df = df[df["case_id"].isin(test_cases)].copy()
    return train_df, test_df


def _desc_split(df: pd.DataFrame, seed: int, col: str):
    """80/20 holdout of unique values in `col`. Test rows have unseen vocab there."""
    rng = random.Random(seed)
    descs = list(df[col].unique())
    rng.shuffle(descs)
    n_test = int(len(descs) * 0.2)
    test_descs = set(descs[:n_test])
    train_df = df[~df[col].isin(test_descs)].copy()
    test_df = df[df[col].isin(test_descs)].copy()
    return train_df, test_df


def curr_desc_holdout_split(df: pd.DataFrame, seed: int):
    return _desc_split(df, seed, "current_desc")


def prior_desc_holdout_split(df: pd.DataFrame, seed: int):
    return _desc_split(df, seed, "prior_desc")


def both_desc_holdout_split(df: pd.DataFrame, seed: int):
    """Hold out 20% of unique current_desc AND 20% of unique prior_desc.

    Test = rows where current_desc OR prior_desc is in either holdout set.
    The two seeds (seed, seed+100) are independent shuffles, matching the notebook.
    """
    rng_c = random.Random(seed)
    rng_p = random.Random(seed + 100)
    cd = list(df["current_desc"].unique()); rng_c.shuffle(cd)
    pd_ = list(df["prior_desc"].unique()); rng_p.shuffle(pd_)
    test_c = set(cd[: int(len(cd) * 0.2)])
    test_p = set(pd_[: int(len(pd_) * 0.2)])
    mask = df["current_desc"].isin(test_c) | df["prior_desc"].isin(test_p)
    return df[~mask].copy(), df[mask].copy()


SPLIT_FNS = {
    "case_grouped":       case_grouped_split,
    "curr_desc_holdout":  curr_desc_holdout_split,
    "prior_desc_holdout": prior_desc_holdout_split,
    "both_desc_holdout":  both_desc_holdout_split,
}
