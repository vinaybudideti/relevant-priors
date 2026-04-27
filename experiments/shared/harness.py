"""Multi-seed × multi-split harness used by every method.

The harness owns the cascade routing logic (raw lookup → canonical lookup →
model fallback) and the locked thresholds. Each method swaps in:
  - canonical_key_fn:    str -> str  (controls c_key/p_key columns)
  - train_predictor_fn:  callable that takes (df_train, X_full, y_full) -> predictor
  - predict_fn:          callable that takes (predictor, df_test, X_full) -> probas

Methods that only change the canonical key (Methods 1, 2) can rely on the
default v1 LR train/predict supplied by `make_default_v1_predictor()` below.
Methods that change the predictor (Methods 3, 4) pass their own pair.

The default uses precomputed engineered + char-wb TF-IDF features over the
full priors_df (matching the notebook's cell 31 setup). This means TF-IDF
vocabulary is shared across folds — that's a deliberate match of v1's
published numbers, not an accident.

Locked cascade thresholds (DO NOT MODIFY — see PROJECT_ARCHITECTURE §11):
"""
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.app.features import feature_vector

from .data_loader import load_priors_df
from .splits import SPLIT_FNS

# Cascade thresholds — exactly v1's values from src/app/cascade.py
RAW_N_MIN = 3
RAW_P_HI = 0.85
RAW_P_LO = 0.15
RAW_STRONG_N = 5
CAN_N_MIN = 10
CAN_P_HI = 0.90
CAN_P_LO = 0.10

DEFAULT_DATA_PATH = "data/relevant_priors_public.json"
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_SPLITS = (
    "case_grouped",
    "curr_desc_holdout",
    "prior_desc_holdout",
    "both_desc_holdout",
)


def build_features(df: pd.DataFrame):
    """Build engineered feature vectors + char-wb (3,4)-gram TF-IDF.

    Matches src/app/train.py and the notebook's cell 26-27.
    """
    X_eng = np.array([
        feature_vector(r["current_desc"], r["prior_desc"],
                       r["current_date"], r["prior_date"], r["n_priors"])
        for _, r in df.iterrows()
    ], dtype=float)
    text_corpus = (df["current_desc"].str.upper() + " [SEP] " +
                   df["prior_desc"].str.upper()).tolist()
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4),
                          min_df=3, max_features=8000)
    X_text = vec.fit_transform(text_corpus)
    X_combined = hstack([csr_matrix(X_eng), X_text]).tocsr()
    return X_combined


def default_v1_train(df_train, X_full, y_full):
    """v1's predictor: LogisticRegression on engineered + TF-IDF features."""
    idx = df_train.index.values
    clf = LogisticRegression(max_iter=2000, solver="liblinear")
    clf.fit(X_full[idx], y_full[idx])
    return clf


def default_v1_predict(clf, df_test, X_full):
    """Return P(label=True) for each row in df_test."""
    idx = df_test.index.values
    return clf.predict_proba(X_full[idx])[:, 1]


def _apply_cascade(train_df: pd.DataFrame, test_df: pd.DataFrame,
                   model_proba: np.ndarray):
    """Run the 3-layer cascade. Returns (preds[bool], layer[1|2|3])."""
    g_raw = train_df.groupby(["current_desc", "prior_desc"])["label"].agg(["sum", "count"])
    g_raw["p"] = g_raw["sum"] / g_raw["count"]
    raw_n = g_raw["count"].to_dict()
    raw_p = g_raw["p"].to_dict()

    g_can = train_df.groupby(["c_key", "p_key"])["label"].agg(["sum", "count"])
    g_can["p"] = g_can["sum"] / g_can["count"]
    can_n = g_can["count"].to_dict()
    can_p = g_can["p"].to_dict()

    raw_keys = list(zip(test_df["current_desc"], test_df["prior_desc"]))
    can_keys = list(zip(test_df["c_key"], test_df["p_key"]))
    out = (model_proba >= 0.5).copy()
    layer = np.full(len(test_df), 3, dtype=int)

    for i in range(len(test_df)):
        rk = raw_keys[i]
        if rk in raw_n:
            n, p = raw_n[rk], raw_p[rk]
            if n >= RAW_N_MIN and p >= RAW_P_HI:
                out[i] = True; layer[i] = 1; continue
            if n >= RAW_N_MIN and p <= RAW_P_LO:
                out[i] = False; layer[i] = 1; continue
            if n >= RAW_STRONG_N:
                out[i] = (p >= 0.5); layer[i] = 1; continue
        ck = can_keys[i]
        if ck in can_n:
            n, p = can_n[ck], can_p[ck]
            if n >= CAN_N_MIN and p >= CAN_P_HI:
                out[i] = True; layer[i] = 2; continue
            if n >= CAN_N_MIN and p <= CAN_P_LO:
                out[i] = False; layer[i] = 2; continue
    return out.astype(bool), layer


def run_full_eval(
    canonical_key_fn: Callable[[str], str],
    train_predictor_fn: Optional[Callable] = None,
    predict_fn: Optional[Callable] = None,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    splits: Iterable[str] = DEFAULT_SPLITS,
    data_path: str = DEFAULT_DATA_PATH,
):
    """Run the full 5-seed × 4-split harness for one method.

    Returns a dict matching the `splits` key in the §7 results.json schema:
    {
      "case_grouped": {
        "cascade_acc_mean": 0.9412, "cascade_acc_std": 0.0028,
        "lr_only_acc_mean": 0.9306, "lr_only_acc_std": 0.0040,
        "canon_override_acc_mean": 0.9712, "canon_override_acc_std": 0.0015,
      },
      ...
    }

    `lift_vs_v1_pp` is filled in by the per-method run script after loading
    v1_baseline/results.json — the harness does not know about v1 here.
    """
    if train_predictor_fn is None:
        train_predictor_fn = default_v1_train
    if predict_fn is None:
        predict_fn = default_v1_predict

    df = load_priors_df(data_path)
    df["c_key"] = df["current_desc"].map(canonical_key_fn)
    df["p_key"] = df["prior_desc"].map(canonical_key_fn)

    X_combined = build_features(df)
    y_all = df["label"].values.astype(int)

    seeds = tuple(seeds)
    splits = tuple(splits)
    out = {}

    for split_name in splits:
        if split_name not in SPLIT_FNS:
            raise KeyError(f"unknown split: {split_name!r}")
        split_fn = SPLIT_FNS[split_name]

        cas_accs, lr_accs, can_ov_accs = [], [], []

        for seed in seeds:
            tr_df, te_df = split_fn(df, seed)
            if len(te_df) < 100:
                continue
            y_te = y_all[te_df.index.values]

            predictor = train_predictor_fn(tr_df, X_combined, y_all)
            proba = predict_fn(predictor, te_df, X_combined)

            preds, layer = _apply_cascade(tr_df, te_df, proba)
            model_pred = (proba >= 0.5)

            cas_accs.append(float((preds == y_te).mean()))
            lr_accs.append(float((model_pred == y_te).mean()))

            mask_can = (layer == 2)
            if mask_can.sum() > 0:
                can_ov_accs.append(float((preds[mask_can] == y_te[mask_can]).mean()))

        out[split_name] = {
            "cascade_acc_mean":         float(np.mean(cas_accs)) if cas_accs else 0.0,
            "cascade_acc_std":          float(np.std(cas_accs))  if cas_accs else 0.0,
            "lr_only_acc_mean":         float(np.mean(lr_accs))  if lr_accs else 0.0,
            "lr_only_acc_std":          float(np.std(lr_accs))   if lr_accs else 0.0,
            "canon_override_acc_mean":  float(np.mean(can_ov_accs)) if can_ov_accs else 0.0,
            "canon_override_acc_std":   float(np.std(can_ov_accs))  if can_ov_accs else 0.0,
        }

    return out
