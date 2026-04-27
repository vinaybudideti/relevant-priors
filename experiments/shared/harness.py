"""Multi-seed × multi-split harness used by every method.

Strict §9 contract — each predictor builds its own features per fold:
  - canonical_key_fn:    str -> str  (controls c_key/p_key columns; harness adds them)
  - train_predictor_fn:  callable: (df_train) -> trained predictor object
  - predict_fn:          callable: (predictor, df_test) -> probas ndarray

The harness owns:
  - data loading
  - canonical_key column attachment
  - split / seed loop
  - cascade routing (raw → canonical → model) with v1's locked thresholds
  - metrics aggregation

The harness does NOT own feature building. Every predictor (including the v1
default below) fits its own TfidfVectorizer on the training fold only — there
is no vocabulary leakage from test into train.

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


def default_v1_train(train_df):
    """v1 LR predictor — fits vectorizer on the training fold ONLY (no leakage).

    Returns a dict with the fitted LR and vectorizer for predict_v1 to consume.
    """
    X_eng = np.array([
        feature_vector(r["current_desc"], r["prior_desc"],
                       r["current_date"], r["prior_date"], r["n_priors"])
        for _, r in train_df.iterrows()
    ], dtype=float)

    text_corpus = (train_df["current_desc"].str.upper() + " [SEP] " +
                   train_df["prior_desc"].str.upper()).tolist()
    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 4),
        min_df=3, max_features=8000,
    )
    X_text = vec.fit_transform(text_corpus)

    X_combined = hstack([csr_matrix(X_eng), X_text]).tocsr()
    y = train_df["label"].values.astype(int)

    lr = LogisticRegression(max_iter=2000, solver="liblinear")
    lr.fit(X_combined, y)

    return {"lr": lr, "vec": vec}


def default_v1_predict(predictor, test_df):
    """Return P(label=True) for each row in test_df, using a vectorizer fit on train."""
    X_eng = np.array([
        feature_vector(r["current_desc"], r["prior_desc"],
                       r["current_date"], r["prior_date"], r["n_priors"])
        for _, r in test_df.iterrows()
    ], dtype=float)

    text_corpus = (test_df["current_desc"].str.upper() + " [SEP] " +
                   test_df["prior_desc"].str.upper()).tolist()
    X_text = predictor["vec"].transform(text_corpus)

    X_combined = hstack([csr_matrix(X_eng), X_text]).tocsr()
    return predictor["lr"].predict_proba(X_combined)[:, 1]


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

    Strict §9 signatures:
      train_predictor_fn(df_train) -> predictor
      predict_fn(predictor, df_test) -> ndarray of P(label=True)

    Returns the `splits` dict matching the §7 results.json schema.
    """
    if train_predictor_fn is None:
        train_predictor_fn = default_v1_train
    if predict_fn is None:
        predict_fn = default_v1_predict

    df = load_priors_df(data_path)
    df["c_key"] = df["current_desc"].map(canonical_key_fn)
    df["p_key"] = df["prior_desc"].map(canonical_key_fn)

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
            y_te = te_df["label"].values.astype(int)

            predictor = train_predictor_fn(tr_df)
            proba = predict_fn(predictor, te_df)

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
