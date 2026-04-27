"""3-layer cascade: raw pair → canonical pair → LR model. Default false."""
import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack

from .features import feature_vector
from .normalize import canonical_key

# LOCKED THRESHOLDS — verified empirically. DO NOT MODIFY.
RAW_N_MIN = 3
RAW_P_HI = 0.85
RAW_P_LO = 0.15
RAW_STRONG_N = 5
CAN_N_MIN = 10
CAN_P_HI = 0.90
CAN_P_LO = 0.10
DEFAULT_FALLBACK = False


class CascadePredictor:
    def __init__(self, artifacts_dir: str):
        d = Path(artifacts_dir)
        self.lr = joblib.load(d / 'lr_model.joblib')
        self.vec = joblib.load(d / 'tfidf_vectorizer.joblib')
        with open(d / 'raw_pair_stats.json') as f:
            self.raw_stats = json.load(f)
        with open(d / 'canonical_pair_stats.json') as f:
            self.can_stats = json.load(f)

    def _raw_lookup(self, current_desc: str, prior_desc: str) -> Optional[bool]:
        key = f"{current_desc}|||{prior_desc}"
        s = self.raw_stats.get(key)
        if s is None:
            return None
        n, p = s['n'], s['p']
        if n >= RAW_N_MIN and p >= RAW_P_HI:
            return True
        if n >= RAW_N_MIN and p <= RAW_P_LO:
            return False
        if n >= RAW_STRONG_N:
            return p >= 0.5
        return None

    def _canonical_lookup(self, c_key: str, p_key: str) -> Optional[bool]:
        key = f"{c_key}|||{p_key}"
        s = self.can_stats.get(key)
        if s is None:
            return None
        n, p = s['n'], s['p']
        if n >= CAN_N_MIN and p >= CAN_P_HI:
            return True
        if n >= CAN_N_MIN and p <= CAN_P_LO:
            return False
        return None

    def predict_batch(self, items: list) -> list:
        if not items:
            return []

        results = [None] * len(items)
        model_idx = []
        for i, it in enumerate(items):
            try:
                cd = it['current_desc']
                pd_ = it['prior_desc']
                r = self._raw_lookup(cd, pd_)
                if r is not None:
                    results[i] = r
                    continue
                ck = canonical_key(cd)
                pk = canonical_key(pd_)
                r = self._canonical_lookup(ck, pk)
                if r is not None:
                    results[i] = r
                    continue
                model_idx.append(i)
            except Exception:
                results[i] = DEFAULT_FALLBACK

        if model_idx:
            try:
                feat_rows = []
                text_rows = []
                for i in model_idx:
                    it = items[i]
                    feat_rows.append(feature_vector(
                        it['current_desc'], it['prior_desc'],
                        it['current_date'], it['prior_date'],
                        it['n_priors']
                    ))
                    text_rows.append(
                        it['current_desc'].upper() + ' [SEP] ' + it['prior_desc'].upper()
                    )
                X_eng = np.array(feat_rows, dtype=float)
                X_text = self.vec.transform(text_rows)
                X = hstack([csr_matrix(X_eng), X_text]).tocsr()
                probas = self.lr.predict_proba(X)[:, 1]
                for j, i in enumerate(model_idx):
                    results[i] = bool(probas[j] >= 0.5)
            except Exception:
                for i in model_idx:
                    if results[i] is None:
                        results[i] = DEFAULT_FALLBACK

        for i in range(len(results)):
            if results[i] is None:
                results[i] = DEFAULT_FALLBACK

        return results
