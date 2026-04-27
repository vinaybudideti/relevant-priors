"""Train cascade artifacts from the public eval JSON.

Usage:
    python -m src.app.train --input data/relevant_priors_public.json --out artifacts/
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .features import feature_vector
from .normalize import canonical_key


def load_priors_df(path: str) -> pd.DataFrame:
    with open(path) as f:
        raw = json.load(f)

    label_map = {}
    for t in raw['truth']:
        v = t['is_relevant_to_current']
        if isinstance(v, str):
            v = v.strip().lower() == 'true'
        elif not isinstance(v, bool):
            v = bool(v)
        label_map[(t['case_id'], t['study_id'])] = v

    rows = []
    for c in raw['cases']:
        cid = c['case_id']
        cd = c['current_study']['study_description']
        cdate = c['current_study']['study_date']
        n_priors = len(c['prior_studies'])
        for p in c['prior_studies']:
            key = (cid, p['study_id'])
            if key not in label_map:
                continue
            rows.append({
                'case_id': cid,
                'study_id': p['study_id'],
                'current_desc': cd,
                'prior_desc': p['study_description'],
                'current_date': cdate,
                'prior_date': p['study_date'],
                'n_priors': n_priors,
                'label': label_map[key],
            })
    df = pd.DataFrame(rows)
    df['c_key'] = df['current_desc'].map(canonical_key)
    df['p_key'] = df['prior_desc'].map(canonical_key)
    return df


def train_artifacts(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training on {len(df)} priors...")
    X_eng = np.array([
        feature_vector(r['current_desc'], r['prior_desc'],
                       r['current_date'], r['prior_date'], r['n_priors'])
        for _, r in df.iterrows()
    ], dtype=float)

    text_corpus = (df['current_desc'].str.upper() + ' [SEP] ' + df['prior_desc'].str.upper()).tolist()
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4),
                          min_df=3, max_features=8000)
    X_text = vec.fit_transform(text_corpus)

    X_combined = hstack([csr_matrix(X_eng), X_text]).tocsr()
    y = df['label'].values.astype(int)

    clf = LogisticRegression(max_iter=2000, solver='liblinear')
    clf.fit(X_combined, y)

    g_raw = df.groupby(['current_desc', 'prior_desc'])['label'].agg(['sum', 'count'])
    g_raw['p'] = g_raw['sum'] / g_raw['count']
    raw_stats = {
        f"{cd}|||{pd_}": {'n': int(row['count']), 'p': float(row['p'])}
        for (cd, pd_), row in g_raw.iterrows()
    }

    g_can = df.groupby(['c_key', 'p_key'])['label'].agg(['sum', 'count'])
    g_can['p'] = g_can['sum'] / g_can['count']
    can_stats = {
        f"{ck}|||{pk}": {'n': int(row['count']), 'p': float(row['p'])}
        for (ck, pk), row in g_can.iterrows()
    }

    joblib.dump(clf, out_dir / 'lr_model.joblib')
    joblib.dump(vec, out_dir / 'tfidf_vectorizer.joblib')
    with open(out_dir / 'raw_pair_stats.json', 'w') as f:
        json.dump(raw_stats, f)
    with open(out_dir / 'canonical_pair_stats.json', 'w') as f:
        json.dump(can_stats, f)

    print(f"  LR feature count: {X_combined.shape[1]}")
    print(f"  Raw pair stats:   {len(raw_stats)}")
    print(f"  Canon pair stats: {len(can_stats)}")
    print(f"  Saved to: {out_dir}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--out', default='artifacts/')
    args = ap.parse_args()

    df = load_priors_df(args.input)
    train_artifacts(df, Path(args.out))
