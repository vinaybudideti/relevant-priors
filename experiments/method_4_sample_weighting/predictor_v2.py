"""Method 4: Rare-pair sample weighting.

Trains LR with sample weights inversely proportional to canonical pair frequency
in the training set. Higher weight on rare pairs forces LR to learn from cases
where the override layer can't help.
"""
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.app.features import feature_vector
from src.app.normalize import canonical_key


def train_predictor_v2(train_df):
    """Train LR with sample weights based on canonical pair frequency.

    Returns dict with 'lr' and 'vec', matching the v1 default predictor's interface.
    """
    # Compute canonical pair counts on the train set
    train_df = train_df.copy()
    train_df['c_key'] = train_df['current_desc'].map(canonical_key)
    train_df['p_key'] = train_df['prior_desc'].map(canonical_key)

    pair_counts = train_df.groupby(['c_key', 'p_key']).size()

    # Sample weight = 1 / log1p(pair_count). Pairs with count=1 get weight ~1.0;
    # pairs with count=100 get weight ~0.22.
    train_df['_pair_count'] = train_df.apply(
        lambda r: pair_counts.get((r['c_key'], r['p_key']), 1),
        axis=1
    )
    sample_weights = 1.0 / np.log1p(train_df['_pair_count'].values.astype(float))

    # Build engineered features (same as v1)
    X_eng = np.array([
        feature_vector(r['current_desc'], r['prior_desc'],
                       r['current_date'], r['prior_date'], r['n_priors'])
        for _, r in train_df.iterrows()
    ], dtype=float)

    # Build TF-IDF char n-grams (fit on train fold only — no leakage)
    text_corpus = (
        train_df['current_desc'].str.upper() + ' [SEP] ' +
        train_df['prior_desc'].str.upper()
    ).tolist()
    vec = TfidfVectorizer(
        analyzer='char_wb', ngram_range=(3, 4),
        min_df=3, max_features=8000,
    )
    X_text = vec.fit_transform(text_corpus)

    X_combined = hstack([csr_matrix(X_eng), X_text]).tocsr()
    y = train_df['label'].values.astype(int)

    # Train LR WITH sample weights
    lr = LogisticRegression(max_iter=2000, solver='liblinear')
    lr.fit(X_combined, y, sample_weight=sample_weights)

    return {'lr': lr, 'vec': vec}


def predict_v2(predictor_obj, test_df):
    """Standard LR prediction. Same logic as v1's default predict."""
    lr = predictor_obj['lr']
    vec = predictor_obj['vec']

    X_eng = np.array([
        feature_vector(r['current_desc'], r['prior_desc'],
                       r['current_date'], r['prior_date'], r['n_priors'])
        for _, r in test_df.iterrows()
    ], dtype=float)

    text_corpus = (
        test_df['current_desc'].str.upper() + ' [SEP] ' +
        test_df['prior_desc'].str.upper()
    ).tolist()
    X_text = vec.transform(text_corpus)

    X_combined = hstack([csr_matrix(X_eng), X_text]).tocsr()
    proba = lr.predict_proba(X_combined)[:, 1]
    return proba
