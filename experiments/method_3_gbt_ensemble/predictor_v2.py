"""Method 3: GBT ensemble.

Trains LR (v1's model) AND a HistGradientBoostingClassifier on the same engineered
features, then averages their probabilities at prediction time.

Important: the GBT model uses ONLY the dense engineered features (14 columns).
The char n-gram features are sparse and HistGradientBoostingClassifier expects dense
input — adapting it would require careful sparse-to-dense conversion that hurts
training time. Restricting GBT to the engineered features is the cleaner trade-off.
The LR keeps using engineered + n-grams as in v1, so the n-gram signal isn't lost.
"""
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.app.features import feature_vector


def train_predictor_v2(train_df):
    """Train LR (engineered + n-grams) AND GBT (engineered only).

    Returns a dict with both fitted models and the vectorizer, matching what
    predict_v2 expects.

    Signature matches harness.default_v1_train(train_df) -> object
    """
    # Build engineered feature matrix (used by both LR and GBT)
    X_eng = np.array([
        feature_vector(r['current_desc'], r['prior_desc'],
                       r['current_date'], r['prior_date'], r['n_priors'])
        for _, r in train_df.iterrows()
    ], dtype=float)

    # Build TF-IDF char n-grams (used only by LR)
    text_corpus = (
        train_df['current_desc'].str.upper() + ' [SEP] ' +
        train_df['prior_desc'].str.upper()
    ).tolist()
    vec = TfidfVectorizer(
        analyzer='char_wb', ngram_range=(3, 4),
        min_df=3, max_features=8000,
    )
    X_text = vec.fit_transform(text_corpus)

    # LR on combined features (matches v1 exactly)
    X_combined = hstack([csr_matrix(X_eng), X_text]).tocsr()
    y = train_df['label'].values.astype(int)

    lr = LogisticRegression(max_iter=2000, solver='liblinear')
    lr.fit(X_combined, y)

    # GBT on engineered features only (dense, no n-grams)
    gbt = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=0,
    )
    gbt.fit(X_eng, y)

    return {
        'lr': lr,
        'gbt': gbt,
        'vec': vec,
    }


def predict_v2(predictor_obj, test_df):
    """Predict probabilities by averaging LR and GBT.

    Signature matches harness.default_v1_predict(predictor, test_df) -> ndarray of probas
    """
    lr = predictor_obj['lr']
    gbt = predictor_obj['gbt']
    vec = predictor_obj['vec']

    # Engineered features
    X_eng = np.array([
        feature_vector(r['current_desc'], r['prior_desc'],
                       r['current_date'], r['prior_date'], r['n_priors'])
        for _, r in test_df.iterrows()
    ], dtype=float)

    # LR proba (engineered + n-grams)
    text_corpus = (
        test_df['current_desc'].str.upper() + ' [SEP] ' +
        test_df['prior_desc'].str.upper()
    ).tolist()
    X_text = vec.transform(text_corpus)
    X_combined = hstack([csr_matrix(X_eng), X_text]).tocsr()
    proba_lr = lr.predict_proba(X_combined)[:, 1]

    # GBT proba (engineered only)
    proba_gbt = gbt.predict_proba(X_eng)[:, 1]

    # Equal-weight average (the simplest ensemble; defensible without tuning)
    proba_avg = 0.5 * proba_lr + 0.5 * proba_gbt

    return proba_avg
