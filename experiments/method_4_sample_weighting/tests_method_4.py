"""Unit tests for Method 4 sample weighting."""
import numpy as np
import pandas as pd

from experiments.method_4_sample_weighting.predictor_v2 import (
    train_predictor_v2,
    predict_v2,
)


def _make_train_df(n_rows=100):
    """Build a small training df with imbalanced canonical pair frequency."""
    rows = []
    # 80 rows of one common pair
    for i in range(80):
        rows.append({
            'case_id': f'c{i}',
            'study_id': f's{i}',
            'current_desc': 'CT CHEST WITH CONTRAST',
            'prior_desc': 'CT CHEST WITHOUT CONTRAST',
            'current_date': '2024-01-01',
            'prior_date': '2023-01-01',
            'n_priors': 5,
            'label': True,
        })
    # 20 rows of varied rare pairs
    for i in range(20):
        rows.append({
            'case_id': f'c_rare{i}',
            'study_id': f's_rare{i}',
            'current_desc': f'MRI BRAIN VARIANT {i}',
            'prior_desc': f'CT HEAD {i}',
            'current_date': '2024-01-01',
            'prior_date': '2023-01-01',
            'n_priors': 2,
            'label': bool(i % 2),
        })
    return pd.DataFrame(rows)


def _make_test_df():
    rows = []
    for i in range(10):
        rows.append({
            'case_id': f'tc{i}',
            'study_id': f'ts{i}',
            'current_desc': 'MRI BRAIN',
            'prior_desc': 'CT HEAD',
            'current_date': '2024-01-01',
            'prior_date': '2023-01-01',
            'n_priors': 2,
            'label': True,
        })
    return pd.DataFrame(rows)


class TestTrainPredictor:
    def test_returns_dict_with_lr_and_vec(self):
        train_df = _make_train_df()
        predictor = train_predictor_v2(train_df)
        assert 'lr' in predictor
        assert 'vec' in predictor

    def test_lr_is_fitted(self):
        train_df = _make_train_df()
        predictor = train_predictor_v2(train_df)
        assert hasattr(predictor['lr'], 'classes_')


class TestPredictV2:
    def test_returns_correct_length(self):
        train_df = _make_train_df()
        test_df = _make_test_df()
        predictor = train_predictor_v2(train_df)
        probas = predict_v2(predictor, test_df)
        assert len(probas) == len(test_df)

    def test_probas_in_unit_interval(self):
        train_df = _make_train_df()
        test_df = _make_test_df()
        predictor = train_predictor_v2(train_df)
        probas = predict_v2(predictor, test_df)
        assert (probas >= 0.0).all()
        assert (probas <= 1.0).all()


class TestSampleWeightsAreApplied:
    """The weighted predictor should produce different probabilities than
    an unweighted one on the same train data.
    """

    def test_weighted_differs_from_unweighted(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.feature_extraction.text import TfidfVectorizer
        from scipy.sparse import csr_matrix, hstack
        from src.app.features import feature_vector

        train_df = _make_train_df()
        test_df = _make_test_df()

        # Train weighted (Method 4)
        weighted_pred = train_predictor_v2(train_df)
        weighted_probas = predict_v2(weighted_pred, test_df)

        # Train unweighted (v1-style)
        X_eng = np.array([
            feature_vector(r['current_desc'], r['prior_desc'],
                           r['current_date'], r['prior_date'], r['n_priors'])
            for _, r in train_df.iterrows()
        ], dtype=float)
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

        unweighted_lr = LogisticRegression(max_iter=2000, solver='liblinear')
        unweighted_lr.fit(X_combined, y)

        X_eng_test = np.array([
            feature_vector(r['current_desc'], r['prior_desc'],
                           r['current_date'], r['prior_date'], r['n_priors'])
            for _, r in test_df.iterrows()
        ], dtype=float)
        text_corpus_test = (
            test_df['current_desc'].str.upper() + ' [SEP] ' +
            test_df['prior_desc'].str.upper()
        ).tolist()
        X_text_test = vec.transform(text_corpus_test)
        X_combined_test = hstack([csr_matrix(X_eng_test), X_text_test]).tocsr()
        unweighted_probas = unweighted_lr.predict_proba(X_combined_test)[:, 1]

        # The two should produce different probas (weighting changed something)
        assert not np.allclose(weighted_probas, unweighted_probas, atol=1e-9), (
            "Weighted predictor must produce different probas than unweighted"
        )
