"""Unit tests for Method 3 GBT ensemble."""
import numpy as np
import pandas as pd
import pytest

from experiments.method_3_gbt_ensemble.predictor_v2 import (
    train_predictor_v2,
    predict_v2,
)


def _make_train_df():
    """Build a small synthetic train df with both classes represented."""
    rows = []
    for i in range(50):
        rows.append({
            'case_id': f'c{i}',
            'study_id': f's{i}',
            'current_desc': 'MRI BRAIN' if i % 2 == 0 else 'CT CHEST',
            'prior_desc': 'CT HEAD' if i % 2 == 0 else 'CT CHEST WITH CONTRAST',
            'current_date': '2024-01-01',
            'prior_date': '2023-01-01',
            'n_priors': 3,
            'label': bool(i % 2 == 0),  # alternating labels
        })
    return pd.DataFrame(rows)


def _make_test_df():
    rows = []
    for i in range(10):
        rows.append({
            'case_id': f'c_test{i}',
            'study_id': f's_test{i}',
            'current_desc': 'MRI BRAIN',
            'prior_desc': 'CT HEAD',
            'current_date': '2024-01-01',
            'prior_date': '2023-01-01',
            'n_priors': 2,
            'label': True,
        })
    return pd.DataFrame(rows)


class TestTrainPredictorV2:
    def test_returns_dict_with_three_keys(self):
        train_df = _make_train_df()
        result = train_predictor_v2(train_df)
        assert isinstance(result, dict)
        assert 'lr' in result
        assert 'gbt' in result
        assert 'vec' in result

    def test_lr_is_fitted(self):
        train_df = _make_train_df()
        predictor = train_predictor_v2(train_df)
        # Calling predict_proba on a fitted model returns shape (n, 2)
        # We don't have the test matrix here, but we can check the model has the attribute
        assert hasattr(predictor['lr'], 'classes_')

    def test_gbt_is_fitted(self):
        train_df = _make_train_df()
        predictor = train_predictor_v2(train_df)
        assert hasattr(predictor['gbt'], 'classes_')


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

    def test_probas_are_average_of_lr_and_gbt(self):
        """Verify the ensemble formula: avg = 0.5 * lr_proba + 0.5 * gbt_proba"""
        from src.app.features import feature_vector
        from scipy.sparse import csr_matrix, hstack

        train_df = _make_train_df()
        test_df = _make_test_df()
        predictor = train_predictor_v2(train_df)

        # Manually compute LR and GBT separately
        X_eng = np.array([
            feature_vector(r['current_desc'], r['prior_desc'],
                           r['current_date'], r['prior_date'], r['n_priors'])
            for _, r in test_df.iterrows()
        ], dtype=float)
        text_corpus = (
            test_df['current_desc'].str.upper() + ' [SEP] ' +
            test_df['prior_desc'].str.upper()
        ).tolist()
        X_text = predictor['vec'].transform(text_corpus)
        X_combined = hstack([csr_matrix(X_eng), X_text]).tocsr()

        manual_lr = predictor['lr'].predict_proba(X_combined)[:, 1]
        manual_gbt = predictor['gbt'].predict_proba(X_eng)[:, 1]
        manual_avg = 0.5 * manual_lr + 0.5 * manual_gbt

        ensemble = predict_v2(predictor, test_df)

        np.testing.assert_array_almost_equal(ensemble, manual_avg, decimal=6)


class TestEnsembleVsIndividual:
    """Check that ensemble probas differ from pure LR — confirms GBT is contributing."""

    def test_ensemble_differs_from_lr_alone(self):
        train_df = _make_train_df()
        test_df = _make_test_df()
        predictor = train_predictor_v2(train_df)

        # Get pure LR probas via the v1-style path
        from src.app.features import feature_vector
        from scipy.sparse import csr_matrix, hstack

        X_eng = np.array([
            feature_vector(r['current_desc'], r['prior_desc'],
                           r['current_date'], r['prior_date'], r['n_priors'])
            for _, r in test_df.iterrows()
        ], dtype=float)
        text_corpus = (
            test_df['current_desc'].str.upper() + ' [SEP] ' +
            test_df['prior_desc'].str.upper()
        ).tolist()
        X_text = predictor['vec'].transform(text_corpus)
        X_combined = hstack([csr_matrix(X_eng), X_text]).tocsr()

        lr_only = predictor['lr'].predict_proba(X_combined)[:, 1]
        ensemble = predict_v2(predictor, test_df)

        # On a real dataset, ensemble must differ from LR somewhere
        # On this synthetic mini-set with only 2 patterns it might be very close,
        # but the formula must produce different numbers if GBT predicts anything other
        # than identical to LR. Check they're not byte-identical.
        assert not np.allclose(ensemble, lr_only, atol=1e-9), (
            "Ensemble should not be byte-identical to LR — GBT should contribute"
        )
