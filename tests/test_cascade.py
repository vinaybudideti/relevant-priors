import pytest
from pathlib import Path
from src.app.cascade import CascadePredictor

ARTIFACTS = Path('artifacts')

@pytest.mark.skipif(not ARTIFACTS.exists(), reason="artifacts not trained yet")
def test_cascade_loads():
    p = CascadePredictor(str(ARTIFACTS))
    assert p.lr is not None
    assert p.vec is not None

@pytest.mark.skipif(not ARTIFACTS.exists(), reason="artifacts not trained yet")
def test_cascade_predicts_batch():
    p = CascadePredictor(str(ARTIFACTS))
    items = [
        {'current_desc': 'MRI BRAIN', 'prior_desc': 'CT HEAD',
         'current_date': '2024-01-01', 'prior_date': '2023-01-01', 'n_priors': 1},
        {'current_desc': 'MAM screen BI with tomo', 'prior_desc': 'MAM screen BI with tomo',
         'current_date': '2024-01-01', 'prior_date': '2022-01-01', 'n_priors': 1},
    ]
    preds = p.predict_batch(items)
    assert len(preds) == 2
    assert all(isinstance(x, bool) for x in preds)

@pytest.mark.skipif(not ARTIFACTS.exists(), reason="artifacts not trained yet")
def test_cascade_handles_empty():
    p = CascadePredictor(str(ARTIFACTS))
    assert p.predict_batch([]) == []
