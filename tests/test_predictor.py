from src.app.predictor import AllFalsePredictor


def test_all_false_returns_correct_count():
    p = AllFalsePredictor()
    assert p.predict_batch([{}, {}, {}]) == [False, False, False]

def test_all_false_empty():
    assert AllFalsePredictor().predict_batch([]) == []
