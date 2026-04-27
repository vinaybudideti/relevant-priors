import pytest
from src.app.anti_skip import assert_no_skips, AntiSkipError
from src.app.schemas import CaseIn, StudyIn, Prediction


def _make_case(case_id, prior_ids):
    return CaseIn(
        case_id=case_id,
        current_study=StudyIn(study_id="cur", study_description="X", study_date="2024-01-01"),
        prior_studies=[
            StudyIn(study_id=pid, study_description="Y", study_date="2023-01-01")
            for pid in prior_ids
        ]
    )


def test_complete_predictions_pass():
    cases = [_make_case("c1", ["p1", "p2"])]
    preds = [
        Prediction(case_id="c1", study_id="p1", predicted_is_relevant=False),
        Prediction(case_id="c1", study_id="p2", predicted_is_relevant=True),
    ]
    assert_no_skips(cases, preds)  # should not raise

def test_missing_prediction_raises():
    cases = [_make_case("c1", ["p1", "p2"])]
    preds = [
        Prediction(case_id="c1", study_id="p1", predicted_is_relevant=False),
    ]
    with pytest.raises(AntiSkipError):
        assert_no_skips(cases, preds)

def test_extra_prediction_raises():
    cases = [_make_case("c1", ["p1"])]
    preds = [
        Prediction(case_id="c1", study_id="p1", predicted_is_relevant=False),
        Prediction(case_id="c1", study_id="p99", predicted_is_relevant=False),
    ]
    with pytest.raises(AntiSkipError):
        assert_no_skips(cases, preds)
