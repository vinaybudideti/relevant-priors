from src.app.schemas import PredictRequest


def test_minimal_valid_request():
    body = {
        "cases": [{
            "case_id": "c1",
            "current_study": {
                "study_id": "s1", "study_description": "MRI BRAIN",
                "study_date": "2024-01-01"
            },
            "prior_studies": []
        }]
    }
    req = PredictRequest(**body)
    assert len(req.cases) == 1
    assert req.cases[0].case_id == "c1"

def test_extra_fields_ignored():
    body = {
        "challenge_id": "abc",
        "unknown_top_field": "ignore me",
        "cases": [{
            "case_id": "c1",
            "patient_name": "Doe, John",
            "current_study": {
                "study_id": "s1", "study_description": "X",
                "study_date": "2024-01-01",
                "extra_field": "ignore"
            },
            "prior_studies": []
        }]
    }
    req = PredictRequest(**body)
    assert req.cases[0].patient_name == "Doe, John"  # accepted but unused

def test_malformed_date_accepted():
    body = {
        "cases": [{
            "case_id": "c1",
            "current_study": {
                "study_id": "s1", "study_description": "X",
                "study_date": "not-a-real-date"
            }
        }]
    }
    # Must not raise
    req = PredictRequest(**body)
    assert req.cases[0].current_study.study_date == "not-a-real-date"
