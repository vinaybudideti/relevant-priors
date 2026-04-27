from fastapi.testclient import TestClient
from src.app.main import app


def _make_request(n_cases, priors_per_case):
    return {
        "cases": [
            {
                "case_id": f"c{i}",
                "current_study": {
                    "study_id": f"cur_{i}",
                    "study_description": "MRI BRAIN",
                    "study_date": "2024-01-01",
                },
                "prior_studies": [
                    {
                        "study_id": f"p_{i}_{j}",
                        "study_description": "CT HEAD",
                        "study_date": "2023-01-01",
                    }
                    for j in range(priors_per_case)
                ]
            }
            for i in range(n_cases)
        ]
    }


def test_healthz():
    with TestClient(app) as client:
        r = client.get('/healthz')
        assert r.status_code == 200

def test_readyz():
    with TestClient(app) as client:
        r = client.get('/readyz')
        assert r.status_code == 200

def test_one_prediction_per_prior():
    with TestClient(app) as client:
        body = _make_request(3, 4)
        r = client.post('/predict', json=body)
        assert r.status_code == 200
        preds = r.json()['predictions']
        assert len(preds) == 12

def test_empty_priors():
    with TestClient(app) as client:
        body = {
            "cases": [{
                "case_id": "c0",
                "current_study": {
                    "study_id": "cur", "study_description": "X",
                    "study_date": "2024-01-01"
                },
                "prior_studies": []
            }]
        }
        r = client.post('/predict', json=body)
        assert r.status_code == 200
        assert r.json()['predictions'] == []

def test_extra_fields_ignored():
    body = _make_request(1, 1)
    body['cases'][0]['unexpected_field'] = 'ignore me'
    body['cases'][0]['current_study']['extra'] = 42
    with TestClient(app) as client:
        r = client.post('/predict', json=body)
        assert r.status_code == 200

def test_malformed_dates():
    body = _make_request(1, 2)
    body['cases'][0]['current_study']['study_date'] = 'not-a-date'
    body['cases'][0]['prior_studies'][0]['study_date'] = ''
    with TestClient(app) as client:
        r = client.post('/predict', json=body)
        assert r.status_code == 200
        assert len(r.json()['predictions']) == 2

def test_huge_payload():
    body = _make_request(20, 50)
    with TestClient(app) as client:
        r = client.post('/predict', json=body)
        assert r.status_code == 200
        assert len(r.json()['predictions']) == 1000

def test_duplicate_priors_emit_two_predictions():
    body = {
        "cases": [{
            "case_id": "c0",
            "current_study": {
                "study_id": "cur", "study_description": "X",
                "study_date": "2024-01-01"
            },
            "prior_studies": [
                {"study_id": "dup", "study_description": "Y", "study_date": "2023-01-01"},
                {"study_id": "dup", "study_description": "Y", "study_date": "2023-01-01"},
            ]
        }]
    }
    with TestClient(app) as client:
        r = client.post('/predict', json=body)
        assert r.status_code == 200
        assert len(r.json()['predictions']) == 2

def test_stub_returns_all_false():
    body = _make_request(2, 3)
    with TestClient(app) as client:
        r = client.post('/predict', json=body)
        assert r.status_code == 200
        for p in r.json()['predictions']:
            assert p['predicted_is_relevant'] is False
