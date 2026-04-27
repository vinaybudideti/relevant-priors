from src.app.features import feature_vector, FEATURE_NAMES


def test_feature_vector_length():
    v = feature_vector("MRI BRAIN", "CT HEAD", "2024-01-01", "2023-01-01", 5)
    assert len(v) == 14
    assert len(FEATURE_NAMES) == 14

def test_same_string_feature():
    v = feature_vector("MRI BRAIN", "MRI BRAIN", "2024-01-01", "2023-01-01", 5)
    # same_string is index 0
    assert v[0] == 1.0

def test_handles_malformed_dates():
    # Must not raise even with bad dates
    v = feature_vector("MRI BRAIN", "CT HEAD", "not-a-date", "", 5)
    assert len(v) == 14
