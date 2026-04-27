from src.app.normalize import (normalize, extract_modality, extract_region,
                                 extract_contrast, canonical_key)


def test_normalize_basic():
    assert normalize("MRI BRAIN STROKE LIMITED WITHOUT CONTRAST") == \
           "MR BRAIN STROKE LIMITED WITHOUT CONTRAST"

def test_normalize_handles_empty():
    assert normalize("") == ""
    assert normalize(None) == ""

def test_normalize_punctuation():
    assert normalize("CT HEAD WITHOUT CNTRST") == "CT HEAD WITHOUT CONTRAST"

def test_extract_modality():
    assert extract_modality("MR BRAIN") == "MR"
    assert extract_modality("CT CHEST") == "CT"
    assert extract_modality("XRAY HAND") == "XRAY"
    assert extract_modality("RANDOM TEXT") == "UNK"

def test_extract_region():
    assert extract_region("MR BRAIN") == "HEAD"
    assert extract_region("CT CHEST") == "CHEST"
    assert extract_region("XRAY KNEE") == "KNEE"

def test_extract_contrast():
    assert extract_contrast("MR BRAIN WITHOUT CONTRAST") == "WITHOUT"
    assert extract_contrast("CT WITH CONTRAST") == "WITH"
    assert extract_contrast("XRAY") == "UNK"

def test_canonical_key_deterministic():
    k1 = canonical_key("MRI BRAIN STROKE LIMITED WITHOUT CONTRAST")
    k2 = canonical_key("MRI BRAIN STROKE LIMITED WITHOUT CONTRAST")
    assert k1 == k2
    assert k1 == "MR|HEAD|WITHOUT"
