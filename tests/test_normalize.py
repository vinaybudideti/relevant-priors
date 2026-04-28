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


class TestModalityOrderRegressions:
    """Regression tests for modality-order bugs caught by the audit."""

    def test_cta_not_classified_as_ct(self):
        assert extract_modality(normalize("CTA HEAD WITH CONTRAST")) == "CTA"
        assert canonical_key("CTA HEAD WITH CONTRAST") == "CTA|HEAD|WITH"

    def test_mra_not_classified_as_mr(self):
        assert extract_modality(normalize("MRA BRAIN WITHOUT CONTRAST")) == "MRA"
        assert canonical_key("MRA BRAIN WITHOUT CONTRAST") == "MRA|HEAD|WITHOUT"

    def test_pet_ct_classified_as_pet(self):
        # PET/CT should be PET, not CT
        assert extract_modality(normalize("PET/CT SKULL TO THIGH")) == "PET"

    def test_pet_fdg_classified_as_pet(self):
        assert extract_modality(normalize("PET FDG WHOLE BODY")) == "PET"

    def test_dxa_normalized_to_dexa(self):
        assert extract_modality(normalize("DXA HIP SPINE")) == "DEXA"
        assert canonical_key("DXA Hip Spine") == "DEXA|HIP|UNK"

    def test_mammography_not_classified_as_mr(self):
        # MAMMOGRAPHY contains 'MR' as substring; must resolve to MAMMOGRAPHY
        assert extract_modality(normalize("MAMMOGRAPHY SCREENING BILATERAL")) == "MAMMOGRAPHY"

    def test_simple_ct_still_works(self):
        # Regression check: don't break the simple cases
        assert extract_modality(normalize("CT CHEST WITH CONTRAST")) == "CT"

    def test_simple_mr_still_works(self):
        assert extract_modality(normalize("MR LUMBAR SPINE")) == "MR"

    def test_mri_normalized_to_mr(self):
        # MRI gets normalized to MR via ABBREV first
        assert extract_modality(normalize("MRI BRAIN")) == "MR"
