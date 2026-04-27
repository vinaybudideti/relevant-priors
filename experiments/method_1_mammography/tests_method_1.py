"""Unit tests for Method 1 canonical_key refinement."""
import pytest
from experiments.method_1_mammography.normalize_v2 import (
    canonical_key,
    extract_mammo_subtype,
    extract_laterality,
)
from src.app.normalize import canonical_key as v1_canonical_key


# ============================================================================
# Mammography descriptions: must use the new 5-part key
# ============================================================================

class TestMammographyKey:
    def test_screen_bilateral_with_tomo(self):
        # "MAM screen BI with tomo" → MAMMOGRAPHY|BREAST|UNK|TOMO|BIL
        # (TOMO takes precedence over SCREEN per the priority order)
        assert canonical_key("MAM screen BI with tomo") == "MAMMOGRAPHY|BREAST|UNK|TOMO|BIL"

    def test_screen_bilateral_no_tomo(self):
        assert canonical_key("MAMMOGRAPHY SCREENING BILATERAL") == "MAMMOGRAPHY|BREAST|UNK|SCREEN|BIL"

    def test_diagnostic_left(self):
        assert canonical_key("MAMMOGRAPHY DIAGNOSTIC LEFT") == "MAMMOGRAPHY|BREAST|UNK|DIAG|LEFT"

    def test_diagnostic_right(self):
        assert canonical_key("MAM DIAG RIGHT") == "MAMMOGRAPHY|BREAST|UNK|DIAG|RIGHT"

    def test_lt_rt_short_codes(self):
        assert canonical_key("MAM LT") == "MAMMOGRAPHY|BREAST|UNK|UNK|LEFT"
        assert canonical_key("MAM RT") == "MAMMOGRAPHY|BREAST|UNK|UNK|RIGHT"

    def test_lt_and_rt_present_means_bilateral(self):
        assert canonical_key("MAM LT RT") == "MAMMOGRAPHY|BREAST|UNK|UNK|BIL"

    def test_mammography_with_no_qualifiers(self):
        assert canonical_key("MAMMOGRAPHY") == "MAMMOGRAPHY|BREAST|UNK|UNK|UNK"

    def test_breast_ultrasound_uses_mammo_key(self):
        # Breast modality → triggers mammography key path even if ultrasound
        # (because the canonical key cares about mammography-style decisions)
        # Breast US should still get the 5-part key because BREAST is in REGION_TOKENS
        result = canonical_key("ULTRASOUND BREAST BILATERAL")
        # Modality is ULTRASOUND, region is BREAST → should still get 5-part key
        assert result == "ULTRASOUND|BREAST|UNK|UNK|BIL"


# ============================================================================
# Non-mammography descriptions: must match v1 exactly
# ============================================================================

class TestNonMammographyUnchanged:
    @pytest.mark.parametrize("desc", [
        "MRI BRAIN STROKE LIMITED WITHOUT CONTRAST",
        "CT CHEST WITH CONTRAST",
        "XRAY HAND",
        "ULTRASOUND ABDOMEN",
        "CT HEAD WITHOUT CNTRST",
        "MR LUMBAR SPINE",
        "PET FDG WHOLE BODY",
        "DEXA HIP SPINE",
    ])
    def test_non_mammography_matches_v1(self, desc):
        v1_key = v1_canonical_key(desc)
        v2_key = canonical_key(desc)
        assert v1_key == v2_key, (
            f"Non-mammography desc must use v1 key. "
            f"desc={desc!r}, v1={v1_key!r}, v2={v2_key!r}"
        )


# ============================================================================
# Subtype helper unit tests
# ============================================================================

class TestExtractMammoSubtype:
    def test_tomosynthesis_takes_priority(self):
        # If both TOMOSYNTHESIS and SCREEN are in the string, TOMO wins
        assert extract_mammo_subtype("MAMMOGRAPHY SCREENING TOMOSYNTHESIS") == "TOMO"

    def test_screen_when_no_tomo(self):
        assert extract_mammo_subtype("MAMMOGRAPHY SCREEN BILATERAL") == "SCREEN"

    def test_diagnostic(self):
        assert extract_mammo_subtype("MAMMOGRAPHY DIAGNOSTIC LEFT") == "DIAG"

    def test_unk_when_no_subtype(self):
        assert extract_mammo_subtype("MAMMOGRAPHY") == "UNK"


# ============================================================================
# Laterality helper unit tests
# ============================================================================

class TestExtractLaterality:
    def test_bilateral_explicit(self):
        assert extract_laterality("MAMMOGRAPHY BILATERAL") == "BIL"

    def test_left_only(self):
        assert extract_laterality("MAM LEFT") == "LEFT"

    def test_right_only(self):
        assert extract_laterality("MAM RIGHT") == "RIGHT"

    def test_lt_short_code(self):
        assert extract_laterality("MAM LT") == "LEFT"

    def test_rt_short_code(self):
        assert extract_laterality("MAM RT") == "RIGHT"

    def test_left_and_right_means_bilateral(self):
        assert extract_laterality("MAM LEFT RIGHT") == "BIL"

    def test_unk_when_no_laterality(self):
        assert extract_laterality("MAMMOGRAPHY") == "UNK"

    def test_left_token_inside_word_does_not_match(self):
        # Don't match LEFT inside a longer word like CLEFT or LEFTOVER
        assert extract_laterality("CLEFTLIP CLEFTPALATE") == "UNK"
