"""Unit tests for Method 2 XR-implicit detection."""
import pytest
from experiments.method_2_xr_implicit.normalize_v2 import (
    canonical_key,
    extract_modality_v2,
    has_view_pattern,
    has_explicit_non_xr_modality,
)
from src.app.normalize import canonical_key as v1_canonical_key, normalize


# ============================================================================
# XR-implicit detection: descriptions without XR token should now infer XRAY
# ============================================================================

class TestXrImplicitDetection:
    def test_chest_two_view_inferred_as_xr(self):
        # "CHEST 2 VIEW FRONTAL & LATRL" — no XR token, has view pattern
        assert canonical_key("CHEST 2 VIEW FRONTAL & LATRL") == "XRAY|CHEST|UNK"

    def test_chest_pa_lat_inferred(self):
        assert canonical_key("CHEST PA/LAT") == "XRAY|CHEST|UNK"

    def test_chest_ap_lat_inferred(self):
        assert canonical_key("CHEST AP/LAT") == "XRAY|CHEST|UNK"

    def test_knee_3_views_inferred(self):
        assert canonical_key("KNEE 3 VIEWS") == "XRAY|KNEE|UNK"

    def test_shoulder_min_3_views_inferred(self):
        assert canonical_key("SHOULDER MIN 3 VIEWS") == "XRAY|SHOULDER|UNK"

    def test_short_view_codes(self):
        assert canonical_key("CHEST 2V") == "XRAY|CHEST|UNK"
        assert canonical_key("HAND 1V") == "XRAY|UNK|UNK"  # HAND not in v1 REGION_TOKENS

    def test_oblique_view_inferred(self):
        assert canonical_key("WRIST OBLIQUE VIEW") == "XRAY|WRIST|UNK"


# ============================================================================
# Non-XR descriptions: must NOT be misidentified as XRAY
# ============================================================================

class TestNonXrUnchanged:
    def test_ct_chest_stays_ct(self):
        # CT description with view-like word should NOT become XRAY
        assert canonical_key("CT CHEST WITH CONTRAST") == "CT|CHEST|WITH"

    def test_mri_brain_stays_mr(self):
        assert canonical_key("MRI BRAIN WITHOUT CONTRAST") == "MR|HEAD|WITHOUT"

    def test_ct_with_view_like_word_stays_ct(self):
        # If "AP/LAT" appears alongside CT, CT wins (v1 explicit detection runs first)
        assert canonical_key("CT CHEST AP/LAT") == "CT|CHEST|UNK"

    def test_mammography_with_views_stays_mammo(self):
        # MAM tokens are in MODALITY_TOKENS so the explicit pass catches it before
        # XR inference runs
        assert canonical_key("MAMMOGRAPHY 4 VIEW") == "MAMMOGRAPHY|BREAST|UNK"

    def test_ultrasound_stays_us(self):
        assert canonical_key("ULTRASOUND ABDOMEN COMPLETE") == "ULTRASOUND|ABDOMEN|UNK"


# ============================================================================
# v1 keys must be preserved when no view pattern is present
# ============================================================================

class TestNoViewPatternMatchesV1:
    @pytest.mark.parametrize("desc", [
        "MRI BRAIN STROKE LIMITED WITHOUT CONTRAST",
        "CT CHEST WITH CONTRAST",
        "ULTRASOUND ABDOMEN",
        "CT HEAD WITHOUT CNTRST",
        "MR LUMBAR SPINE",
        "PET FDG WHOLE BODY",
        "DEXA HIP SPINE",
        "MAMMOGRAPHY SCREENING BILATERAL",
        "ECHO 2D Mmode transthorac TTE",
    ])
    def test_no_view_pattern_matches_v1(self, desc):
        v1_key = v1_canonical_key(desc)
        v2_key = canonical_key(desc)
        assert v1_key == v2_key, (
            f"Description without view pattern must match v1. "
            f"desc={desc!r}, v1={v1_key!r}, v2={v2_key!r}"
        )


# ============================================================================
# Helper unit tests
# ============================================================================

class TestHasViewPattern:
    def test_n_views(self):
        assert has_view_pattern(normalize("CHEST 2 VIEWS"))
        assert has_view_pattern(normalize("KNEE 3 VIEW"))

    def test_ap_lat_variants(self):
        assert has_view_pattern(normalize("CHEST AP/LAT"))
        assert has_view_pattern(normalize("CHEST PA/LAT"))

    def test_frontal_lateral(self):
        assert has_view_pattern(normalize("CHEST 2 VIEW FRONTAL & LATRL"))

    def test_short_codes(self):
        assert has_view_pattern(normalize("HAND 2V"))

    def test_oblique(self):
        assert has_view_pattern(normalize("WRIST OBLIQUE VIEW"))

    def test_no_pattern(self):
        assert not has_view_pattern(normalize("MRI BRAIN"))
        assert not has_view_pattern(normalize("CT CHEST WITH CONTRAST"))
        assert not has_view_pattern(normalize("ULTRASOUND ABDOMEN"))


class TestHasExplicitNonXrModality:
    @pytest.mark.parametrize("desc", [
        "CT CHEST 2 VIEW",
        "MR LUMBAR FRONTAL",
        "MRI BRAIN AP/LAT",
        "PET FDG",
        "MAMMO BILATERAL",
    ])
    def test_disqualifies_non_xr(self, desc):
        assert has_explicit_non_xr_modality(normalize(desc))

    @pytest.mark.parametrize("desc", [
        "CHEST 2 VIEW FRONTAL & LATRL",
        "KNEE 3 VIEWS",
        "WRIST OBLIQUE VIEW",
    ])
    def test_does_not_disqualify_actual_xr(self, desc):
        assert not has_explicit_non_xr_modality(normalize(desc))
