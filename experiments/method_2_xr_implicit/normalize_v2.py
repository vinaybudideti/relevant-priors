"""Method 2: XR-implicit detection.

Detects view-pattern markers in descriptions that lack explicit XR/X-RAY tokens
and infers modality=XRAY. Region detection then uses XR-specific region tokens.

Non-XR descriptions are unaffected — modality detection falls through to v1's logic.
"""
import re

# Reuse v1 primitives where unchanged
from src.app.normalize import (
    normalize,
    extract_contrast,
    MODALITY_TOKENS,
    REGION_TOKENS,
)

# View-pattern regexes — presence of any of these (in absence of explicit modality
# tokens) signals a plain radiograph
VIEW_PATTERNS = [
    re.compile(r'\b\d+\s*VIEWS?\b'),                       # "2 VIEWS", "3 VIEW"
    re.compile(r'\bMIN\s+\d+\s+VIEWS?\b'),                 # "MIN 3 VIEWS"
    re.compile(r'\bAP\s*[/&]\s*LAT(ERAL|RL)?\b'),          # "AP/LAT", "AP & LATERAL"
    re.compile(r'\bPA\s*[/&]\s*LAT(ERAL|RL)?\b'),          # "PA/LAT"
    re.compile(r'\bFRONTAL(\s+(AND|&)\s+LATER(AL|L))?\b'), # "FRONTAL", "FRONTAL & LATRL"
    re.compile(r'\b[123]V\b'),                              # "1V", "2V", "3V"
    re.compile(r'\bOBLIQUE\s+VIEW'),                        # "OBLIQUE VIEW"
]

# Tokens that indicate the description is NOT a plain radiograph even if it has
# view-like words. If any of these appear, do NOT infer XRAY.
NON_XR_DISQUALIFIERS = [
    'CT', 'MR', 'MRI', 'ULTRASOUND', 'US ',
    'MAMMO', 'MAM ', 'MAMMOGRAPHY',
    'PET', 'NM ', 'SCINT', 'DEXA', 'DXA',
    'ECHO', 'TTE', 'TEE',
    'CTA', 'MRA',
    'FLUORO', 'BARIUM', 'ESOPHAGRAM',
]

# XR-typical regions (subset of REGION_TOKENS that appear on plain radiographs).
# We accept any v1 region token, but this set documents what we expect.
XR_TYPICAL_REGIONS = {
    'CHEST', 'CSPINE', 'LSPINE',
    'SHOULDER', 'KNEE', 'HIP', 'ANKLE', 'WRIST', 'ELBOW', 'FOOT',
    'PELVIS',  # plain pelvis x-rays exist
}


def has_view_pattern(normed: str) -> bool:
    """True iff the normalized description matches any view-pattern regex."""
    return any(p.search(normed) for p in VIEW_PATTERNS)


def has_explicit_non_xr_modality(normed: str) -> bool:
    """True iff a non-XR modality token appears in the description.

    Used to disqualify XR inference when the description is clearly a different modality.
    """
    return any(tok in normed for tok in NON_XR_DISQUALIFIERS)


def extract_modality_v2(normed: str) -> str:
    """Modality extraction with XR-implicit inference.

    Order:
      1. Try v1 explicit-token detection (CT, MR, XRAY, ULTRASOUND, etc.)
      2. If still UNK and has a view pattern AND no non-XR disqualifier, infer XRAY
      3. Otherwise UNK
    """
    # Step 1: replicate v1 logic
    for canon, toks in MODALITY_TOKENS.items():
        for tok in toks:
            if tok in normed:
                return canon

    # Step 2: XR-implicit inference
    if has_view_pattern(normed) and not has_explicit_non_xr_modality(normed):
        return 'XRAY'

    # Step 3: still unknown
    return 'UNK'


def extract_region_v2(normed: str) -> str:
    """Region extraction — same as v1 for now.

    Method 2 doesn't change region detection; XR-typical regions are already
    in v1's REGION_TOKENS. This function is provided for symmetry and to make
    canonical_key self-contained within this module.
    """
    for canon, toks in REGION_TOKENS.items():
        for tok in toks:
            if tok in normed:
                return canon
    return 'UNK'


def canonical_key(desc: str) -> str:
    """Method 2 canonical key: 3-part (modality|region|contrast) with XR inference."""
    n = normalize(desc)
    mod = extract_modality_v2(n)
    reg = extract_region_v2(n)
    con = extract_contrast(n)
    return f"{mod}|{reg}|{con}"
