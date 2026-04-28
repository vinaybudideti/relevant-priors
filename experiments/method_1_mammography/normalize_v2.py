"""Method 1: Mammography canonical key refinement.

Extends v1's canonical_key with subtype + laterality dimensions FOR MAMMOGRAPHY ONLY.
Non-mammography descriptions return the v1 key unchanged.
"""
import re

# Reuse v1's normalizer primitives — do NOT reimplement them
from src.app.normalize import (
    normalize,
    extract_modality,
    extract_region,
    extract_contrast,
)


def extract_mammo_subtype(normed: str) -> str:
    """Returns SCREEN | DIAG | TOMO | UNK based on normalized description."""
    if 'TOMOSYNTHESIS' in normed:
        return 'TOMO'
    if 'SCREEN' in normed or 'SCREENING' in normed:
        return 'SCREEN'
    if 'DIAG' in normed or 'DIAGNOSTIC' in normed:
        return 'DIAG'
    return 'UNK'


def extract_laterality(normed: str) -> str:
    """Returns LEFT | RIGHT | BIL | UNK based on normalized description.

    BIL means bilateral. Detected by: explicit BILATERAL token, OR both LEFT and RIGHT
    tokens present.
    """
    # Pad with spaces so word-boundary checks don't miss leading/trailing tokens
    padded = ' ' + normed + ' '

    has_bilateral = 'BILATERAL' in normed
    has_left = ' LEFT ' in padded or ' LT ' in padded
    has_right = ' RIGHT ' in padded or ' RT ' in padded

    if has_bilateral:
        return 'BIL'
    if has_left and has_right:
        return 'BIL'
    if has_left:
        return 'LEFT'
    if has_right:
        return 'RIGHT'
    return 'UNK'


def canonical_key(desc: str) -> str:
    """Method 1 canonical key.

    For mammography descriptions: returns 5-part key with subtype + laterality.
    For all other descriptions: returns v1's 3-part key unchanged.
    """
    n = normalize(desc)
    mod = extract_modality(n)
    reg = extract_region(n)
    con = extract_contrast(n)

    is_mammography = (mod == 'MAMMOGRAPHY' or reg == 'BREAST')

    if is_mammography:
        subtype = extract_mammo_subtype(n)
        laterality = extract_laterality(n)
        return f"{mod}|{reg}|{con}|{subtype}|{laterality}"

    return f"{mod}|{reg}|{con}"
