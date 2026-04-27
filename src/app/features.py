from datetime import datetime
import numpy as np
from .normalize import normalize, extract_modality, extract_region, extract_contrast

FEATURE_NAMES = [
    'same_string', 'same_modality', 'same_region',
    'cross_mod_same_region', 'same_mod_diff_region',
    'jaccard', 'gap_log', 'gap_le_30', 'gap_gt_5y',
    'n_priors_log', 'contrast_match',
    'both_breast', 'both_chest', 'cur_chest_prior_abd',
]


def feature_vector(current_desc: str, prior_desc: str,
                   current_date: str, prior_date: str,
                   n_priors: int) -> list:
    c = normalize(current_desc)
    p = normalize(prior_desc)
    cm = extract_modality(c); cr = extract_region(c); cc = extract_contrast(c)
    pm = extract_modality(p); pr = extract_region(p); pc = extract_contrast(p)

    try:
        gap = (datetime.fromisoformat(current_date) - datetime.fromisoformat(prior_date)).days
    except Exception:
        gap = 0

    c_tokens = set(c.split())
    p_tokens = set(p.split())
    union = c_tokens | p_tokens
    jaccard = len(c_tokens & p_tokens) / max(len(union), 1)

    return [
        float(c == p),
        float(cm == pm and cm != 'UNK'),
        float(cr == pr and cr != 'UNK'),
        float(cm != pm and cr == pr and cr != 'UNK'),
        float(cm == pm and cr != pr and cm != 'UNK'),
        jaccard,
        float(np.log1p(max(gap, 1))),
        float(gap <= 30),
        float(gap > 365 * 5),
        float(np.log1p(n_priors)),
        float(cc == pc and cc != 'UNK'),
        float(cr == 'BREAST' and pr == 'BREAST'),
        float(cr == 'CHEST' and pr == 'CHEST'),
        float(cr == 'CHEST' and pr == 'ABDOMEN'),
    ]
