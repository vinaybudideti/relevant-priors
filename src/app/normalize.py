import re

ABBREV = [
    (r'\bCNTRST\b', 'CONTRAST'), (r'\bCONTR\b', 'CONTRAST'),
    (r'\bWO\b', 'WITHOUT'), (r'\bW/O\b', 'WITHOUT'),
    (r'\bW/\b', 'WITH '),
    (r'\bW CON\b', 'WITH CONTRAST'), (r'\bWO CON\b', 'WITHOUT CONTRAST'),
    (r'\bBI\b', 'BILATERAL'),
    (r'\bMAM\b', 'MAMMOGRAPHY'), (r'\bMAMMO\b', 'MAMMOGRAPHY'),
    (r'\bABD\b', 'ABDOMEN'), (r'\bPEL\b', 'PELVIS'),
    (r'\bMRI\b', 'MR'),
    (r'\bUS\b', 'ULTRASOUND'),
    (r'\bXR\b', 'XRAY'), (r'\bX-RAY\b', 'XRAY'),
    (r'\bECHO\b', 'ECHOCARDIOGRAM'), (r'\bTTE\b', 'ECHOCARDIOGRAM'),
    (r'\bTOMO\b', 'TOMOSYNTHESIS'), (r'\bDBT\b', 'TOMOSYNTHESIS'),
]

MODALITY_TOKENS = {
    'CT': ['CT'], 'MR': ['MR'], 'XRAY': ['XRAY'],
    'ULTRASOUND': ['ULTRASOUND'], 'MAMMOGRAPHY': ['MAMMOGRAPHY'],
    'NM': ['NM', 'SCINT'], 'PET': ['PET', 'FDG'],
    'DEXA': ['DEXA'], 'ECHO': ['ECHOCARDIOGRAM'],
    'CTA': ['CTA'], 'MRA': ['MRA'],
}

REGION_TOKENS = {
    'HEAD': ['HEAD', 'BRAIN', 'SKULL'],
    'NECK': ['NECK', 'THYROID', 'CAROTID'],
    'CHEST': ['CHEST', 'THORAX', 'LUNG', 'RIB'],
    'CARDIAC': ['HEART', 'CARDIAC', 'CORONARY'],
    'BREAST': ['BREAST', 'MAMMOGRAPHY'],
    'ABDOMEN': ['ABDOMEN', 'LIVER', 'PANCREAS', 'SPLEEN', 'RENAL', 'KIDNEY'],
    'PELVIS': ['PELVIS', 'BLADDER', 'PROSTATE', 'UTERUS', 'OVARY'],
    'CSPINE': ['CSPINE', 'CERVICAL'],
    'LSPINE': ['LSPINE', 'LUMBAR'],
    'SHOULDER': ['SHOULDER'], 'KNEE': ['KNEE'], 'HIP': ['HIP'],
    'ANKLE': ['ANKLE'], 'WRIST': ['WRIST'], 'ELBOW': ['ELBOW'], 'FOOT': ['FOOT'],
}


def normalize(s: str) -> str:
    if not isinstance(s, str):
        return ''
    t = s.upper()
    t = re.sub(r'[^\w\s/+\-]', ' ', t)
    for pat, rep in ABBREV:
        t = re.sub(pat, rep, t)
    return ' '.join(t.split())


def extract_modality(normed: str) -> str:
    for canon, toks in MODALITY_TOKENS.items():
        for tok in toks:
            if tok in normed:
                return canon
    return 'UNK'


def extract_region(normed: str) -> str:
    for canon, toks in REGION_TOKENS.items():
        for tok in toks:
            if tok in normed:
                return canon
    return 'UNK'


def extract_contrast(normed: str) -> str:
    if 'WITHOUT CONTRAST' in normed:
        return 'WITHOUT'
    if 'WITH CONTRAST' in normed:
        return 'WITH'
    return 'UNK'


def canonical_key(desc: str) -> str:
    n = normalize(desc)
    return f"{extract_modality(n)}|{extract_region(n)}|{extract_contrast(n)}"
