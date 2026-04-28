"""Load the public eval JSON into a flat priors DataFrame.

One row per (case_id, study_id) prior. Used by every method's harness run.
Returns the same logical shape the notebook produces in its Cell 8.
"""
import json

import pandas as pd


def load_priors_df(path: str = "data/relevant_priors_public.json") -> pd.DataFrame:
    """Read the public eval JSON, build a flat priors DataFrame.

    Returns a DataFrame with columns:
      case_id, study_id, current_desc, prior_desc,
      current_date, prior_date, n_priors, label
    Index is 0..N-1 so harness code can use df.index.values to slice
    a precomputed feature matrix aligned with this DataFrame.
    """
    with open(path) as f:
        raw = json.load(f)

    label_map = {}
    for t in raw["truth"]:
        v = t["is_relevant_to_current"]
        if isinstance(v, str):
            v = v.strip().lower() == "true"
        elif not isinstance(v, bool):
            v = bool(v)
        label_map[(t["case_id"], t["study_id"])] = v

    rows = []
    for c in raw["cases"]:
        cid = c["case_id"]
        cd = c["current_study"]["study_description"]
        cdate = c["current_study"]["study_date"]
        n_priors = len(c["prior_studies"])
        for p in c["prior_studies"]:
            key = (cid, p["study_id"])
            if key not in label_map:
                continue
            rows.append({
                "case_id": cid,
                "study_id": p["study_id"],
                "current_desc": cd,
                "prior_desc": p["study_description"],
                "current_date": cdate,
                "prior_date": p["study_date"],
                "n_priors": n_priors,
                "label": label_map[key],
            })
    return pd.DataFrame(rows).reset_index(drop=True)
