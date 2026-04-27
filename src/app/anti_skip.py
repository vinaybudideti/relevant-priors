class AntiSkipError(AssertionError):
    pass


def assert_no_skips(request_cases, predictions):
    expected = {(c.case_id, p.study_id)
                for c in request_cases
                for p in c.prior_studies}
    got = {(p.case_id, p.study_id) for p in predictions}

    if expected != got:
        missing = expected - got
        extra = got - expected
        raise AntiSkipError(
            f"anti-skip violated: missing={len(missing)} extra={len(extra)}"
        )
    if len(predictions) != sum(len(c.prior_studies) for c in request_cases):
        raise AntiSkipError(
            f"prediction count mismatch: got {len(predictions)}"
        )
