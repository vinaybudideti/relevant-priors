import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .anti_skip import assert_no_skips, AntiSkipError
from .logging_setup import setup_logging, new_request_id, log_event
from .predictor import AllFalsePredictor
from .schemas import PredictRequest, PredictResponse, Prediction


PREDICTOR = None
READY = False
DEFAULT_FALLBACK = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global PREDICTOR, READY
    setup_logging()
    artifacts_dir = os.environ.get('ARTIFACTS_DIR', 'artifacts')
    use_stub = os.environ.get('USE_STUB_PREDICTOR', '1') == '1'

    if use_stub or not Path(artifacts_dir).exists():
        log_event('predictor_init', kind='stub_all_false')
        PREDICTOR = AllFalsePredictor()
    else:
        # Cascade predictor will be added in Chunk 5
        from .cascade import CascadePredictor
        log_event('predictor_init', kind='cascade', artifacts_dir=artifacts_dir)
        PREDICTOR = CascadePredictor(artifacts_dir)

    READY = True
    log_event('startup_complete')
    yield
    log_event('shutdown')


app = FastAPI(lifespan=lifespan)


@app.get('/healthz')
async def healthz():
    return {'status': 'ok'}


@app.get('/readyz')
async def readyz():
    if not READY:
        return JSONResponse({'status': 'starting'}, status_code=503)
    return {'status': 'ready'}


@app.post('/predict', response_model=PredictResponse)
async def predict(req: PredictRequest, request: Request):
    new_request_id()
    t0 = time.time()
    log_event('predict_start',
              cases=len(req.cases),
              total_priors=sum(len(c.prior_studies) for c in req.cases))

    by_key = {}
    keys_in_order = []
    items = []
    for c in req.cases:
        for p in c.prior_studies:
            key = (c.case_id, p.study_id)
            by_key[key] = DEFAULT_FALLBACK
            keys_in_order.append(key)
            items.append({
                'current_desc': c.current_study.study_description,
                'prior_desc':   p.study_description,
                'current_date': c.current_study.study_date,
                'prior_date':   p.study_date,
                'n_priors':     len(c.prior_studies),
            })

    try:
        preds = PREDICTOR.predict_batch(items)
        for k, v in zip(keys_in_order, preds):
            by_key[k] = bool(v)
    except Exception as e:
        log_event('predictor_error', error=type(e).__name__)

    predictions = [
        Prediction(case_id=cid, study_id=sid,
                   predicted_is_relevant=by_key[(cid, sid)])
        for (cid, sid) in keys_in_order
    ]

    try:
        assert_no_skips(req.cases, predictions)
    except AntiSkipError as e:
        log_event('anti_skip_violation', error=str(e))
        # Repair
        repaired = []
        for c in req.cases:
            for p in c.prior_studies:
                k = (c.case_id, p.study_id)
                repaired.append(Prediction(
                    case_id=c.case_id,
                    study_id=p.study_id,
                    predicted_is_relevant=by_key.get(k, DEFAULT_FALLBACK)
                ))
        predictions = repaired

    elapsed_ms = int((time.time() - t0) * 1000)
    log_event('predict_done', predictions=len(predictions), elapsed_ms=elapsed_ms)
    return PredictResponse(predictions=predictions)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log_event('unhandled_exception', error=type(exc).__name__, path=str(request.url.path))
    return JSONResponse({'error': 'internal'}, status_code=500)
