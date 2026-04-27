import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar('request_id', default='-')

PHI_BLACKLIST = {'patient_name', 'patient_id', 'mrn'}


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            'ts': time.time(),
            'level': record.levelname,
            'msg': record.getMessage(),
            'request_id': request_id_var.get(),
        }
        if hasattr(record, 'extra_fields'):
            for k, v in record.extra_fields.items():
                if k.lower() in PHI_BLACKLIST:
                    continue
                payload[k] = v
        return json.dumps(payload, default=str)


def setup_logging():
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(h)
    root.setLevel(logging.INFO)


def new_request_id() -> str:
    rid = uuid.uuid4().hex[:12]
    request_id_var.set(rid)
    return rid


def log_event(name: str, **fields):
    safe = {k: v for k, v in fields.items() if k.lower() not in PHI_BLACKLIST}
    logger = logging.getLogger('app')
    record = logger.makeRecord('app', logging.INFO, '', 0, name, (), None)
    record.extra_fields = safe
    logger.handle(record)
