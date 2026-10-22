"""Classify provider errors only after an adapter verifies the error envelope."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
import json
import math
import re
import time


@dataclass(frozen=True)
class ModelServiceFailure:
    kind: str
    code: str
    retry_after: float = 0.0


def classify_service_error(error: Mapping) -> ModelServiceFailure | None:
    # Error messages can contain an SSE/JSON provider response (not CLI prose).
    fields = ("error", "errorMessage", "code", "status", "status_code", "statusCode", "api_error")
    text = json.dumps({key: error[key] for key in fields if key in error},
                      ensure_ascii=False).lower()
    quota_codes = ("insufficient_quota", "billing_hard_limit_reached", "credit_balance_too_low")
    rate_codes = ("throttling", "rate_limit_error", "rate_limit_exceeded", "too_many_requests")
    code = next((code for code in quota_codes if code in text), "")
    kind = "model_quota_exhausted" if code else "model_rate_limited"
    code = code or next((code for code in rate_codes if code in text), "")
    if not code:
        if not re.search(r"\b429\b", text):
            return None
        code = "429"
    headers = error.get("headers")
    headers = headers if isinstance(headers, Mapping) else {}
    raw = error.get("retry_after", error.get("retryAfter"))
    if raw is None:
        raw = next((value for key, value in headers.items()
                    if str(key).lower() == "retry-after"), None)
    delay = 0.0
    if raw is not None:
        try:
            delay = float(raw)
        except (ValueError, TypeError):
            try:
                delay = parsedate_to_datetime(str(raw)).timestamp() - time.time()
            except (ValueError, TypeError, OverflowError):
                pass
    return ModelServiceFailure(kind, code, max(0.0, delay) if math.isfinite(delay) else 0.0)
