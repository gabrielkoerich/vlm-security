"""JSON over HTTP with the standard library, typed at the boundary.

Everything a remote service returns arrives as `object` and is narrowed with the helpers below,
so a malformed answer fails a type check here instead of somewhere deep in the pipeline.
"""

import json
import urllib.request
from http.client import HTTPResponse
from typing import TypeIs, cast


def is_list(value: object) -> TypeIs[list[object]]:
    return isinstance(value, list)


def is_dict(value: object) -> TypeIs[dict[str, object]]:
    return isinstance(value, dict)


def as_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    return float(value) if isinstance(value, int | float) else default


def as_str(value: object, default: str = "") -> str:
    return value if isinstance(value, str) else default


def field(obj: object, key: str) -> object:
    return obj.get(key) if is_dict(obj) else None


def loads(raw: str | bytes) -> object:
    # json's own signature says Any, object is the honest type for untrusted input
    return cast(object, json.loads(raw))


def fetch(req: urllib.request.Request | str, timeout: float) -> bytes:
    # urlopen is typed Any in the stdlib stubs, for http(s) it returns an HTTPResponse
    with cast(HTTPResponse, urllib.request.urlopen(req, timeout=timeout)) as resp:
        return resp.read()


def get_json(url: str, timeout: float, headers: dict[str, str] | None = None) -> object:
    return loads(fetch(urllib.request.Request(url, headers=headers or {}), timeout))


def post_json(url: str, body: object, timeout: float, headers: dict[str, str] | None = None) -> object:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    return loads(fetch(req, timeout))
