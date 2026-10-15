"""Unauthenticated business endpoints may only be reached from this machine."""

import ipaddress
from urllib.parse import urlsplit
from uuid import uuid4

from starlette.requests import Request

from contracts.errors import _error_response
from contracts.v1 import ErrorCode


def loopback(host: str | None) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host or "").is_loopback
    except ValueError:
        return False


class LoopbackOnly:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = {}
            for key, value in scope.get("headers", []):
                headers.setdefault(key, []).append(value.decode("latin-1"))
            hosts, origins = headers.get(b"host", []), headers.get(b"origin", [])
            valid = len(hosts) == 1 and loopback((scope.get("client") or ("",))[0])
            try:
                parsed = urlsplit("http://" + hosts[0]) if len(hosts) == 1 else None
                valid = (
                    valid
                    and parsed is not None
                    and loopback(parsed.hostname)
                    and not parsed.username
                    and not parsed.password
                    and not parsed.path
                    and not parsed.query
                    and not parsed.fragment
                )
                if parsed:
                    parsed.port
                if origins:
                    origin = urlsplit(origins[0])
                    valid = (
                        valid
                        and len(origins) == 1
                        and origin.scheme in {"http", "https"}
                        and loopback(origin.hostname)
                        and not origin.username
                        and not origin.password
                        and not origin.path
                        and not origin.query
                        and not origin.fragment
                    )
                    origin.port
            except ValueError:
                valid = False
            if not valid:
                scope.setdefault("state", {}).setdefault(
                    "request_id", f"req_{uuid4().hex}"
                )
                await _error_response(
                    Request(scope), 403, ErrorCode.FORBIDDEN, "Local access required"
                )(scope, receive, send)
                return
        await self.app(scope, receive, send)
