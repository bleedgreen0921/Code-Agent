import logging
from collections.abc import Mapping
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from contracts.v1 import ErrorCode

log = logging.getLogger(__name__)


class ApiError(Exception):
    def __init__(self, status: int, code: ErrorCode, message: str, *, headers=None):
        self.status = status
        self.code = code
        self.message = message
        self.headers = headers


def _error_response(
    request: Request,
    status: int,
    code: ErrorCode,
    message: str,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    request_id = request.state.request_id
    response = JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message}, "request_id": request_id},
        headers=headers,
    )
    response.headers["X-Request-Id"] = request_id
    return response


def install_errors(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def api_error(request: Request, exc: ApiError):
        return _error_response(
            request, exc.status, exc.code, exc.message, headers=exc.headers
        )

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        return _error_response(
            request, 422, ErrorCode.INVALID_REQUEST, "Invalid request"
        )

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        code = (
            ErrorCode.NOT_FOUND
            if exc.status_code == 404
            else ErrorCode.FORBIDDEN
            if exc.status_code == 403
            else ErrorCode.UNAUTHENTICATED
            if exc.status_code == 401
            else ErrorCode.INVALID_REQUEST
        )
        return _error_response(
            request, exc.status_code, code, "Request failed", headers=exc.headers
        )

    @app.exception_handler(Exception)
    async def internal_error(request: Request, exc: Exception):
        log.exception("Unhandled API error", exc_info=exc)
        return _error_response(
            request, 500, ErrorCode.INTERNAL_ERROR, "Internal server error"
        )

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        request.state.request_id = f"req_{uuid4().hex}"
        response = await call_next(request)
        response.headers["X-Request-Id"] = request.state.request_id
        return response
