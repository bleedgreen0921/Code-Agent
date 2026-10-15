"""Reject real HTTP, failed/empty suites, and every skipped test (including xfail)."""

import socket
import sys
from contextlib import contextmanager
from unittest.mock import patch

import httpx
import httpx2
import pytest


@contextmanager
def no_real_http():
    attempts = []
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def guard_connect(original):
        def connect(sock, address):
            if sock.family in (socket.AF_INET, socket.AF_INET6):
                attempts.append(str(address[0]))
                raise AssertionError(
                    "Real HTTP/network connections are disabled in synthetic tests"
                )
            return original(sock, address)

        return connect

    def blocked(_transport, request):
        attempts.append(request.url.host)
        raise AssertionError("Real HTTP is disabled in synthetic tests")

    async def blocked_async(_transport, request):
        return blocked(_transport, request)

    with (
        patch.object(socket.socket, "connect", guard_connect(original_connect)),
        patch.object(socket.socket, "connect_ex", guard_connect(original_connect_ex)),
        patch.object(httpx.HTTPTransport, "handle_request", blocked),
        patch.object(httpx.AsyncHTTPTransport, "handle_async_request", blocked_async),
        patch.object(httpx2.HTTPTransport, "handle_request", blocked),
        patch.object(httpx2.AsyncHTTPTransport, "handle_async_request", blocked_async),
    ):
        yield attempts


class StrictResults:
    def __init__(self):
        self.skipped = 0

    def pytest_runtest_logreport(self, report):
        self.skipped += report.skipped

    def pytest_collectreport(self, report):
        self.skipped += report.skipped


def main(argv=None):
    results = StrictResults()
    with no_real_http() as attempts:
        code = int(
            pytest.main(sys.argv[1:] if argv is None else argv, plugins=[results])
        )
    if attempts or results.skipped:
        print(
            f"Rejected: {len(attempts)} real HTTP attempts; {results.skipped} skipped tests",
            file=sys.stderr,
        )
        return 1
    return code


if __name__ == "__main__":
    raise SystemExit(main())
