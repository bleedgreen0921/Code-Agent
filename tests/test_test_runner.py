import socket

import pytest
from scripts.integration_pytest import no_real_http
from scripts.postgres_integration import validate_report


def test_synthetic_guard_blocks_python_network_and_counts_attempts():
    with no_real_http() as attempts:
        with socket.socket() as connection:
            with pytest.raises(AssertionError, match="disabled"):
                connection.connect(("127.0.0.1", 9))
        assert attempts == ["127.0.0.1"]


def test_report_requires_nonempty_zero_failure_error_skip(tmp_path):
    report = tmp_path / "pytest.xml"
    report.write_text('<testsuite><testcase name="ok"/></testsuite>')
    assert validate_report(report) == {
        "tests": 1,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
    }
    for body in (
        "",
        "<testcase><failure/></testcase>",
        "<testcase><error/></testcase>",
        "<testcase><skipped/></testcase>",
    ):
        report.write_text("<testsuite>" + body + "</testsuite>")
        with pytest.raises(RuntimeError, match="rejected"):
            validate_report(report)
