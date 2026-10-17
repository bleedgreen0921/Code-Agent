"""Image-only unittest/pytest event runner; protocol uses a separate file descriptor.

Project stdout/stderr never enters the result channel. Projects execute as untrusted
code in an isolated container; event evidence does not prove test semantics.
"""

import json
import os
import sys
import unittest
from pathlib import Path

PROTOCOL = 1


def run_unittest(definition, result):
    loader = unittest.TestLoader()
    sys.path.insert(0, str(Path.cwd()))
    if definition["cases"]:
        suite = loader.loadTestsFromNames(definition["cases"])
    else:
        suite = loader.discover(definition["paths"][0], pattern=definition["pattern"])
    result["selected"] = suite.countTestCases()
    result["collection_errors"] = list(loader.errors)

    class Events(unittest.TestResult):
        def startTest(self, test):
            super().startTest(test)
            result["started"].append(test.id())

        def stopTest(self, test):
            result["finished"].append(test.id())
            super().stopTest(test)

        def addSuccess(self, test):
            super().addSuccess(test)
            result["events"].append({"case": test.id(), "outcome": "passed"})

        def addFailure(self, test, err):
            super().addFailure(test, err)
            result["events"].append({"case": test.id(), "outcome": "failed"})

        def addError(self, test, err):
            super().addError(test, err)
            result["events"].append(
                {
                    "case": test.id(),
                    "outcome": "collection_error"
                    if isinstance(test, unittest.loader._FailedTest)
                    else "error",
                }
            )

        def addSkip(self, test, reason):
            super().addSkip(test, reason)
            result["events"].append({"case": test.id(), "outcome": "skipped"})

        def addExpectedFailure(self, test, err):
            super().addExpectedFailure(test, err)
            result["events"].append({"case": test.id(), "outcome": "xfailed"})

        def addUnexpectedSuccess(self, test):
            super().addUnexpectedSuccess(test)
            result["events"].append({"case": test.id(), "outcome": "xpassed"})

        def addSubTest(self, test, subtest, err):
            super().addSubTest(test, subtest, err)
            if err:
                result["events"].append(
                    {
                        "case": test.id(),
                        "outcome": "failed"
                        if issubclass(err[0], test.failureException)
                        else "error",
                    }
                )

    events = Events()
    suite.run(events)
    for _, detail in [*events.failures, *events.errors]:
        print(detail, file=sys.stderr)
    return 0 if events.wasSuccessful() and not loader.errors else 1


def run_pytest(definition, result):
    import pytest

    sys.path.insert(0, str(Path.cwd()))

    class Events:
        def pytest_collection_finish(self, session):
            result["selected"] = len(session.items)

        def pytest_collectreport(self, report):
            if report.failed:
                result["collection_errors"].append(str(report.longrepr))
            if report.skipped:
                result["events"].append({"case": report.nodeid, "outcome": "skipped"})

        def pytest_runtest_logstart(self, nodeid, location):
            result["started"].append(nodeid)

        def pytest_runtest_logfinish(self, nodeid, location):
            result["finished"].append(nodeid)

        def pytest_runtest_logreport(self, report):
            outcome = None
            if report.failed:
                outcome = "failed" if report.when == "call" else "error"
                if getattr(report, "wasxfail", None) or "XPASS(strict)" in str(
                    report.longrepr
                ):
                    outcome = "xpassed"
            elif report.skipped:
                outcome = "xfailed" if getattr(report, "wasxfail", None) else "skipped"
            elif report.when == "call":
                outcome = "xpassed" if getattr(report, "wasxfail", None) else "passed"
            if outcome:
                result["events"].append({"case": report.nodeid, "outcome": outcome})

    # Ignore project addopts/config and environment/plugin autoload. Project conftest
    # remains project test code and runs inside the same containment boundary.
    return int(
        pytest.main(
            [
                "-c",
                "/dev/null",
                "-o",
                "addopts=",
                "-p",
                "no:cacheprovider",
                "--basetemp=/tmp/pytest",
                "--",
                *definition["paths"],
            ],
            plugins=[Events()],
        )
    )


def run(request):
    result = {
        "protocol": PROTOCOL,
        "adapter": request["adapter"],
        "complete": False,
        "selected": 0,
        "started": [],
        "finished": [],
        "events": [],
        "collection_errors": [],
    }
    code = 2
    try:
        definition = request["definition"]
        os.chdir(Path("/workspace") / definition["cwd"])
        if request["adapter"] == "python-unittest-v1":
            code = run_unittest(definition, result)
        elif request["adapter"] == "python-pytest-v1":
            code = run_pytest(definition, result)
        else:
            raise ValueError("Unknown adapter")
        result["complete"] = True
    except BaseException as exc:
        result["collection_errors"].append(type(exc).__name__ + ": " + str(exc)[:2000])
    result["exit_code"] = code
    return result, code


def main():
    request = json.load(sys.stdin)
    # stdout is the protocol descriptor; immediately replace Python and OS stdout
    # with stderr so project prints and subprocess output cannot forge the protocol.
    protocol = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    result, code = run(request)
    raw = json.dumps(result, ensure_ascii=True).encode()
    with os.fdopen(protocol, "wb") as stream:
        stream.write(raw)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
