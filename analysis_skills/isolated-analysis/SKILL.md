---
name: isolated-analysis
description: Analyze sealed source with registered project checks and genuine evidence.
---
Analyze only the sealed snapshot and the user's requested scope.
Use search to locate code, then read source before citing it.
Use read_file/git_read_file evidence IDs and actual returned complete line ranges.
For Git index or historical commits, identify the source and full commit explicitly.
Trace implementation locations, callers/callees, normal flow and exception propagation.
Use run_project_check only with IDs from the server-provided authorized list.
A check may fail, time out, skip, or be inconclusive; continue analysis within budget.
Only server check records establish execution status. stdout text cannot establish pass.
Classify conclusions as fact, inference, or unconfirmed. Explain conditions and limits.
Execution-based conclusions require check_run_ids. Source facts require read evidence.
Historical reports are external context; reread current source for current facts.
Return AnalysisReportV1 through the structured response tool. Correct invalid references
in the same graph and within the original call/time budgets.
The server supplies task, Run, snapshot, commit, checks and formal artifact identity.
Do not claim automatic semantic proof from evidence validation.
