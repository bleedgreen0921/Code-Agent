---
name: isolated-repair
description: Repair an isolated candidate with versioned evidence and independent acceptance.
---
The user requested a candidate-only repair. Original workspace files are never changed.
Search current candidate code, read relevant functions, callers and exception paths,
then explain the cause with evidence and explicitly distinguish inference from facts.
Use read_file for current file SHA-256 and delivered complete line ranges.
Use edit_file with expected_hash and old_text, or create=true for a new file.
Shell commands must finish in the foreground; stop background processes before sealing.
Each successful write and completed Shell saves a checkpoint before success is returned.
Use candidate_diff to inspect baseline-to-candidate changes, including files and modes.
User staged, unstaged and untracked changes already belong to baseline B.
Use only the authorized development check IDs. Read actual execution records and logs.
Old check records identify old revisions; they cannot establish a changed version passed.
Historical reports are clues: reread current code and reconfirm their claims at baseline B.
If baseline acceptance passed, do not invent a failing baseline or reproduction.
Development tests may change, but independent image acceptance assets cannot change.
Submit RepairDraftV1 alone in one tool batch. Include every target's handling, unfinished
items, test changes, current source digests/lines, actual check references and limitations.
A report declaring unfinished or unconfirmed targets cannot produce a successful candidate.
Gate feedback continues this same graph, candidate and budgets. Fix failures and resubmit.
Only trusted gate records establish acceptance; model prose or Shell output cannot do so.
Passing acceptance proves its covered conditions, not all causal explanations.
Never perform delivery, merging, writeback, delegation or external operations.
