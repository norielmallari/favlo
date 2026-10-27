# Coding primitives

The coding harness exposes six tools. `python -m harness.coding` is the client that drives them. This page is the contract those tools implement.

A tool result is one JSON object. A failure has `error` and `message`. The tool does not rewrite an argument or call another model.

| `error` | When |
|---|---|
| `invalid_arguments` | A field is missing, has the wrong type, or does not belong to that action |
| `not_found` | The file, directory, or job is absent |
| `permission_denied` | The path leaves the workspace, is reserved, or cannot be read |
| `conflict` | The create target exists, an idempotency key was reused with different arguments, or a file hash does not match |
| `execution_failed` | The tool could not finish the operation |
| `match_not_found` | `edit` found zero copies of `old_string` |
| `ambiguous_match` | `edit` found several copies and `replace_all` is false |
| `invalid_range` | A read offset or shell cursor is past the end |
| `binary_content` | The file contains a null byte or is not UTF-8 |

## Shared rules

Paths are resolved, including symlinks, before they are used. An absolute path is accepted when the resolved path stays inside the workspace. A path outside the workspace is `permission_denied`.

`.favlo` is harness state. `glob` and `grep_search` skip it. `write_file` and `edit` cannot change it. `read_file` can read `.favlo/artifacts/` and nothing else under `.favlo`.

Each call is appended to `.favlo/journal.jsonl` with `session_id`, `run_id`, `tool_call_id`, `tool`, `error`, and `message` when the call failed. The model-facing JSON does not include those ids. A call that returns an error is not counted as a successful use of that tool.

A boolean argument accepts `true` or `false` as text. An integer or number accepts a JSON number and a decimal numeral as text.

File creates and edits take a lock on that path. A shell job holds the workspace until it reaches a terminal state. `start` acquires that hold before it returns and does not wait for the command. Another file tool waits while the hold is taken. A new shell waits until file edits and the previous shell have finished. `status`, `input`, and `cancel` do not take the file lock.

`write_file`, `edit`, shell `start`, and shell `input` require `idempotency_key`. The same key and the same arguments return the recorded JSON and do not run again. An attempt that fails before it publishes a result drops the reservation, so the same key can be tried again. Different arguments for that key are `conflict`. A key is at most 256 characters. The recorded outcome is stored under `.favlo/idempotency/`.

Hashes are the lowercase hex SHA-256 of the complete file bytes.

## `read_file`

Reads a window of an existing text file. `offset` is zero-based and defaults to 0. Each returned line has a one-based `line` and the line `text`. `content_hash` covers the whole file, including a partial read. `next_offset` is present when more lines remain. An offset past the last line is `invalid_range`. An offset at the end returns no lines and no `next_offset`.

The window is at most 200 lines and 32,768 bytes of line text. A larger request returns a shorter window and `next_offset`. A file above 8 MiB is `execution_failed`. A missing file is `not_found`. A directory is `invalid_arguments`.

## `glob`

Lists files whose paths match `pattern`. It does not read file contents. The search root is `path`, or the workspace when `path` is omitted. The listing does not add parent directories.

The listing is `rg --files --hidden --no-require-git`. Hidden names are included when the pattern matches them. `.gitignore` rules apply even when the directory is not its own git repository, and a parent repository's ignore file applies too. A search that starts inside an ignored directory still lists the files there. A pattern such as `*.log` from a parent ignore file still hides matching names. `.ignore` applies. `.git` and `.favlo` are excluded. Ripgrep does not follow directory symlinks. A result whose resolved path leaves the workspace is dropped. The pattern is applied after that listing, so a positive glob does not put an ignored file back. A pattern with no slash matches the file name in any directory. `**` matches across directories. `..` and an absolute pattern are `invalid_arguments`.

Paths are workspace-relative and sorted. The default limit is 100 and the maximum is 1,000. `truncated` is true when more paths remain, and `next_cursor` is the continuation token.

## `grep_search`

Searches text with `rg --json`. `pattern` is a regular expression unless `literal` is true. `case_sensitive` defaults to true, which disables smart case. `path` is a file or directory. `glob` restricts the files after the search, using the same pattern rules as `glob`. `context_lines` adds `context_before` and `context_after`. Those lines are not part of `text`. The maximum context is 20 lines. There is one result per matching line.

No matches is a successful result with `match_count` 0. An invalid regular expression is `invalid_arguments`. The default limit is 50 and the maximum is 200. `truncated` and `next_cursor` continue the list. When the full result is larger than 32,768 bytes, the response also names `artifact`, a file under `.favlo/artifacts/` that holds every match.

This is text search. It does not resolve symbols.

## `write_file`

Creates a new UTF-8 file. The bytes are the `content` string with no added newline and no other rewrite. An empty string is a valid file. The destination must not exist. The write lands in a temporary file under `.favlo/tmp` and is linked into place only after the full write finishes. A failed attempt removes that temporary file. The result is `file_path` and `content_hash`.

`create_directories` defaults to false. A missing parent is then `not_found`. A parent that is a file is `invalid_arguments`. An existing destination is `conflict`. Content above 8 MiB is `invalid_arguments`.

## `edit`

Replaces an exact string in an existing text file. `old_string` must be non-empty. `new_string` may be empty. The match is non-overlapping and is not fuzzy. `replace_all` defaults to false. Zero matches returns `match_not_found`. Several matches with `replace_all` false returns `ambiguous_match`. The file is unchanged in both cases.

`expected_hash` is optional. It must be a SHA-256 hex digest. A mismatch returns `conflict` and does not write. The replacement is written to a temporary file and moved into place. The previous file mode is kept. The result has `replacements`, `before_hash`, `after_hash`, and `diff`. A diff longer than 80 lines sets `diff_truncated` and `diff_artifact`. The result file is still the full edit. A result above 8 MiB is `invalid_arguments` and is not written.

## `run_shell_command`

Starts or manages one asynchronous job. `action` defaults to `start`. Each action accepts only its own fields.

| Action | Required | Optional |
|---|---|---|
| `start` | `command`, `idempotency_key` | `directory`, `description`, `timeout_seconds` |
| `status` | `job_id` | `cursor`, `max_bytes` |
| `input` | `job_id`, `input`, `idempotency_key` | `close_stdin` |
| `cancel` | `job_id` | `grace_period_seconds` |

`start` returns `job_id`, `status` `queued`, and `next_cursor` `"0"` without waiting for the command. A submitted job is not evidence that the command succeeded. `status` returns the current record immediately. It does not wait. `output` is a byte window of the merged stdout and stderr. `has_more` is true when unread bytes remain or the job is not terminal. The default window is 8,192 bytes and the maximum is 65,536. `exit_code` is present only after a process exit was observed.

`input` writes to the pipe and can set `close_stdin`. It waits until the process is running, or until it has already ended, for at most 5 seconds. It does not wait for the command to finish. Stdin is a pipe. A `pty` field is rejected. `cancel` returns `cancel_requested` while the process is still alive. A later `status` returns `cancelled` after the process has been reaped. The grace period defaults to 2 seconds: `SIGTERM`, then `SIGKILL`. The same signals go to the process group, so descendants are included.

A `timeout_seconds` value starts that same termination and ends in `timed_out`. Output above 1 MiB ends the job as `failed` with `output_truncated`. The retained bytes stay readable.

The executable is `/bin/sh -c`. It is not a login shell and it does not read the process environment. The environment is `PATH=/usr/bin:/bin:/usr/sbin:/sbin`, `LANG=C.UTF-8`, `LC_ALL=C.UTF-8`, `PYTHONUNBUFFERED=1`, `PYTHONDONTWRITEBYTECODE=1`, and `HOME` set to the workspace. `directory` must resolve to a directory inside the workspace and defaults to the workspace root. The command string is host shell text. A path inside that string is not rewritten.

Job states:

```text
queued → running → succeeded
                 → failed
                 → timed_out
                 → cancelled
                 → interrupted
```

A job record under `.favlo/jobs/` is not a live process. When the session opens, a record that is still `queued` or `running` becomes `interrupted`. The command is not run again.

## Coding session

The session seeds `notes/brief.txt` when that file is absent. The prompt then asks for the six tools in order: find the note, search it, read it, create `ledger.py` with the difference reversed, edit that return to `after - before`, and run `python3 -c` to print `collect(15, 21)`. The shell step polls `status` until the job succeeds. The loop allows 16 rounds.

Exit 0 means the hidden suite passed and each tool returned successfully at least once. The suite is `harness/coding/checks/test_ledger.py`. It is not copied into the workspace.
