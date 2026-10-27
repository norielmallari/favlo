"""Model-facing tool schemas for the six coding primitives."""

from __future__ import annotations

from typing import Any


def _tool(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str],
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": properties,
                "required": required,
            },
        },
    }


TOOLS: list[dict[str, Any]] = [
    _tool(
        "read_file",
        "Read a bounded portion of an existing text file. Returns numbered lines, "
        "the file's content hash, and next_offset when more content remains. "
        "offset is zero-based. Line numbers are one-based. The hash covers the "
        "whole file. This operation does not modify files. A binary file is an error.",
        {
            "file_path": {
                "type": "string",
                "description": "Absolute path or workspace-relative path.",
            },
            "offset": {
                "type": "integer",
                "description": "Zero-based starting line. Default 0.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum lines to return, subject to the harness cap.",
            },
        },
        ["file_path"],
    ),
    _tool(
        "glob",
        "Find files whose paths match a glob pattern. Returns workspace-relative "
        "paths without reading file contents. Paths are sorted. Hidden files are "
        "included when the pattern matches them. gitignore and .ignore rules apply. "
        ".git and .favlo are excluded. The search stays under path.",
        {
            "pattern": {"type": "string", "description": "Glob pattern, such as **/test_*.py."},
            "path": {"type": "string", "description": "Search root. Default is the workspace."},
            "limit": {"type": "integer", "description": "Maximum returned paths."},
            "cursor": {
                "type": "string",
                "description": "Continuation token from a previous result.",
            },
        },
        ["pattern"],
    ),
    _tool(
        "grep_search",
        "Search text within files using a literal string or regular expression. "
        "Returns matching paths, line numbers, and optional surrounding lines. "
        "No matches is a successful search. An invalid regular expression is an "
        "argument error. Matching text is separate from surrounding context. "
        "This is text search, not symbol resolution.",
        {
            "pattern": {"type": "string", "description": "Search expression."},
            "path": {"type": "string", "description": "File or directory to search."},
            "glob": {"type": "string", "description": "Restrict matching files."},
            "literal": {
                "type": "boolean",
                "description": "Treat the pattern as literal text. Default false.",
            },
            "case_sensitive": {"type": "boolean", "description": "Default true."},
            "context_lines": {"type": "integer", "description": "Surrounding lines per match."},
            "limit": {"type": "integer", "description": "Maximum returned matches."},
            "cursor": {"type": "string", "description": "Continuation token."},
        },
        ["pattern"],
    ),
    _tool(
        "write_file",
        "Create a new text file with the supplied content. Fails if the destination "
        "already exists. Use edit to modify an existing file. Returns the created "
        "path and content hash. The same idempotency key and arguments return the "
        "recorded outcome. Different arguments for that key are a conflict.",
        {
            "file_path": {"type": "string", "description": "Destination path."},
            "content": {"type": "string", "description": "Exact file content."},
            "create_directories": {
                "type": "boolean",
                "description": "Create missing parent directories. Default false.",
            },
            "idempotency_key": {
                "type": "string",
                "description": "Identifies this creation attempt across retries.",
            },
        },
        ["file_path", "content", "idempotency_key"],
    ),
    _tool(
        "edit",
        "Replace an exact string in an existing text file. By default the old string "
        "must occur exactly once. No fuzzy matching or automatic correction is "
        "performed. Zero matches returns match_not_found. Several matches with "
        "replace_all false returns ambiguous_match. A hash mismatch returns conflict.",
        {
            "file_path": {"type": "string", "description": "Existing file to modify."},
            "old_string": {
                "type": "string",
                "description": "Exact content to replace. Must not be empty.",
            },
            "new_string": {
                "type": "string",
                "description": "Replacement content. May be empty to delete.",
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace every exact occurrence. Default false.",
            },
            "expected_hash": {
                "type": "string",
                "description": "Reject the edit if the complete file has changed.",
            },
            "idempotency_key": {"type": "string", "description": "Deduplicate retries."},
        },
        ["file_path", "old_string", "new_string", "idempotency_key"],
    ),
    _tool(
        "run_shell_command",
        "Start or manage an asynchronous shell job. Starting a command returns a "
        "job id without waiting for completion. Use status to inspect progress, "
        "input to send stdin, and cancel to request termination. A submitted job "
        "is not evidence that its command succeeded. The default action is start. "
        "start requires command and idempotency_key. status requires job_id. "
        "input requires job_id, input, and idempotency_key. cancel requires job_id. "
        "Other combinations are rejected. The executable is /bin/sh -c with a fixed "
        "environment, not a login shell. Stdin is a pipe. A pty argument is rejected. "
        "The working directory stays inside the workspace. The command is host shell text.",
        {
            "action": {
                "type": "string",
                "enum": ["start", "status", "input", "cancel"],
                "description": "Default start.",
            },
            "command": {"type": "string", "description": "Shell command for start."},
            "directory": {
                "type": "string",
                "description": "Working directory inside the workspace. Default .",
            },
            "description": {"type": "string", "description": "Short label for the command."},
            "timeout_seconds": {
                "type": "number",
                "description": "Kill the job after this many seconds.",
            },
            "idempotency_key": {"type": "string", "description": "Required for start and input."},
            "job_id": {"type": "string", "description": "Required for status, input, and cancel."},
            "cursor": {"type": "string", "description": "Output byte offset for status."},
            "max_bytes": {"type": "integer", "description": "Maximum output bytes for status."},
            "input": {"type": "string", "description": "Stdin text for input."},
            "close_stdin": {"type": "boolean", "description": "Close stdin after this input."},
            "grace_period_seconds": {
                "type": "number",
                "description": "Seconds between SIGTERM and SIGKILL for cancel.",
            },
        },
        [],
    ),
]
