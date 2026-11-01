---
name: maf-tools
description: >-
  Create and register Microsoft Agent Framework function tools so the model
  receives the schema and a result it can act on. Use when adding, binding,
  or debugging a FunctionTool, @tool, tool schema, tool error, or
  function_invocation_configuration in Favlo.
---

# MAF tools

Microsoft Agent Framework function tools. The published pages are [Using function tools](https://learn.microsoft.com/en-us/agent-framework/agents/tools/function-tools) and [Adding tools](https://learn.microsoft.com/en-us/agent-framework/journey/adding-tools). The invocation settings are [FunctionInvocationConfiguration](https://learn.microsoft.com/en-us/python/api/agent-framework-core/agent_framework.functioninvocationconfiguration).

Favlo opens the client in `src/favlo/chat_client.py` and passes `tools` to `Agent`. The coding tools are bound in `harness/coding/agent_tools.py` from the schemas in `harness/coding/tools/schemas.py`. The contract those tools implement is [docs/coding-primitives.md](../../../docs/coding-primitives.md).

## Define the tool

One tool, one operation. The name and the description are what the model uses to choose it. Say what it does, what each parameter means, and what it returns. Register only the tools this agent calls.

Pass the schema as `input_model` on `FunctionTool`, or as `schema` on `@tool`. A Pydantic model is validated and converted before the function runs. Use it when a constraint must hold. A JSON schema dictionary is kept as written and checked only for top-level `required`, `additionalProperties: false`, property `enum`, and primitive `type`. Nested keywords, `oneOf`, and `$ref` are not enforced. A dictionary schema is not an authorization boundary.

Values the model must not supply stay off the schema. Pass them through `FunctionInvocationContext`, or close over them in the bound function, as `coding_tools` closes over `ToolSession`.

A model often sends a boolean, integer, or number as text. Coerce those strings to the schema type before the framework type check. `harness/coding/agent_tools.py` does this in `_prepare_arguments`.

## Register it

```python
agent = Agent(
    client=client,
    name="coding",
    instructions=instructions,
    tools=coding_tools(session),
    default_options=options,
)
```

`tools` is a list of `FunctionTool`. A declaration with `func=None` is sent to the model and is not executed. Do not use that form for a tool this process must run.

Set the client limits next to the other invocation settings:

```python
client.function_invocation_configuration["allow_concurrent_invocation"] = False
client.function_invocation_configuration["include_detailed_errors"] = True
```

Turn concurrency off when the tools share a workspace, a file, or a shell. Leave it on only when every tool in that list is safe to run at the same time. `max_iterations` and `max_consecutive_errors_per_request` cap the rounds. The coding client sets both to the session round count.

`include_detailed_errors` defaults to false. A validation failure then returns `Error: Argument parsing failed.` An exception returns `Error: Function failed.` The field name and the exception stay on `Content.exception` and are not sent to the model. Set the flag only when that result stays on a trusted channel. The coding harness does. A channel that leaves the process does not, because the exception text can hold secrets.

`approval_mode="always_require"` pauses before a tool that deletes, sends, or spends. The coding tools use `never_require`.

## Keep a required field reachable

A name in `required` is rejected inside `_prepare_arguments`, before `func` runs. The model cannot see which field failed unless `include_detailed_errors` is true, and even then the text is prefixed with `Error: Argument parsing failed.`

A bookkeeping field the instructions do not name will be omitted. Do not leave that field only in `required`. Supply it in the tool's `_prepare_arguments` before `super()`, or drop it from `required` and return the failure from the function. The coding binding supplies a SHA-256 of the tool name and the other arguments for `write_file`, `edit`, and shell `start` or `input`. A call that sends `idempotency_key` keeps that key. The same omitted-key call returns the recorded outcome.

An expected failure is a string the function returns, in this repo `{"error","message"}`. Do not raise for a missing path, a bad argument, or a conflict. A raised exception becomes the generic function-failed line.

## Check

Call `FunctionTool.invoke`. A check of the inner function does not show what the model is sent.

- The tool `parameters()` are the schema dictionary, including each description and `required` name.
- A missing `required` field raises `TypeError` and the message names that field.
- An omitted idempotency key still runs, and the same arguments return the same JSON.
- A call that sends a key is not rewritten. A second key for a create that already landed is `conflict`.
- A boolean or number sent as text is accepted.
- An expected failure is the tool's JSON, not `Error: Argument parsing failed.`

`tests/test_coding_harness.py` is the pattern. `tests/test_coding_tools.py` covers the inner contract.
