# Benchmarks

The harness scores three published sets against `serve`. IFEval, GSM8K, and HumanEval turn thinking on at reasoning effort low. The draw is greedy, temperature 0. IFEval sends presence 1.5 so the trace does not repeat itself, and the checker reads the reply after the think block. GSM8K omits presence, so thinking mode uses 0. HumanEval sends presence 0. Each request is one chat completion. A leading close tag is removed. An empty HumanEval reply, or a reply with no function, is scored from the reasoning text when that text contains the function. The intelligence smoke is the first 30 rows of each file, in file order. The full run keeps those rows and continues through the rest. Responses resume from `artifacts/benchmarks`.

## Sets

IFEval is Zhou et al., Apache-2.0, 541 prompts, checker commit `e6890f85757d`. Thinking is on at reasoning effort low, and presence is 1.5. The user message is the published prompt. The checker reads the reply after the think block, and an unclosed trace scores as an empty reply. A reply may run to 2048 tokens. The report is prompt-level and instruction-level accuracy, strict and loose.

GSM8K is Cobbe et al., MIT, 1,319 test problems, data commit `b0bb162abedc`. Thinking is on at reasoning effort low. The user message is the question, and the last line of the reply must be `####` and the final number. A reply may run to 1536 tokens. The match is the first number on the `####` line. Words and a currency sign may sit between the marker and the number. A reply with no such line uses its last number.

HumanEval is Chen et al., MIT, 164 problems, checker commit `37c4dd63798c`. Thinking is on at reasoning effort low, and presence is 0, because a function body repeats indentation and keywords. The user message asks for the function. A python fence is reduced to the function body, then the published tests run in a subprocess with a 3 second limit. The score is pass@1 for one greedy sample. A reply may run to 1024 tokens.

```text
uv run python -m harness.benchmarks
uv run python -m harness.benchmarks --sample-only
uv run python -m harness.benchmarks --limit 100
uv run python -m harness.benchmarks --sample-only --only humaneval
```

## Intelligence smoke

The smoke is the first 30 rows. After the tenth row, GSM8K and HumanEval have to stay at or above 0.80, and IFEval has to stay at or above 0.70, through the thirtieth.

<!-- smoke -->
- GSM8K: n=30  27/30  0.900
- HumanEval: n=30  25/30  0.833
<!-- /smoke -->

## Sample of 100

<!-- sample -->
The run was stopped during HumanEval. IFEval and GSM8K are the first 100 rows. HumanEval is the first 40. Those requests sent presence 0.

- IFEval: n=100  prompt strict 0.320  prompt loose 0.340  instruction strict 0.436  instruction loose 0.448
- GSM8K: n=100  59/100  0.590
- HumanEval: n=40  32/40  0.800
<!-- /sample -->

## Full

<!-- full -->
not run
<!-- /full -->
