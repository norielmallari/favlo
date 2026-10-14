# Benchmarks

The harness scores three published sets against `serve`. Thinking is off, and the open think tag stays out of the draw. The draw is greedy, temperature 0. Presence is omitted, so instruct mode uses the model card value of 1.5. Each request is one chat completion. A leading close tag is removed, and the checker reads the reply that follows. The sample is the first 100 rows of each file, in file order. The full run keeps those rows and continues through the rest. Responses resume from `artifacts/benchmarks`.

## Sets

IFEval is Zhou et al., Apache-2.0, 541 prompts, checker commit `e6890f85757d`. The user message is the published prompt. A reply may run to 2048 tokens. The report is prompt-level and instruction-level accuracy, strict and loose.

GSM8K is Cobbe et al., MIT, 1,319 test problems, data commit `b0bb162abedc`. The user message is the question plus one line asking for a final `####` number. The match is the published `####` extractor.

HumanEval is Chen et al., MIT, 164 problems, checker commit `37c4dd63798c`. The user message asks for the function. A python fence is reduced to the function body, then the published tests run in a subprocess with a 3 second limit. The score is pass@1 for one greedy sample. A reply may run to 512 tokens. GSM8K replies may run to 768 tokens.

```text
uv run python -m harness.benchmarks
uv run python -m harness.benchmarks --sample-only
```

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
