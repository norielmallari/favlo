# Benchmarks

As-built harness scores for this pipeline. The process contract is [../../ARCHITECTURE.md](../../ARCHITECTURE.md).

The harness scores IFEval, GSM8K, and HumanEval against `serve`. The `model` field on `POST /v1/chat/completions` names this pipeline. The checker reads `message.content`. A dropped connection restarts the server and sends the same request again.

## Request

<!-- request -->
- thinking off, temperature 0, presence 0, max tokens ifeval 2048, gsm8k 1024, humaneval 1024
- ifeval: the dataset prompt, scored by the published strict and loose checkers
- gsm8k: five training shots, then Question / Answer; the first #### number must match
- humaneval: the instruct prompt; the continuation after the published signature is tested
<!-- /request -->

## Intelligence smoke

The smoke is the first 30 rows of each set, in file order. This smoke is `Qwen3.6-35B-A3B-fp8-e20`.

<!-- smoke-ifeval -->
- ifeval: n=30  prompt strict 0.833  prompt loose 0.933  instruction strict 0.889  instruction loose 0.956
<!-- /smoke-ifeval -->

<!-- smoke-gsm8k -->
- gsm8k: n=30  29/30  0.967
<!-- /smoke-gsm8k -->

<!-- smoke-humaneval -->
- humaneval: n=30  30/30  1.000
<!-- /smoke-humaneval -->

## Sample of 100

The sample is the first 100 rows of each set, in file order. This sample is `Qwen3.6-35B-A3B-fp8-e20`.

<!-- sample-ifeval -->
- ifeval: n=100  prompt strict 0.840  prompt loose 0.890  instruction strict 0.896  instruction loose 0.933
<!-- /sample-ifeval -->

<!-- sample-gsm8k -->
- gsm8k: n=100  97/100  0.970
<!-- /sample-gsm8k -->

<!-- sample-humaneval -->
- humaneval: n=100  98/100  0.980
<!-- /sample-humaneval -->

## Full

<!-- full-ifeval -->
not run
<!-- /full-ifeval -->

<!-- full-gsm8k -->
not run
<!-- /full-gsm8k -->

<!-- full-humaneval -->
not run
<!-- /full-humaneval -->
