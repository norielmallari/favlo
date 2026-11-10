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

The smoke is the first 30 rows of each set, in file order. MTP on and MTP off are separate runs. MTP on was recorded on 30 October 2026 at depth 3.

<!-- smoke-mtp-on-ifeval -->
- ifeval, mtp on: n=30  prompt strict 0.833  prompt loose 0.867  instruction strict 0.889  instruction loose 0.911
<!-- /smoke-mtp-on-ifeval -->

<!-- smoke-mtp-on-gsm8k -->
- gsm8k, mtp on: n=30  28/30  0.933
<!-- /smoke-mtp-on-gsm8k -->

<!-- smoke-mtp-on-humaneval -->
- humaneval, mtp on: n=30  30/30  1.000
<!-- /smoke-mtp-on-humaneval -->

<!-- smoke-mtp-off-ifeval -->
not run
<!-- /smoke-mtp-off-ifeval -->

<!-- smoke-mtp-off-gsm8k -->
not run
<!-- /smoke-mtp-off-gsm8k -->

<!-- smoke-mtp-off-humaneval -->
not run
<!-- /smoke-mtp-off-humaneval -->

## Sample of 100

The sample is the first 100 rows of each set, in file order. This sample is `Qwen3.8-27B-fp8-e20`, recorded on 23 October 2026.

<!-- sample-ifeval -->
- ifeval: n=100  prompt strict 0.750  prompt loose 0.780  instruction strict 0.828  instruction loose 0.853
<!-- /sample-ifeval -->

<!-- sample-gsm8k -->
- gsm8k: n=100  93/100  0.930
<!-- /sample-gsm8k -->

<!-- sample-humaneval -->
- humaneval: n=100  94/100  0.940
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
