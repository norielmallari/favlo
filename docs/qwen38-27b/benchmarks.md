# Benchmarks

As-built harness scores for this pipeline. The process contract is [../../ARCHITECTURE.md](../../ARCHITECTURE.md).

The harness scores IFEval, GSM8K, and HumanEval against `serve`. The `model` field on `POST /v1/chat/completions` names this pipeline. The checker reads `message.content`. A dropped connection restarts the server and sends the same request again. The scores below were recorded on 22 October 2026.

## Request

<!-- request -->
- thinking off, temperature 0.7, top-p 0.8, top-k 20, presence 1.5, max tokens ifeval 2048, gsm8k 1024, humaneval 1024
<!-- /request -->

## Intelligence smoke

The smoke is the first 30 rows of each set, in file order.

<!-- smoke-ifeval -->
- ifeval: n=30  prompt strict 0.900  prompt loose 0.867  instruction strict 0.933  instruction loose 0.911
<!-- /smoke-ifeval -->

<!-- smoke-gsm8k -->
- gsm8k: n=30  28/30  0.933
<!-- /smoke-gsm8k -->

<!-- smoke-humaneval -->
- humaneval: n=30  30/30  1.000
<!-- /smoke-humaneval -->

## Sample of 100

The sample is the first 100 rows of each set, in file order.

<!-- sample-ifeval -->
- ifeval: n=100  prompt strict 0.740  prompt loose 0.750  instruction strict 0.810  instruction loose 0.822
<!-- /sample-ifeval -->

<!-- sample-gsm8k -->
- gsm8k: n=100  87/100  0.870
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
