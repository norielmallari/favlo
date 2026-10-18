# Benchmarks

As-built harness scores for this pipeline. The process contract is [../../ARCHITECTURE.md](../../ARCHITECTURE.md).

The harness scores IFEval, GSM8K, and HumanEval against `serve`. The `model` field on `POST /v1/chat/completions` names this pipeline. The checker reads `message.content`. A dropped connection restarts the server and sends the same request again.

## Request

<!-- request -->
- thinking off, temperature 0.7, top-p 0.8, top-k 20, presence 1.5, max tokens ifeval 2048, gsm8k 1024, humaneval 1024
<!-- /request -->

## Intelligence smoke

The smoke is the first 30 rows of each set, in file order.

<!-- smoke-ifeval -->
- ifeval: n=30  prompt strict 0.833  prompt loose 0.867  instruction strict 0.889  instruction loose 0.911
<!-- /smoke-ifeval -->

<!-- smoke-gsm8k -->
- gsm8k: n=30  29/30  0.967
<!-- /smoke-gsm8k -->

<!-- smoke-humaneval -->
- humaneval: n=30  29/30  0.967
<!-- /smoke-humaneval -->

## Sample of 100

The sample is the first 100 rows of each set, in file order.

<!-- sample-ifeval -->
- ifeval: n=100  prompt strict 0.690  prompt loose 0.770  instruction strict 0.791  instruction loose 0.847
<!-- /sample-ifeval -->

<!-- sample-gsm8k -->
- gsm8k: n=100  93/100  0.930
<!-- /sample-gsm8k -->

<!-- sample-humaneval -->
- humaneval: n=100  97/100  0.970
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
