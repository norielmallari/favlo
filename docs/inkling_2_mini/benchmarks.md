# Benchmarks

As-built harness scores for this pipeline. The process contract is
[../../ARCHITECTURE.md](../../ARCHITECTURE.md).

- The harness scores IFEval, GSM8K, and HumanEval against `serve`.
- The `model` field names this pipeline. The checker reads `message.content`.
- A dropped connection unloads the resident model and sends the same request again.

## Request

<!-- request -->
- thinking off, temperature 0, presence 0, max tokens ifeval 2048, gsm8k 1024, humaneval 1024
- ifeval: the dataset prompt, scored by the published strict and loose checkers
- gsm8k: five training shots, then Question / Answer; the first #### number matches
- humaneval: the instruct prompt; the continuation after the signature is tested
<!-- /request -->

## Intelligence smoke

The smoke is the first 30 rows of each set, in file order. This smoke is nearest E4M3 attention and output projections, a float16 head, GPTQ-FOEM Metal fp4 experts in layers 0–19, MXFP8 experts in layers 20–23, and `half` keys and values.

<!-- smoke-ifeval -->
- ifeval: n=30  prompt strict 0.833  prompt loose 0.800  instruction strict 0.889  instruction loose 0.867
<!-- /smoke-ifeval -->

<!-- smoke-gsm8k -->
- gsm8k: n=30  26/30  0.867
<!-- /smoke-gsm8k -->

<!-- smoke-humaneval -->
- humaneval: n=30  30/30  1.000
<!-- /smoke-humaneval -->

<!-- smoke-mtp-on-ifeval -->
not run
<!-- /smoke-mtp-on-ifeval -->

<!-- smoke-mtp-on-gsm8k -->
not run
<!-- /smoke-mtp-on-gsm8k -->

<!-- smoke-mtp-on-humaneval -->
not run
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

The sample is the first 100 rows of each set, in file order.

<!-- sample-ifeval -->
not run
<!-- /sample-ifeval -->

<!-- sample-gsm8k -->
not run
<!-- /sample-gsm8k -->

<!-- sample-humaneval -->
not run
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
