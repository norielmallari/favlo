# MoE fp4 error

Measurement note for the `zonos2` pipeline. The serving contract is [../README.md](../README.md). This file records the per-layer fp4 relative error of the [Zyphra/ZONOS2](https://huggingface.co/Zyphra/ZONOS2) MoE experts. It does not record a generation-time sum.

## Remainder

Each of the 24 MoE layers, 3 through 26, is scored by the RMS of its BF16 expert matrix against the restored fp4 matrix, divided by the source RMS. Gate, up, and down of one expert share the worst of those three scores, and the layer score is the worst expert. The four highest layers are 26, 21, 3, and 22, and those layers are stored as fp8. A search on 24 October 2026 did not find this table.

## Predecessors

- Method. MixQuant ([arXiv:2309.17341](https://arxiv.org/abs/2309.17341)) scores a layer by the mean squared gap between the full-precision weights and the restored quantized weights, then keeps a wider code where that gap is large. The round-up E8M0 miss bound used as a feed-forward budget is recorded for a different checkpoint in [the 27B note](../../qwen38-27b/research/ffn-fp8-budget.md).
- Model. [Zyphra/ZONOS2-GGUF](https://huggingface.co/Zyphra/ZONOS2-GGUF) keeps attention, the router, the embeddings, and the speaker projection at F16 and quantizes every expert matrix at one width. Its Q4_K build reports mean logit KLD 0.072 against F16. [drbaph/ZONOS2-FP8](https://huggingface.co/drbaph/ZONOS2-FP8) stores every expert gate and up projection as FP8 E4M3 and leaves every down projection in BF16. [amal-david/Zyphra-ZONOS2-4bit](https://huggingface.co/amal-david/Zyphra-ZONOS2-4bit) stores every expert at 4-bit and attention and embeddings at 8-bit, and reports that 4-bit attention samples an early end of sequence.
- Measurement. [arXiv:2603.08747](https://arxiv.org/abs/2603.08747) ranks transformer components by the perplexity change of keeping that component in float16, on Qwen2.5 models. [olka-fi/Ling-3.0-flash-MXFP4](https://huggingface.co/olka-fi/Ling-3.0-flash-MXFP4) reports per-layer relative error for that checkpoint and finds it flat from layer 2 to layer 41. Neither scores this model.
- Mechanism. A generated frame reads the experts its router selects. Layer 26 selects two. A layer stored as fp8 costs its extra bytes on the frames that select it.

## Rule

The sort key is the maximum expert score in one layer. The budget is the largest whole count that does not exceed 20% of 24, which is 4. An equal score keeps the lower layer index. The other 20 MoE layers stay fp4. Dense layers 0, 1, 2, and 27 are outside this table.

## Command

`uv run favlo --zonos2 quantize --src <checkout>` writes `models/ZONOS2-fp8-fp4/ffn_error.json` before it packs that record. A later run reads that file and does not rescore. The table below is that file, rounded to six digits.

## Table

`score` is the worst expert in the layer. `store` is the width kept for that layer's gate, up, and down matrices.

| Layer | Score | Store |
|---:|---:|---|
| 3 | 0.122816 | fp8 |
| 4 | 0.120012 | fp4 |
| 5 | 0.119860 | fp4 |
| 6 | 0.120188 | fp4 |
| 7 | 0.121941 | fp4 |
| 8 | 0.121029 | fp4 |
| 9 | 0.119766 | fp4 |
| 10 | 0.119857 | fp4 |
| 11 | 0.120254 | fp4 |
| 12 | 0.120062 | fp4 |
| 13 | 0.119815 | fp4 |
| 14 | 0.119748 | fp4 |
| 15 | 0.120323 | fp4 |
| 16 | 0.120047 | fp4 |
| 17 | 0.119789 | fp4 |
| 18 | 0.120024 | fp4 |
| 19 | 0.120907 | fp4 |
| 20 | 0.121879 | fp4 |
| 21 | 0.129431 | fp8 |
| 22 | 0.122056 | fp8 |
| 23 | 0.121300 | fp4 |
| 24 | 0.121362 | fp4 |
| 25 | 0.121177 | fp4 |
| 26 | 0.131465 | fp8 |

## Search

24 October 2026. Queries: "ZONOS2 MoE fp4 fp8 mixed precision relative RMS quantization error layers Zyphra", then "ZONOS2 MXFP4 NVFP4 GGUF expert layer sensitivity quantization Clark arXiv 2606.24320", then "ZONOS2 expert weight reconstruction error per layer relative RMS MXFP4 promoted layers". The predecessors in this note are the overlaps those queries returned. The search did not find the 24-layer relative-error table above.

## Corrections
