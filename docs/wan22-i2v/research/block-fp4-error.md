# Block fp4 error

Measurement note for the `wan22-i2v` pipeline. The serving contract is [../README.md](../README.md). This file records which blocks of [Wan-AI/Wan2.2-I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B) stay fp8 after the fp4 relative-error ranking. It does not record a generation-time sum.

## Remainder

Each of the forty blocks in an expert is scored by the maximum relative RMS error of its ten fp4 matrices: self and cross q, k, v, and o, and the two feed-forward matrices, after one K permutation. The eight highest blocks are stored as fp8. The high-noise blocks are 15, 16, 17, 18, 19, 20, 24, and 26. The low-noise blocks are 11, 12, 15, 17, 18, 21, 24, and 26. A search on 24 October 2026 did not find those two lists.

## Predecessors

- Method. MixQuant ([arXiv:2309.17341](https://arxiv.org/abs/2309.17341)) scores a layer by the mean squared gap between the full-precision weights and the restored quantized weights, then keeps a wider code where that gap is large.
- Model. [Kijai/WanVideo_comfy_fp8_scaled](https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled) stores every I2V block linear as fp8 E4M3. The official [Wan-AI/Wan2.2-I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B) card describes the high-noise and low-noise experts and does not name an fp4 block ranking.
- Measurement. [arXiv:2605.27003](https://arxiv.org/abs/2605.27003) ranks Wan2.2-I2V modules by cosine reconstruction error under W4A4 and finds the largest errors in `self_attn.o` and `ffn.2`, concentrated in the middle and deeper blocks. That note keeps at most five whole blocks in higher precision. [arXiv:2606.29337](https://arxiv.org/abs/2606.29337) keeps outlier FFN columns in higher precision and reports VBench quality within 2–3.5% of FP16. Neither publishes these eight-of-forty lists.
- Mechanism. One expert is resident for a step. A block stored as fp8 costs its extra bytes on every step that expert runs.

## Rule

The sort key is the worst of the ten relative errors in one block. The budget is 20% of 40, which is 8. An equal score keeps the lower block index. The other 32 blocks of that expert stay fp4. Text, time, and the head stay fp4.

## Command

`uv run favlo --wan22-i2v quantize --high <high.safetensors> --low <low.safetensors>` prints each block score and the winner list, then writes `high_promoted` and `low_promoted` in `models/Wan2.2-I2V-A14B-fp4/manifest.json`. The per-matrix errors are not stored in that file. The table below is those two lists.

## Table

| Expert | fp8 blocks |
|---|---|
| high-noise | 15, 16, 17, 18, 19, 20, 24, 26 |
| low-noise | 11, 12, 15, 17, 18, 21, 24, 26 |

## Search

24 October 2026. Queries: "Wan2.2-I2V-A14B fp8 blocks 15 16 17 18 19 20 24 26 high noise low noise 11 12 15", then "MixQuant Wan2.2-I2V-A14B fp4 relative RMS error promote 20 percent high-noise blocks". The predecessors in this note are the overlaps those queries returned. The search did not find the two lists above.

## Corrections

Later finds are appended here. The sections above stay as committed.

25 October 2026. The head weight and its bias are stored as float16 and sit outside this ranking. The eight-block lists above are the 20% pack.

25 October 2026. `--fp8-percent 50` stores twenty of the forty blocks in each expert as fp8 and writes `models/Wan2.2-I2V-A14B-fp8-e50`. The high-noise blocks are 2, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, and 27. The low-noise blocks are 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 24, and 26. Queries that day: "Wan2.2-I2V-A14B fp8 blocks 2 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 high noise", then "Wan2.2 I2V fp4 relative RMS 50 percent promote blocks low-noise 3 4 5 6 7 8 21 22 24 26". The search did not find those two lists.
