# Ceilings

Kernel ceilings from `uv run favlo --wan22-ti2v-5b ceilings` on 25 October 2026. Each time is one command buffer, then a wait, the median of three dispatches after one discarded sample. Host uploads sit outside the timed region.

These rows time the 576×320, 9-frame token grid: four knots, 30 blocks, token rows padded from 540 to 576. The served smoke is 832×480 and 9 frames for both clips. A native clip is 1280×704, 121 frames, token rows padded from 27280 to 27328. Self-attention counts one query tile of 64 against 27296 keys.

| Kernel | Shape | Count | Seconds |
|---|---|---:|---:|
| `gemm_i8` + `scale_i8` | 576×3072, K 3072 | 960 | 0.001500 |
| `gemm_i8` + `scale_i8` | 576×14336, K 3072 | 120 | 0.002491 |
| `gemm_i8` + `scale_i8` | 576×3072, K 14336 | 120 | 0.002600 |
| `gemm_f16_tn` | 576×576, K 128 | 2880 | 0.000210 |
| `gemm_f16_nn` | 576×128, K 576 | 2880 | 0.000188 |
| `gemm_f16_tn` | 576×512, K 128 | 2880 | 0.000182 |
| `gemm_f16_nn` | 576×128, K 512 | 2880 | 0.000190 |

The smoke DiT sum is **4.267 s**. Text, the VAE, and the host copies around these launches are outside that sum.

| Kernel | Shape | Count | Seconds |
|---|---|---:|---:|
| `gemm_i8` + `scale_i8` | 27328×3072, K 3072 | 960 | 0.011822 |
| `gemm_i8` + `scale_i8` | 27328×14336, K 3072 | 120 | 0.053226 |
| `gemm_i8` + `scale_i8` | 27328×3072, K 14336 | 120 | 0.045727 |
| `gemm_f16_tn` | 64×27296, K 128 | 1,229,760 | 0.000233 |
| `gemm_f16_nn` | 64×128, K 27296 | 1,229,760 | 0.000554 |

The native linears sum to **23.224 s**. The native down GEMM reached 52.64 TOPS. The native attention sum is **967.822 s**, one launch per query tile per head. A streaming copy was not in this command, so this note does not state a bandwidth ceiling.
