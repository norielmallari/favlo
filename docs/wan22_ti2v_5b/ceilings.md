# Ceilings

Kernel ceilings from `uv run favlo --wan22_ti2v_5b ceilings` on 19 November 2026.

- Each time is one command buffer, then a wait.
- The time is the median of three dispatches after one discarded sample.
- Host uploads sit outside the timed region.
- A generation may submit several of these kernels before it waits.
- This table still times each kernel on its own.

These rows time the 576×320, 9-frame token grid.

- Four knots, 30 blocks. The int8 rows are padded from 540 tokens to 576.
- `attend_flash` uses the live 540 token rows and the 512 text rows. Sixteen queries share a key tile of 48. Query, key, and value are half. The destination is float.
- The recorded clips are 832×480 and 9 frames.
- A native clip is 1280×704, 121 frames. The int8 rows are padded from 27280 tokens to 27328. Flash attention uses the live 27280 rows.

| Kernel | Shape | Count | Seconds |
|---|---|---:|---:|
| `gemm_i8` + `scale_i8` | 576×3072, K 3072 | 960 | 0.001467 |
| `gemm_i8` + `scale_i8` | 576×14336, K 3072 | 120 | 0.003103 |
| `gemm_i8` + `scale_i8` | 576×3072, K 14336 | 120 | 0.002650 |
| `attend_flash` | 540×540, 24×128 | 120 | 0.000722 |
| `attend_flash` | 540×512, 24×128 | 120 | 0.000603 |

The isolated DiT sum is **2.258 s**. Text, the VAE, and the host copies around these launches are outside that sum.

| Kernel | Shape | Count | Seconds |
|---|---|---:|---:|
| `gemm_i8` + `scale_i8` | 27328×3072, K 3072 | 960 | 0.012388 |
| `gemm_i8` + `scale_i8` | 27328×14336, K 3072 | 120 | 0.055755 |
| `gemm_i8` + `scale_i8` | 27328×3072, K 14336 | 120 | 0.048801 |
| `attend_flash` | 27280×27280, 24×128 | 120 | 0.980029 |
| `attend_flash` | 27280×512, 24×128 | 120 | 0.016364 |

The native linears sum to **24.440 s**. The native down GEMM reached 49.32 TOPS.

The native attention sum is **119.567 s**. A streaming copy was not in this command, so this note does not state a bandwidth ceiling.
