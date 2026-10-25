# Generation ceiling

Measurement note for the `z-image-turbo` pipeline. The serving contract is [../README.md](../README.md). The per-dispatch rows are in [../ceilings.md](../ceilings.md). This file records the isolated sum of one 1280 generation. It is a measurement of this shader library.

## Remainder

Eight velocity steps and the decoder, summed without overlap, are 202.87 s. A printed 1280 generation of the smoke prompt was 200.58 s. A search on 21 October 2026 did not find that sum.

## Predecessors

- Method. Cai et al. ([arXiv:2511.22699](https://arxiv.org/abs/2511.22699)) define the single-stream diffusion transformer. Apple, WWDC26 session 330, specifies `matmul2d`. The OCP MX specification gives the fp4 block.
- Task. [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo). [zimgturbo](https://pypi.org/project/zimgturbo/) reports 13.7 s for a 1024 image at 8 steps with int8 Metal kernels. [MLX_z-image](https://github.com/uqer1244/MLX_z-image) reports about 150 s and 240 s for a 1024 image at 9 steps with 4-bit MLX weights.
- Measurement. Those figures are other engines, another quantizer, and a 1024 edge. They do not publish this per-kernel sum at 1280.
- Mechanism. Overlapping command buffers are not added. The 90% test uses the 202.87 s kernel ceiling. The attention bandwidth ceiling is 1301 s. A 2 GiB copy finished in 16.674 ms, 257.59 GB/s counting the read and the write.

## Rule

Each listed kernel runs in its own command buffer. The sample is the median of five dispatches after one discarded sample. The sequence in the block sum is 6528 tokens. A noise refiner in the forward sees only the image tokens. Caption prefill, the context refiner, the time embed, the image embed, and the final readout are outside the sum.

## Command

`favlo --z-image-turbo kernels` checks a synthetic fp4 tile. That command does not print this table. No committed command regenerates the medians. Recompute [../ceilings.md](../ceilings.md) when a listed kernel changes.

## Table

| Piece | Time |
|---|---|
| 256 blocks | 195.74 s |
| Decoder | 7.13 s |
| Kernel sum | 202.87 s |
| Attention bandwidth ceiling | 1301 s |
| Printed generation | 200.58 s total, caption 0.18 s, decoder 9.86 s, steps 23.73–23.99 s |

## Search

21 October 2026. Query: "Z-Image-Turbo inference seconds per image 1280 Metal diffusion". The predecessors in this note are the overlaps that query returned. The search did not find the 202.87 s sum.

## Corrections

Later finds are appended here. The sections above stay as committed.

23 October 2026. The latent stays on device for patchify, the Euler update, the float32 time basis, the pad fill, and the image/caption join. The VAE scale and the uint8 clip are shaders. The listed block kernels are unchanged, so the 202.87 s sum stands. The added dispatches measure 11.6 ms at 1280 and sit outside that sum. A printed 1280 generation of the smoke prompt was 202.56 s, caption 3.10 s, decoder 9.75 s, steps 23.54–24.06 s.

23 October 2026. `attend_q4` scores four keys into registers and rescales the softmax once per tile. Eight queries share each key load. The same 6528×30×128 dispatch measured 242.959 ms, against 528.848 ms for the previous tile. Two hundred fifty-six calls are 62.60 s. With the other block rows unchanged, one block is 490.5 ms, the 256 blocks are 125.57 s, and the generation sum is 132.70 s. The full-walk bandwidth line stays 1301 s. A printed 1280 generation of the smoke prompt was 134.36 s, caption 0.17 s, decoder 9.90 s, steps 15.48–15.72 s.

23 October 2026. A search the same day did not find the 132.70 s sum. [z-image-swift](https://github.com/xocialize/z-image-swift) reports about 13 s for a bf16 1024 image at 8 steps, and 11.07 s for a 768 bf16 run on an M5 Max. [zimage-mnn](https://pypi.org/project/zimage-mnn/) reports about 13 s per step and about 2 minutes end to end for a 1024 image at 8 steps on an M5 Pro with int8 Metal weights. A ComfyUI bf16 run on an M5 Max reports 13.1 s for a loaded 1024 image at 8 steps ([Voyager](https://voyager.so/blog/local-image-generation-vs-api-cost)). Those figures are other engines, other quants, and a 768 or 1024 edge. Two hundred fifty-six calls of the measured 242.959 ms tile are 62.20 s. The generation sum stays 132.70 s, from the 125.57 s block total and the 7.13 s decoder. The listed kernels are unchanged.

25 October 2026. `attend_flash` multiplies a 32×32 tile with `matmul2d` and keeps that score tile in the threadgroup. The same 6528×30×128 dispatch measured 97.455 ms, against 242.959 ms for `attend_q4`. Two hundred fifty-six calls are 24.95 s. With the other block rows unchanged, one block is 344.989 ms, the 256 blocks are 88.32 s, and the generation sum is 95.45 s. The attention read is 41,010,462,720 bytes per call, 40.76 s at 257.59 GB/s. A search the same day for that 95.45 s sum did not find it. [zimgturbo](https://pypi.org/project/zimgturbo/) reports 13.7 s for a 1024 image at 8 steps. That figure is another engine, another quantizer, and a 1024 edge. A printed 1280 generation of the smoke prompt was 100.65 s, caption 0.35 s, decoder 10.43 s, steps 11.12–11.32 s.

25 October 2026. The int8 record stores symmetric int8 codes and one float32 scale per output row. A printed 1280 generation of the smoke prompt from that record was 51.74 s, caption 0.15 s, decoder 9.27 s, steps 5.24–5.37 s. A 512 generation printed 4.43 s, with each step at 0.44 s. The int8 matmul medians were not retimed, so the 95.45 s kernel sum stays the fp4 `attend_flash` sum. Queries the same day: "Z-Image-Turbo 51.74 seconds 1280 int8 generation", "Z-Image-Turbo 1280x1280 8 steps seconds Metal int8 M5", "Z-Image-Turbo 1280 seconds per step inference", and "Z-Image 1280 51.74 OR 5.24s OR 5.37s Turbo". The search did not find the 51.74 s print. [zimgturbo](https://pypi.org/project/zimgturbo/) reports 13.7 s for a 1024 image at 8 steps on an M5 Pro. A Draw Things report gives 27 s for a 1920×1080 image and 11 s for 1024×768 on an M5 Pro Max ([drawthingsapp](https://www.reddit.com/r/drawthingsapp/comments/1tf07qa/m5_maxed_out_version_performance/)). An RTX 5090 bf16 table reports 3.18 s at 1024 and 9.99 s at 1536 and does not list 1280 ([witcheer](https://huggingface.co/datasets/witcheer/rtx-5090-benchmarks/blob/main/reports/z-image-turbo.md)). Those figures are other engines, other quants, and another edge.
