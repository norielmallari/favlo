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
