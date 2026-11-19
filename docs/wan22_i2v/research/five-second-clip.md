# Five-second clip

Measurement note for the `wan22_i2v` pipeline. The serving contract is [../ARCHITECTURE.md](../ARCHITECTURE.md). This file records one printed 81-frame clip. It is a measurement of this implementation.

## Remainder

On an M5 Pro with 48 GB, one 576×320 first-to-last clip of 81 frames printed 176.76 s of stage time. Encode was 23.46 s, the four steps were 29.35 s, 29.41 s, 29.16 s, and 29.10 s, and decode was 35.45 s. A search on 19 November 2026 did not find that print.

## Predecessors

- Method. Wan et al. ([arXiv:2503.20314](https://arxiv.org/abs/2503.20314)) define the dual-expert image-to-video transformer, the 81-frame length, and the Wan2.1 causal VAE. The [Wan-AI/Wan2.2-I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B) card serves that checkpoint. ComfyUI's [Wan 2.2 14B FLF2V](https://docs.comfy.org/tutorials/video/wan/wan2_2) template uses the same I2V experts for a first and last still.
- Task.
  - The [vLLM-Omni cookbook](https://github.com/hsliuustc0106/vllm-omni-cookbook/blob/main/diffusion/wan2.2/index.md) lists 26.0 s for 832×480, 81 frames, and 4 steps.
  - A Tenstorrent p100a steady run of a 480p 81-frame 4-step segment is 95.0 s, with encode 20.7 s, sampling 60.5 s, and decode 13.2 s ([sunbin.kim](https://sunbin.kim/wan-2-2-video-single-tenstorrent-p100a/)).
  - The [nalexand FP8 card](https://huggingface.co/nalexand/Wan2.2-I2V-A14B-FP8) prints 58.23 s/it, and 51.32 s/it for FP8, at 640×352 and 81 frames on a 3070 Ti.
  - An M1 Max ComfyUI run at 480×320 and 25 frames finished in 182.7 s, at 27.6 s per step ([Zenn](https://zenn.dev/umamon/articles/e24fbee8dc54d4)).
  - [PocketAI](https://modelfit.io/blog/wan-2-2-on-mac/) reports 10:25 end to end for a Q4 pack at 512×320, 81 frames, and 10 steps on an M5 Max with 128 GB.
- Measurement. Those figures are other engines, other machines, or another frame count or step count. They do not publish this 576×320 stage print.
- Mechanism. Guidance is 1, so each step is one forward. The same still is the first pixel and the last pixel. The text encoder and the two VAE loads sit outside the stage sum.

## Rule

The clip is 576×320, 81 frames, seed 0, and four Euler steps. The record is the int8 pack. The stage sum adds the printed text, encode, step, and decode lines. It does not add the VAE loads or the expert loads.

## Command

```text
uv run favlo media --pipeline wan22_i2v prompt="The horse gallops at speed, its raised foreleg driving down and the trailing legs swinging forward into the next stride, mane and tail whipping. Fixed camera." image=file:///absolute/still.png last_image=file:///absolute/still.png size=576x320 frames=81 seed=0
```

The table is the print from 19 November 2026. A later run is a new measurement.

## Table

| Stage | Time |
|---|---|
| umT5 | 0.83 s |
| VAE encode | 23.46 s |
| Step 0 | 29.35 s |
| Step 1 | 29.41 s |
| Step 2 | 29.16 s |
| Step 3 | 29.10 s |
| VAE decode | 35.45 s |
| Stage sum | 176.76 s |

## Search

19 November 2026. Queries: "Wan2.2 I2V 576x320 81 frames 176.92 seconds OR 29.35 seconds step Metal 2026" and "Wan2.2 I2V A14B 576x320 81 frames 4 steps 23.46 encode OR 29.35 step Apple Silicon". The predecessors in this note are the overlaps those queries returned. The search did not find the 176.76 s print.

## Corrections
