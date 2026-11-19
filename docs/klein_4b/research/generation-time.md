# Generation time

Measurement note for the `klein_4b` pipeline. The serving contract is [../ARCHITECTURE.md](../ARCHITECTURE.md). This file records two printed generations of the int4 record. It is a measurement of this implementation.

## Remainder

On a GeForce RTX 3070 Laptop GPU with 8192 MiB, one 1024 image at 4 steps printed 22.25 s. One 256 image at 4 steps printed 6.32 s. A search on 19 November 2026 did not find those prints.

## Predecessors

- Method. The [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) card is Apache-2.0. Its example is 4 steps, guidance 1.0, and 1024×1024. The card says the checkpoint fits in about 13 GB.
- Task. [ComfyUI](https://docs.comfy.org/tutorials/flux/flux-2-klein) reports the distilled 4B variant at about 1.2 s on an RTX 5090, using 8.4 GB. [VRAM Ready](https://vramready.com/benchmarks/) reports 2.85 s for the FP8 file, 3.93 s for the bf16 file, and 4.34 s for Q8_0, each a 1024 image at 4 steps on an RTX 5060 Ti 16 GB. [Home AI Bench](https://homeaibench.com/notes/2026-10-24-flux2-klein-4b/) reports a server-side median of 3.3 s on an RTX 5070 12 GB and 3.1 s on an RTX 4070 12 GB, at 4 steps and cfg 1. [TitanXT](https://titanxt.io/gpu-fit/rtx-3070-8gb-for-flux-2-klein-4b/) reports an RTX A2000 12 GB reference of about 8.0 s at 4 steps and 1024, and says a GeForce RTX 3070 8 GB was not timed.
- Measurement. Those figures are other engines, other weight formats, or other cards. They do not publish these two prints.
- Mechanism. Weights are symmetric int4 and the activations stay fp16. The text tower, the VAE encoder, the DiT, and the VAE decoder are not resident together. The caption clock includes the text upload, and the first step includes the DiT upload.

## Rule

Each printed line is one generation of the int4 record. The seed is 0. The step count is 4. The 1024 prompt is the smoke prompt. The 256 prompt is `a red square on a white field`. The 1024 total is the outer clock. It is not the sum of the stage lines. The 256 print has no stage split.

## Command

```text
uv run favlo media --pipeline klein_4b prompt="A tan paper horse mid-gallop on a white field, a soft gray shadow under the hooves, even studio light from the upper left." size=1024x1024 steps=4 seed=0
uv run favlo media --pipeline klein_4b prompt="a red square on a white field" size=256x256 steps=4 seed=0
```

The table is the pair printed on 19 November 2026 by an in-process generate after open, which is gone. The commands above are the same body through `favlo media`. A later run is a new measurement.

## Table

| Edge | Total | Caption | Steps | Decode |
|---|---|---|---|---|
| 256 | 6.32 s |  |  |  |
| 1024 | 22.25 s | 2.19 s | 4.93 s, 3.83 s, 3.85 s, 3.98 s | 3.45 s |

## Search

19 November 2026. Queries: `FLUX.2-klein-4B inference seconds 1024 4 steps int4 OR FP8 RTX 2026` and `FLUX.2 klein 4B "22.25" OR "GeForce RTX 3070 Laptop" int4 1024 4 steps ComfyUI`. The predecessors in this note are the overlaps those queries returned. The search did not find the 22.25 s print or the 6.32 s print.

## Corrections

Later finds are appended here. The sections above stay as committed.
