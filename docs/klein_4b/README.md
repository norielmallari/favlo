# FLUX.2 [klein] 4B

Text-to-image and image edit in one distilled rectified-flow model.

The official checkpoint is [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B).

Those weights are under the Apache License, Version 2.0.

Apple Silicon loads `pipelines/klein_4b/metal41/`. Windows loads `pipelines/klein_4b/cuda/`.

`favlo --klein_4b` loads this package for setup, quantize, and kernels.

Start `favlo serve`.

Then call `klein_4b__generate_image` or `klein_4b__edit_image` on the listener, or use `favlo media --pipeline klein_4b`.

The Apple Silicon registry model id is `FLUX.2-klein-4B-int8`. Windows registers `FLUX.2-klein-4B-int4`.

```text
uv run favlo --klein_4b setup
uv run favlo --klein_4b quantize
uv run favlo --klein_4b kernels
uv run favlo media --pipeline klein_4b prompt="A tan paper horse mid-gallop on a white field." size=1024x1024 seed=0
uv run favlo media --pipeline klein_4b prompt="Add a red saddle on the horse." image=file:///absolute/still.png size=576x320 seed=0
```

Kernels, the packed record, the forward, and the listener fields are in [ARCHITECTURE.md](ARCHITECTURE.md).

The model card's call is 4 steps and guidance 1.0. Guidance is baked in. An omitted size on a text-to-image call is 1024×1024. An omitted size on an edit is the first reference after it is fitted. `image` is one still or a list of one to four. The publisher quotes about 13 GB with the stages resident together. This package drops the text tower, the encoder, and the DiT after each stage finishes.

On Apple Silicon the packed record is symmetric int8. On Windows `quantize` writes `models/FLUX.2-klein-4B-int4`. That record is symmetric int4 weights with fp16 activations.

Measured kernel ceilings for one 576×320 edit are in [ceilings.md](ceilings.md). A 1024 generation on a GeForce RTX 3070 Laptop GPU printed 22.25 s. The print is in [research/generation-time.md](research/generation-time.md).
