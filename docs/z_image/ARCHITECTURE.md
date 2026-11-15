# Z-Image shared forward

This note is the as-built detail for this package. [README.md](README.md) is the summary.

`pipelines/z_image/metal41/z_image/` is the forward shared by Z-Image-Turbo and Z-Image.
The directory has no entry file and no `PIPELINE` assignment.
`favlo` does not list it.

The package holds:

- the int8 and fp4 pack
- the Metal shaders
- the Qwen3 caption encoder
- the S3-DiT
- the Flux VAE decoder
- the flow-matching knots
- the resumable file download

A caller passes:

- the step count
- the shift
- the format id
- the source repository

- Turbo calls it with eight steps and shift 3.
- The base checkpoint calls it with the requested step count and shift 6.

Each pipeline entry inserts `pipelines/z_image/metal41` on `sys.path` before it imports its own package.

## Registry

- This package has no registry row.
- Turbo and the base checkpoint each store their own folder.
- The fields those rows read are on [the Turbo note](../z_image_turbo/ARCHITECTURE.md#registry) and [the base note](../z_image_base/ARCHITECTURE.md#registry).
- Sampler defaults for the base checkpoint (`steps`, `guidance`, `shift`, `cfg_truncation`, `cfg_normalization`) live on that row.
- Turbo runs eight steps and has no sampler keys.