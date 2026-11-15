# Z-Image shared forward

`pipelines/z_image/metal41/z_image/` is the forward shared by Z-Image-Turbo and Z-Image.
The directory has no entry file and no `PIPELINE` assignment.
`favlo` does not list it as a CLI flag.

- Turbo and the base checkpoint each have their own pipeline folder and registry row.
- Sampler defaults and the fields those rows expose are on [the Turbo note](../z_image_turbo/ARCHITECTURE.md#registry) and [the base note](../z_image_base/ARCHITECTURE.md#registry).

For pack layout, Metal shaders, encoders, decoders, and how callers wire the shared forward, see [ARCHITECTURE.md](ARCHITECTURE.md).