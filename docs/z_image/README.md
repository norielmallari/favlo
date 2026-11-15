# Z-Image shared forward

Shared Metal forward for Z-Image-Turbo and Z-Image. The package lives at `pipelines/z_image/metal41/`. It has no typed `pipeline` export and no `PIPELINE` assignment. `favlo` does not list it as a CLI flag.

- Served pipelines import `z_image.metal41`.
- Those pipelines are `z_image_turbo` and `z_image_base`.
- Each owns download, pack, registry, and `favlo media`.
- Setup and registry fields are on [the Turbo note](../z_image_turbo/README.md) and [the base note](../z_image_base/README.md).

[ARCHITECTURE.md](ARCHITECTURE.md) is the as-built record: pack layout, shaders, encoders, decoders, schedule, and how callers wire this library.
