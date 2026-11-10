# Z-Image shared forward

`pipelines/z_image/metal41/z_image/` is the forward shared by Z-Image-Turbo and Z-Image. The directory has no entry file and no `PIPELINE` assignment, so `favlo` does not list it.

The package holds the int8 and fp4 pack, the Metal shaders, the Qwen3 caption encoder, the S3-DiT, the Flux VAE decoder, the flow-matching knots, and the resumable file download. A caller passes the step count, the shift, the format id, and the source repository. Turbo calls it with eight steps and shift 3. The base checkpoint calls it with the requested step count and shift 6.

Each pipeline entry inserts `pipelines/z_image/metal41` on `sys.path` before it imports its own package.
