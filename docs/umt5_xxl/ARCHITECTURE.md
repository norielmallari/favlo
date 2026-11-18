# umT5-XXL — architecture

This page is the as-built detail for the `umt5_xxl` pack library. [README.md](README.md) is the summary.

Shared text-encoder pack for Wan video pipelines. Not a Favlo CLI pipeline.

- The pack library lives under `pipelines/umt5_xxl/metal41/`.
- Wan I2V and TI2V attach `umt5 setup` and `umt5 quantize`.
- There is no `favlo --umt5_xxl` flag.
- Those pipelines open the pack through `parts.umt5` and read the same bytes.

```bash
uv run favlo --wan22_i2v umt5 setup
uv run favlo --wan22_i2v umt5 quantize --format wan
```

- `umt5 setup --umt5 org/name` downloads that repository instead.
- `umt5 quantize` defaults to `--format wan` (the official zip `.pth`).
- `--format hf` packs a Hugging Face `.safetensors` encoder; pass `--src` for that file.
- `--out` sets the pack directory; the default is `models/google/umt5-xxl-int8`.
- `--tokenizer` supplies the tokenizer directory when it is not beside `--src`.
- `--keep-source` leaves the BF16 checkout in place.
- The default pack removes that checkout after a successful write.

## Pack layout

- Every attention and feed-forward linear is symmetric int8.
- One float32 scale sits per output row.
- Output rows pad to 128. K pads to 256.
- Each linear is one `.npz` with codes, scales, bias, and extents.
- The token embedding (`embed.npy`), RMSNorm weights, and relative position buckets stay float16.
- `weights.bin` and `planes.json` are written beside those files. The contract is [below](#weightsbin).
- Tokenizer files are copied into the pack directory.
- `manifest.json` records weight kind and the worst relative RMS name and value.
- A clip never opens the BF16 file.

## weights.bin

`quantize` writes `weights.bin` and `planes.json` in the pack directory. The `.npy` files and the `.npz` files stay.

- `weights.bin` is raw array bytes. It has no zip header and no `.npy` header.
- `planes.json` is a JSON list in file order. That list is the only index.
- Each object has `name`, `dtype`, `offset`, `nbytes`, and `shape`, in that key order.
- `offset` is a multiple of 128. The first plane starts at 0.
- Bytes between planes are `0`. Those pad bytes are not included in `nbytes`.
- `nbytes` is the array size in bytes. `shape` is that array's shape.
- `dtype` is the NumPy dtype name: `float16`, `int8`, or `float32`.
- The embedding, layer norms, position tables, and the final norm are the `.npy` payloads.
- The position table is the float16 array in `blocks.{layer}.pos.npy`. It is not cast to float32.
- A linear contributes `packed` (int8), `scales` (float32), and `bias` (float16).
- `perm` is not copied. `kind` and the extent fields stay in the `.npz`.
- Tokenizer files and `manifest.json` are not in the blob.

Plane order:

- `embed`
- For layer 0, then 1, through 23:
  - `blocks.{layer}.norm1`
  - `blocks.{layer}.attn.q.weight.packed`, `.scales`, `.bias`
  - the same three names for `attn.k`, `attn.v`, and `attn.o`
  - `blocks.{layer}.pos`
  - `blocks.{layer}.norm2`
  - the same three names for `ffn.gate.0`, `ffn.fc1`, and `ffn.fc2`
- `norm`

`blocks.0.attn.q.weight.packed` is `packed` in `blocks.0.attn.q.weight.npz`. `blocks.0.pos` is the payload of `blocks.0.pos.npy`. `embed` is the payload of `embed.npy`. `norm` is the payload of `norm.npy`.

## Registry

This pack is not a registry row.

- The writable field is `parts.umt5` on the Wan row that opens the clip.
- The value is this int8 directory.
- The field is copied at open, so close the resident model after the write.
- The keys, the fallbacks, and the other fields on those rows are in [the I2V note](../wan22_i2v/ARCHITECTURE.md#registry) and [the TI2V note](../wan22_ti2v_5b/ARCHITECTURE.md#registry).
