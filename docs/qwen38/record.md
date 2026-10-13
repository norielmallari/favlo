# Qwen3.8 record

As-built files for the Qwen3.8-Flash-Next Metal record. Scope and the tier rules are in [ARCHITECTURE.md](../../ARCHITECTURE.md). If this note disagrees with that file, ARCHITECTURE.md wins.

Record version 4. The file is the buffer image Metal reads. `resident.bin` is one sequential load at startup, and its length is a multiple of 1 MiB. `ngram.bin` is random 16 KB pages. Each expert layer is `experts/NN-hot.bin` and `experts/NN-cold.bin`. A layer that has a super-expert pin also has `experts/NN-super.bin`. MTP experts stay one int4 file, `mtp-experts.bin`. Inside a live image, the data plane comes first and the E8M0 scale plane follows. Each plane starts on a 128-byte boundary. One Metal IO load covers one expert, both projections, both planes, and stops at the end of that image.

That image is what `MTLTensor` documents for a buffer-backed multi-plane tensor, and what WWDC26 session 330 passes to `matmul2d`. The innermost stride is 1 element. For int4, int8, int2, and E8M0, every later stride is a multiple of 128 bytes. The scale plane is `MTLTensorDataTypeMetalFloat8UE8M0` with block factors `{32, 1}`: one byte covers 32 values along K and the other dimension is not blocked. TensorOps slices both planes together and dequantizes in the matmul. A device-allocated tensor is the opaque layout Apple names as the fastest resident copy. The file stays the buffer image. After a weight is resident (the output head, and later a Super Expert), the runtime may upload that image into a device tensor once. An expert that is not already in the atlas is that load, whichever tier it belongs to. The decode shader reads the data plane and the scale plane at their offsets in the buffer.

int4 rows are stored at a K rounded up to a multiple of 256, so the row is a whole number of 128-byte groups. int8 rounds K up to a multiple of 128. int2 rounds K up to a multiple of 512. The extra K columns are zero codes. The scale row is `K/32` bytes, packed tight. `tensor_inline` takes that stride from the data extent and the block of 32, so a padded scale stride would make `matmul2d` read the wrong byte. The data plane length is a multiple of 128, and the scale plane starts on that boundary. The reconstruct is `code × 2^(e − 127)`. Exponents stay in `0..254`. There is no zero point.

A hot expert is `gate_up` at K 2560 and `down` at stored K 768 (logical 640), packed int4. The live image is 2,785,280 bytes, which is 170 pages of 16 KB, and that image is the on-disk stride. Row `r` in the hot file starts at `r × live`. The row is the index of that expert in the layer's hot list, which the manifest stores. A cold expert is the same projections packed int2. Gate K 2560 is already a multiple of 512. Down logical K 640 is stored at 1024. The live image is 1,658,880 bytes. The on-disk stride rounds that up to the next 16 KB page, 1,671,168 bytes, and the tail is zeros. A run is consecutive rows in one file. The two files are preallocated before their images are written, so the logical range is real SSD extents rather than a hole. A wide int8 image is 5,068,800 bytes. Gate K 2560 and down K 640 are already multiples of 128, so the image has no extra K columns. The on-disk stride rounds that live image up to the next 16 KB page, 5,079,040 bytes. Those five images are loaded once into device buffers and are not staged through the atlas. A live image that is short of a 16 KB page is read through the next page boundary. The same expert id is the same tier in both projections. `num_experts` stays 512. The router id maps to `(tier, row)`. Hot, cold, and super images are separate dispatches, because the down stored K and the byte stride differ.

| Tier | Data plane | One expert | Who |
|---|---|---|---|
| Wide | `metal_int8` | 4.83 MB | Super Experts. Resident |
| Middle | `metal_int4` | 2.66 MB | Domain pin and the REAP fill |
| Narrow | `metal_int2` | 1.58 MB | The complement, including experts that never fired |

The keep set is 384 ids per layer. Five of those ids are the int8 pin: layer 2 expert 322, layer 4 experts 108 and 373, layer 6 expert 223, and layer 10 expert 107. Those layers therefore have 383 or 382 hot int4 experts. The other 44 layers have 384 hot int4 experts. Every layer has 128 cold int2 experts, the complement of the original keep set.

A record directory starts with the Hugging Face model name, then `-madmetal-`, then each tier that contains experts. Counts are experts per layer. `s` is the super-expert pin and its plane, `h` is the hot keep set, `c` is the cold complement. The planes are `int8`, `int4`, and `int2`. A tier with zero experts is omitted. The default directory is `models/Qwen3.8-Flash-Next-madmetal-s5-int8-h384-int4-c128-int2`. The `h384` count is the keep-set width. The manifest holds each layer's hot ids and super ids, and those lists are the per-layer counts.

| Weight | Record |
|---|---|
| Output head | Middle int4. Loaded once into a device tensor |
| Token embedding | Middle int4. Loaded once. A token gathers one row from that copy |
| Gated residual, DeltaNet projections, sparse-attention projections, indexer, shared expert, n-gram `key_proj` and `value_proj` | Middle int4. Loaded once |
| Router, 512×2560 | 16-bit. Loaded once. It picks the tier |
| N-gram table | One int4 table. Each row occupies 128 bytes. The request path loads that slot |
| MTP experts | The same middle image, packed with no hole between experts |
| Norms, DeltaNet convolution, n-gram convolution | 16-bit. Loaded once |
| Rotary embedding | Computed, not stored |
| Running residual, four branches | 16-bit |

The n-gram table is not tiered. A row's payload is 160 int4 channels plus five E8M0 bytes, stored in a 128-byte slot so a 16 KB page holds 128 complete rows. The hash that selects a row is in [ngram.md](ngram.md).

The scale for a block of 32 is one E8M0 byte, bias 127. It is the power of two at or above `amax / qmax`, so the peak fits in the signed codes. `qmax` is 7 for int4, 127 for int8, and 1 for int2. The stored exponent is `log2(scale) + 127`, clamped to `0..254`. `0xFF` is not written. A zero block stores 0. Elements use round-ties-to-even and then saturate. The citations are [REFERENCES.md](../../REFERENCES.md).

int4 is packed low nibble first, signed −8..7, as measured in [probe.md](probe.md). int2 is packed low two bits first, signed −2..1. Metal 4.1 `unpack` does not accept `int2b` on this SDK, so that int2 layout is the one the shader shifts.

Expert loads use an uncached Metal IO handle. Uncached reads on this machine stay on the direct path when the file offset, the length, and the destination offset are multiples of the 16 KB VM page. The volume's preferred transfer is 1 MiB, so a sequential file is a multiple of that and a single expert image is larger than that. A single load of `resident.bin` returns `MTLIOStatusError` on this GPU, so that startup read is split into 256 MB ranges inside one IO command. `MTLIOStatusComplete` is 3. When the runtime reads these files is in [schedule.md](schedule.md).
