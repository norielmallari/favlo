# Qwen3.8 record

As-built files for the Qwen3.8-Flash-Next Metal record. Scope and the tier rules are in [ARCHITECTURE.md](../../ARCHITECTURE.md). If this note disagrees with that file, ARCHITECTURE.md wins.

Record version 5. The file is the buffer image Metal reads. `resident.bin` is one sequential load at startup, and its length is a multiple of 1 MiB. `ngram.bin` is random 16 KB pages. Each expert layer is `experts/NN-hot.bin` and `experts/NN-cold.bin`. A layer that has a super-expert pin also has `experts/NN-super.bin`. MTP experts stay one fp4 file, `mtp-experts.bin`. Inside a live image, the data plane comes first and the E8M0 scale plane follows. Each plane starts on a 128-byte boundary. One Metal IO load covers one expert, both projections, both planes, and stops at the end of that image.

That image is what `MTLTensor` documents for a buffer-backed multi-plane tensor, and what WWDC26 session 330 passes to `matmul2d`. The innermost stride is 1 element. For fp4, fp8, int2, and E8M0, every later stride is a multiple of 128 bytes. The scale plane is `MTLTensorDataTypeMetalFloat8UE8M0` with block factors `{32, 1}`: one byte covers 32 values along K and the other dimension is not blocked. TensorOps slices both planes together and dequantizes in the matmul. A device-allocated tensor is the opaque layout Apple names as the fastest resident copy. The file stays the buffer image. After a weight is resident (the output head, and later a Super Expert), the runtime may upload that image into a device tensor once. An expert that is not already in the atlas is that load, whichever tier it belongs to. The decode shader reads the data plane and the scale plane at their offsets in the buffer.

fp4 rows are stored at a K rounded up to a multiple of 256, so the row is a whole number of 128-byte groups. fp8 rounds K up to a multiple of 128. int2 rounds K up to a multiple of 512. The extra K columns are zero codes. The scale row is `K/32` bytes, packed tight. `tensor_inline` takes that stride from the data extent and the block of 32, so a padded scale stride would make `matmul2d` read the wrong byte. The data plane length is a multiple of 128, and the scale plane starts on that boundary. The reconstruct is `value × 2^(e − 127)`, where `value` is the fp4 E2M1 or fp8 E4M3 number. Exponents stay in `0..254`. There is no zero point.

A cold expert is `gate_up` at K 2560 and `down` at stored K 768 (logical 640), packed fp4. The live image is 2,785,280 bytes, which is 170 pages of 16 KB, and that image is the on-disk stride. Row `r` in the cold file starts at `r × live`. The row is the index of that expert in the complement of the hot list. A hot expert is the same projections packed fp8. Gate K 2560 and down K 640 are already multiples of 128, so the image has no extra K columns. The live image is 5,068,800 bytes. The on-disk stride rounds that up to the next 16 KB page, 5,079,040 bytes. Row `r` in the hot file starts at `r × stride`. The two files are preallocated before their images are written, so the logical range is real SSD extents rather than a hole. An int2 image, used when a record asks for width 2, stores gate K 2560 and down logical K 640 at 1024. That live image is 1,658,880 bytes and the on-disk stride is 1,671,168 bytes. A live image that is short of a 16 KB page is read through the next page boundary. The same expert id is the same tier in both projections. `num_experts` stays 512. The router id maps to `(tier, row)`. Hot, cold, and super images are separate dispatches, because the down stored K and the byte stride differ.

| Tier | Data plane | One expert | Who |
|---|---|---|---|
| Wide | `metal_fp8_e4m3` | 4.83 MB | Hot file and a super-expert pin. Resident on decode |
| Middle | `metal_fp4_e2m1` | 2.66 MB | Cold complement |
| Narrow | `metal_int2` | 1.58 MB | The complement, including experts that never fired |

The default record uses the wide fp8 image for 128 hot experts and the middle fp4 image for the other 384. A super-expert pin, when the manifest lists one, is the same fp8 image in `NN-super.bin` and is removed from the hot file.

A record directory starts with the Hugging Face model name, then `-madmetal-`, then each tier that contains experts. Counts are experts per layer. `s` is the super-expert pin and its plane, `h` is the hot file, `c` is the cold file. The planes are `fp8`, `fp4`, and `int2`. A tier with zero experts is omitted. `hot_bits` and `cold_bits` in the manifest are the data-plane widths of those two files. `hot_format` and `cold_format` name the Metal types, `fp8_e4m3` and `fp4_e2m1` on the default record. The default directory is `models/Qwen3.8-Flash-Next-madmetal-h128-fp8-c384-fp4`: 128 hot experts at fp8 and the other 384 at fp4, with no super file. The manifest holds each layer's hot ids and super ids, and those lists are the per-layer counts.

| Weight | Record |
|---|---|
| Output head | Middle fp4. Loaded once into a device tensor |
| Token embedding | Middle fp4. Loaded once. A token gathers one row from that copy |
| Gated residual, DeltaNet projections, sparse-attention projections, indexer, shared expert, n-gram `key_proj` and `value_proj` | Middle fp4. Loaded once |
| Router, 512×2560 | 16-bit. Loaded once. It picks the tier |
| N-gram table | One fp4 table. Each row occupies 128 bytes. The request path loads that slot |
| MTP experts | The same middle image, packed with no hole between experts |
| Norms, DeltaNet convolution, n-gram convolution | 16-bit. Loaded once |
| Rotary embedding | Computed, not stored |
| Running residual, four branches | 16-bit |

The n-gram table is not tiered. A row's payload is 160 fp4 channels plus five E8M0 bytes, stored in a 128-byte slot so a 16 KB page holds 128 complete rows. The hash that selects a row is in [ngram.md](ngram.md).

The scale for a block of 32 is one E8M0 byte, bias 127. It is the power of two at or above `amax / qmax`, so the peak fits in the format. `qmax` is 6 for fp4 E2M1, 448 for fp8 E4M3, and 1 for int2. The stored exponent is `log2(scale) + 127`, clamped to `0..254`. `0xFF` is not written. A zero block stores 0. fp4 picks the nearest of `0, 0.5, 1, 1.5, 2, 3, 4, 6` and stores the sign in bit 3 of the nibble, low nibble first. fp8 stores one OCP E4M3FN byte per value, ties to the even mantissa. The citations are [REFERENCES.md](../../REFERENCES.md).

int2 is packed low two bits first, signed −2..1. Metal 4.1 `unpack` does not accept `int2b` on this SDK, so that int2 layout is the one the shader shifts. The integer int4 probe, signed −8..7 and low nibble first, is in [probe.md](probe.md). That integer layout is not a weight plane.

Expert loads use an uncached Metal IO handle. Uncached reads on this machine stay on the direct path when the file offset, the length, and the destination offset are multiples of the 16 KB VM page. The volume's preferred transfer is 1 MiB, so a sequential file is a multiple of that and a single expert image is larger than that. A single load of `resident.bin` returns `MTLIOStatusError` on this GPU, so that startup read is split into 256 MB ranges inside one IO command. `MTLIOStatusComplete` is 3. When the runtime reads these files is in [schedule.md](schedule.md).
