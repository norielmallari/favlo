# Memory usage

Device bytes for `qwen38-27b` at the served window of 131,072 tokens. A GB is 10^9 bytes. The table is the opening with MTP on: the spans `startup_bytes` claims for the installed `Qwen3.8-27B-fp8-e20` record, with the MTP head resident, all 64 language feed-forward layers wired, and the default `half` key/value cache. The vision tower stays in the record file until a message contains an image. An opening with MTP off does not copy the MTP block or the MTP plane.

The weight image is 16.750 GB, including the half MTP block. The key/value cache is 17 planes, one for each language full-attention layer and one for the MTP layer, 69,632 bytes per token, 9.127 GB. The half attend keeps its state in registers, so that mode claims no attention scratch. The prefill workspace, the DeltaNet state, the convolution history, and the MTP verify snap stay for the life of the opening.

| Allocation | GB |
|---|---:|
| Weight image | 16.750 |
| Key/value cache | 9.127 |
| Prefill workspace | 0.860 |
| DeltaNet state | 0.151 |
| Convolution history | 0.003 |
| MTP verify snap | 0.154 |
| MTP span logits | 0.003 |
| Decode buffers and the draft hold | 0.002 |
| Steady total | 27.050 |

A verify window keeps that image and adds the three-token span, 0.011 GB. The snap and the span logits are already in the steady total. The high point of a request already at this length is 27.061 GB.

The step that grows from 129,024 to 131,072 allocates the new cache before it releases the old planes. The host still holds the packed copy of the live cache. That overlap is 45.304 GB. Once the old planes and the host copy are released, the opening returns to the steady total.

An opening with MTP off keeps the 16 language planes and leaves the MTP block in the record file. The decode buffers on either opening include the grouped top-k lists, 248,336 bytes. The cache is then 65,536 bytes per token. A fit that cannot keep the head drops it the same way, before any language feed-forward layer.

An opening that sets `kv_cache` to `fp8_e4m3` keeps the quantized planes: 35,904 bytes per token with MTP (4.706 GB at this length) plus 0.288 GB of attention scratch, for a steady total of 22.917 GB with MTP on.
