# Memory usage

Device bytes for `qwen38-27b` at the served window of 131,072 tokens. A GB is 10^9 bytes. The figures are the spans `startup_bytes` claims for the installed `Qwen3.8-27B-fp8-e20` record, with the MTP head resident and all 64 language feed-forward layers wired. The vision tower stays in the record file until a message contains an image.

The weight image is 16.339 GB, including the fp8 MTP block. The key/value cache is 17 planes, one for each language full-attention layer and one for the MTP layer, 35,904 bytes per token, 4.706 GB. The prefill workspace, the attention scratch, the DeltaNet state, and the convolution history stay for the life of the opening.

| Allocation | GB |
|---|---:|
| Weight image | 16.339 |
| Key/value cache | 4.706 |
| Prefill workspace | 0.860 |
| Attention scratch | 0.288 |
| DeltaNet state | 0.151 |
| Convolution history | 0.003 |
| Decode buffers and the draft hold | 0.001 |
| Steady total | 22.348 |

A verify window keeps that image and adds the three-token span, 0.011 GB, plus a host copy of the DeltaNet state and the convolution history, 0.154 GB. The high point of a request already at this length is 22.513 GB.

The step that grows from 129,024 to 131,072 allocates the new cache before it releases the old planes. The host still holds the packed copy of the live cache. That overlap is 31.615 GB. Once the old planes and the host copy are released, the opening returns to the steady total.

With the MTP head dropped and the 16 language planes kept, the same length is 21.550 GB. The cache is then 33,792 bytes per token.
