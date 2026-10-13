# Qwen3.8 n-gram table

As-built n-gram injection for Qwen3.8-Flash-Next. The slot layout is in [record.md](record.md). Scope is in [ARCHITECTURE.md](../../ARCHITECTURE.md). If this note disagrees with that file, ARCHITECTURE.md wins.

The n-gram table is injected at decoder layer 1 (`ple_layer_ids` is `[2]`, compared as `layer_idx + 1`), before that layer's residual read. The first eight heads hash the bigram ending at the current token. The next eight hash the trigram. Shift 0 is the current token and uses the first checkpoint multiplier; each older token uses the next multiplier. A lookback that starts before the segment, or that crosses EOS, contributes the EOS id. The mix is a wrapping int64 XOR of token × multiplier, then the positive remainder of that head's prime, plus the head's offset. Each row is 160 channels, and the sixteen rows concatenate to 2560. `value_proj` is shared. `key_proj` produces one key per branch. The branch score is `sign(s) √|s|` where `s = RMSNorm(R_m) · RMSNorm(K_m) / √d`, and the gate is σ of that score. The delta is the gated value plus SiLU of a depthwise causal convolution of its branch RMSNorm, kernel 4 and dilation 3, with nine frames of history. About 320 million rows.

The request path loads the 16 KB page that holds the selected 128-byte slot. The matmul that consumes a gathered row is `value_proj`, and that projection is one of the matrices loaded at startup. There is no per-row score.

The residual read normalizes each branch with a Gemma gain, `x / rms · (1 + w)`, then a low-rank sigmoid gate. The low-rank map is scaled by `1/n_r` before SiLU. The block input is the mean of the gated branches, the second `1/n_r`. The write is `2 σ(proj / n_r)` per branch. The final mixer is the read only.
