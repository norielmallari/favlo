# References

Papers and specifications for the ZONOS2 pipeline. The model card is [Zyphra/ZONOS2](https://huggingface.co/Zyphra/ZONOS2).

- Clark, Zyphra team. ZONOS2. arXiv:2606.24320. Decoder, delay pattern, attention gate, speaker projection, and the sampling report.
- Washbourne, Zyphra team. ZAYA1. arXiv:2605.05365. Router: down-projection, EDA, RMSNorm, three-layer GELU MLP.
- Kumar, Seetharaman, Luebs, Kumar, and Kumar. High-Fidelity Audio Compression with Improved RVQGAN. arXiv:2306.06546. DAC hop 512 at 44.1 kHz. The decoder follows SoundStream.
- Zeghidour, Luebs, Omran, Skoglund, and Tagliasacchi. SoundStream: An End-to-End Neural Audio Codec. arXiv:2107.03312. Three residual units, dilations 1, 3, and 9, after each transposed convolution.
- Desplanques, Thienpondt, and Demuynck. ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification. Interspeech 2020.
- Hu et al. Qwen3-TTS. arXiv:2601.15621. The 2048-d speaker encoder used when it is absent from `model.pth`.
- Su, Lu, Pan, Murtadha, Wen, and Liu. RoFormer: Enhanced Transformer with Rotary Position Embedding. arXiv:2104.09864.
- Shazeer. GLU Variants Improve Transformer. arXiv:2002.05202. SwiGLU.
- Zhang and Sennrich. Root Mean Square Layer Normalization. arXiv:1910.07467.
- Open Compute Project. OCP Microscaling Formats (MX) Specification. E2M1, E4M3, and E8M0.
