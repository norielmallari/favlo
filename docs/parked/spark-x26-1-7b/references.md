# References

The checkpoint was read on 1 November 2026 from [XHToken/Spark-X2.6-1.7B](https://huggingface.co/XHToken/Spark-X2.6-1.7B). The card says this release keeps the Spark-X2.5-1.7B architecture and changes post-training. Citation on the card: SparkLLM Team, *Spark-X2.6-4B & 1.7B: Advancing the Frontier of On-Device Agents for Real-World Tasks*, 2026.

- [XHToken/Spark-X2.5](https://github.com/XHToken/Spark-X2.5). The series note for the hybrid stack: one full-attention layer for every three sliding-window layers, native context 1,048,576.
- Zhang and Sennrich, *Root Mean Square Layer Normalization*. The publisher multiplies by the weight, with \(\varepsilon = 10^{-6}\).
- Su et al., *RoFormer*. The half-split frequencies and the two \(\theta\) values are the checkpoint fields `rope_parameters`.
- Hendrycks and Gimpel, *Gaussian Error Linear Units*. The erf form is the Transformers 4.57 `gelu` selected by `hidden_act`.
- Open Compute Project, *Microscaling Formats (MX) Specification*, and [WWDC26 session 330](https://developer.apple.com/videos/play/wwdc2026/330/). fp8 E4M3, E8M0, and `matmul2d` are the types this pipeline dispatches for weights. The key/value cache is float16 by default, and `kv_cache` `e4m3` keeps the packed E4M3 planes.
