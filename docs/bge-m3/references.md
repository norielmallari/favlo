# References

Chen, Xiao, Zhang, Luo, Lian, and Liu. *BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation.* 2024. [arXiv:2402.03216](https://arxiv.org/abs/2402.03216). Section 3.2: the dense vector is the normalized CLS state, the lexical weight is ReLU of a 1024-to-1 map with a max over repeated token ids, and the multi-vector is a normalized 1024 projection.

Beijing Academy of Artificial Intelligence. *BAAI/bge-m3.* Model card. [Hugging Face](https://huggingface.co/BAAI/bge-m3). XLM-RoBERTa, 24 layers, hidden 1024, 16 heads, intermediate 4096, GELU, absolute positions, vocabulary 250002, sequence 8192. This pipeline serves the longest function in the packed record. Quantize builds functions through 4096.

Apple. *Quantization performance.* [Core ML Tools guide](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-perf.html). W8A8 is symmetric int8 weights and per-tensor int8 activations. On the Neural Engine the weight scale is per channel.

Apple. *Neural Engine rules.* [apple/coreai-models](https://github.com/apple/coreai-models/blob/main/skills/skills/model-authoring/references/neural_engine_rules.md). The pad fill is -40000. Attention is one head at a time. The embedding table is a gather.
