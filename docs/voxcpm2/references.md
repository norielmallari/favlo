# References

Papers and specifications for the VoxCPM2 pipeline. The serving note is [README.md](README.md).

Zhou, Yixuan, and the VoxCPM team. *VoxCPM2: Tokenizer-Free TTS for Multilingual Speech Generation, Creative Voice Design, and True-to-Life Cloning.* [arXiv:2606.06928](https://arxiv.org/html/2606.06928).

Zhou, Yixuan, Guoyang Zeng, Xin Liu, Xiang Li, Renjie Yu, Ziyang Wang, Runchuan Ye, Weiyue Sun, Jiancheng Gui, Kehan Li, Zhiyong Wu, and Zhiyuan Liu. *VoxCPM: Tokenizer-Free TTS for Context-Aware Speech Generation and True-to-Life Voice Cloning.* [arXiv:2509.24650](https://arxiv.org/abs/2509.24650).

MiniCPM Team. *MiniCPM4: Ultra-Efficient LLMs on End Devices.* [arXiv:2506.07900](https://arxiv.org/abs/2506.07900).

Mentzer, Fabian, David Minnen, Eirikur Agustsson, and Michael Tschannen. *Finite Scalar Quantization: VQ-VAE Made Simple.* ICLR 2024.

Su, Jianlin, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. *RoFormer: Enhanced Transformer with Rotary Position Embedding.* Neurocomputing, 2024.

Kazemnejad, Amirhossein, Inkit Padhi, Karthikeyan Natesan Ramamurthy, Payel Das, and Siva Reddy. *The Impact of Positional Encoding on Length Generalization in Transformers.* NeurIPS 2023.

OpenBMB. [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2). Apache-2.0. The checkout `config.json` is the width table.

Apple. *Explore Metal 4 games.* [WWDC26 session 330](https://developer.apple.com/videos/play/wwdc2026/330/). `matmul2d` and the tensor types. The language-model plane is `half` in and float out. The AudioVAE convolutions are float32.
