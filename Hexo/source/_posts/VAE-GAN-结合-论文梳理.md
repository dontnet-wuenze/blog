title: VAE-GAN 结合 论文梳理
author: boss
date: 2022-07-19 16:52:00
tags:
 - GAN
 - VAE
 - 深度学习
categories: 
 - 实验室
---

# VAE-GAN 结合 论文梳理

## 回顾

## 论文梳理

### VAE-GAN


[[2016][ICML]Autoencoding beyond pixels using a learned similarity metric-VAE-GANs](https://arxiv.org/abs/1512.09300)

这篇论文模型的结构就是连接了 VAE 和 GAN。

<div align="center" size="80%"><img src=".\VAE-GAN 结合\VAE-GAN.png" width="80%", heigt="80%"></div>

&emsp;

#### 不同视角

- VAE视角 : 判别器具有 GAN 的性质, 生成图像更真实, 弥补 VAE 生成图像模糊的缺点。

- GAN视角: 需要额外计算 VAE 的重构 loss, 提升了模型的稳定性。

#### 训练过程

<div>
- 训练 Encoder 来降低降低生成 $$\tilde{x}$$ 与 真实 $$x$$差距 $$\left|| \tilde{x} - x|\right| $$, 即降低 $KL(P(\tilde{z}|x)||p(z))$ 
</div>
