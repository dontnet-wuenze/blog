---
title: VAE-GAN 结合 论文梳理
author: boss
tags:
  - GAN
  - VAE
  - 深度学习
categories:
  - 实验室
abbrlink: 3dc06956
date: 2022-07-19 16:52:00
---

# VAE-GAN 结合 论文梳理

## 回顾

## 论文梳理

### VAE-GAN


[[2016][ICML]Autoencoding beyond pixels using a learned similarity metric-VAE-GANs](https://arxiv.org/abs/1512.09300)

这篇论文模型的结构就是连接了 VAE 和 GAN。

<div align="center" size="80%"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/VAE-GAN结合/VAE-GAN.png" width="80%", heigt="80%"></div>

&emsp;

#### 不同视角

- VAE视角 : 判别器具有 GAN 的性质, 生成图像更真实, 弥补 VAE 生成图像模糊的缺点。

- GAN视角: 需要额外计算 VAE 的重构 loss, 提升了模型的稳定性。

#### 训练过程

每个 iteration:
- 从数据中采样 M 个图片 &nbsp;`$ x^{1}, x^{2}, ..., x^{M}$`
- Encoder 生成 M 个隐变量 &nbsp;`$\tilde{z}^{1}, \tilde{z}^{2}, ..., \tilde{z}^{M}$`
    - &nbsp;`$\tilde{z}^{i} = En(x^{i})$`
- Decoder 生成 M 个图片 &nbsp;`$\tilde{x}^{1}, \tilde{x}^{2}, ..., \tilde{x}^{M}$`
    - &nbsp;`$\tilde{x}^{i} = De(\tilde{z}^{i})$`
- 从先验分布&nbsp; `$p(z)$`&nbsp; 采样 M 个隐变量 &nbsp;`$z^{1}, z^{2}, ..., z^{M}$`
- Decoder 生成 M 个图片 &nbsp;`$\hat{x}^{1}, \hat{x}^{2}, ..., \hat{x}^{M}$`
    - &nbsp;`$\hat{x}^{i} = De(z^{i})$`
- 更新 Encoder 来降低降低生成 &nbsp; `$ \tilde{x} $` &nbsp; 与 真实&nbsp; `$ x $` &nbsp;差距&nbsp;`$ \left|| \tilde{x} - x|\right| $`&nbsp;, 
    即降低&nbsp; `$KL(P(\tilde{z}|x)||p(z))$`  

- 更新 Decoder 来降低降低生成 &nbsp; `$ \tilde{x} $` &nbsp; 与 真实&nbsp; `$ x $` &nbsp;差距&nbsp;`$ \left|| \tilde{x} - x|\right| $`&nbsp;, 
    同时提高&nbsp; `$Dis(\tilde{x}) 和 Dis(\hat{x})$`    
- 更新判别器来提高&nbsp; `$Dis(x^{i})$`
    同时减小&nbsp; `$Dis(\tilde{x}^{i})$` 和 &nbsp; `$Dis(\hat{x}^{i})$`

### BiGANs

    
[[2017][ICLR] Adversarial Feature Learning-BiGANs](https://arxiv.org/abs/1605.09782)

[[2017][ICLR]Adversarially Learned Inference-BiGANs](https://arxiv.org/abs/1606.00704)

原始的 GAN 并不具备 Reconstruct 的能力, BiGAN 为 GAN 添加了 Encoder 的部分

<div align="center" size="80%"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/VAE-GAN结合/BiGAN.png" width="80%", heigt="80%"></div>

判别器的输入是原始图片和隐变量, 并判断这两个输入是来自 Encoder 还是 Decoder
理想情况下, BiGAN 就是一个 AutoEncoder, 此时 Encoder 和 Decoder 应该是互逆的, 即

<div align="center" size="80%"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/VAE-GAN结合/optimal.png" width="50%", heigt="50%"></div>

可是实践中并不能把这个模型训练到最优的情况, BiGAN 的重构效果也不是很好, 比如可能把一个鸟重构成另一只鸟。

### AAE

[[2016][ICLR]Adversarial Autoencoders-AAE](https://arxiv.org/abs/1511.05644)

AAE 使用一个对抗网络来替代 VAE 中后验分布的 KL 散度

使&nbsp; `$q(z|x) 和 p(x)$` &nbsp;不断靠近

<div align="center" size="80%"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/VAE-GAN结合/AAE.png" width="80%", heigt="80%"></div>

#### 优点
- 相比 VAE 预设了 N(0,1) 的高斯分布, P(z)可以是任意的分布。

#### 训练过程
- Reconstruction: 训练 Encoder 和 Decoder
- Regularization: 更新 D 和 Encoder(G)