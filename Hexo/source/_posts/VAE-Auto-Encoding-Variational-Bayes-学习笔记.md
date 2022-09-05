---
title: VAE Auto-Encoding Variational Bayes 学习笔记
author: boss
tags:
  - VAE
  - 深度学习
  - 笔记
categories:
  - 实验室
abbrlink: 2abd261b
date: 2022-07-11 13:35:00
---

# VAE 学习笔记

李宏毅老师的 [课程视频](https://www.bilibili.com/video/av15889450/?p=33) 讲的很详细，记录一下

## 直观理解

VAE 的大体结构和 Auto-Encoder 类似
区别在于 VAE 在生成的过程中加入了噪音, 使模型具有了生成能力

<div align=center><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/noise.jpg"></div>


## VAE 的构造
从下图的结构可以看出，VAE 的 Encoder 会求出一个隐藏层的分布，然后对这个分布进行采样，生成一个 X', 理想的情况下，这个 X' 和 X 的距离应该是很小的

<div align=center><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/model.jpg"></div>

但是问题在于，如果我们优化函数只优化最终结果的距离，那么X'最准确的情况的下分布的方差会变为 0, 此时的 VAE 事实上就退化成了 auto-encoder。 因此我们需要在 Loss 中加入对分布的限制, 从而达到较好的生成效果。

## 公式推导

我们的目标是求解

<div align=center><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/maximum.png"></div>


这个式子可以进行变换:

<div align=center><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/trans.png"></div>

于是我们得到了一个 logP(x)的下界:
<div align=center><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Lb.png"></div>

此时原式化为: 
<div align=center><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/yuanshi.png"></div>

这里要补充一点，当我们在优化 Lb 的时候, KL散度是像0趋近的，这样我们logP(x)就会接近优化的下界。

注意到 

<div align=center><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/lbtuidao.png"></div>


因此只需要构造两个求解器，分别是 Encoder 和 Decoder

其中 Encoder 求解的是 q(z|x) 的分布，我们希望所有的 q(z|x) 都能近似 p(z) 的分布。z 的分布在 VAE 中定义成 N(0, 1)

<div align=center><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Encoder.jpg"></div>

Decoder 求解的是 p(x|z) 的分布

<div align=center><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Decoder.jpg"></div>


至此证明完毕




