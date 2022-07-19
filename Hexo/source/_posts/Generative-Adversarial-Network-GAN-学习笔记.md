title: Generative Adversarial Network(GAN)学习笔记
author: boss
date: 2022-07-18 18:22:04
tags:
 - GAN
 - 深度学习
categories:
 - 实验室
---

# Generative Adversarial Network(GAN)学习笔记

重度参考李宏毅老师的课程:

https://www.youtube.com/watch?v=4OWp0wDu6Xw&list=PLJV_el3uVTsMhtt7_Y6sgTHGHp1Vb2P2J&index=14&ab_channel=Hung-yiLee

## 直观理解

从理解的角度来讲, GAN 由一个生成器和一个判别器组成。训练的过程就是生成器和判别器对抗的过程。判别器希望能分别出生成器生成的图片和真实的图片。而生成器则希望"以假乱真", 使最终的判别器无法区分真实的图片和生成的图片。理想的情况下，最终的判别器具有很高的辨认能力, 而生成器生成的图片则几乎无法与真实图片区分。

<div align="center" size="80%"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Generative Adversarial Network(GAN)/zhiguanlijie.png" width="80%", heigt="80%"></div>

&emsp;

比如一开始判别器会学习到真实图片都有眼睛, 这样生成器就会学习生成眼睛的能力, 之后判别器又会学习到真实的头像都有嘴巴, 生成器为了以假乱真也要学会生成嘴巴, 这就是一个对抗的过程, 最终使生成器具有生成能力。

## 训练过程

训练过程很清晰, 首先固定生成器的参数, 对判别器进行训练。 之后固定判别器的参数, 对生成器进行训练。这样迭代多轮后直到训练结束。

<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Generative Adversarial Network(GAN)/algorithm1.png" width="80%", heigt="80%"></div>

## 理论介绍

### 目标函数

我们现在有一个生成器 G, 我们最终的目标就是希望 G 生成的分布和真实 Data 的分布是相似的。

<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Generative Adversarial Network(GAN)/theoryimage.png" width="80%", heigt="80%"></div>

&emsp;

写成公式就是下面的式子:

<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Generative Adversarial Network(GAN)/theoryshizi.png" width="40%", heigt="70%"></div>

### 求解分布 Diversity

而问题在于, 我们有什么方法能求得这两个分布之间的 Diversity?

首先我们肯定没法直接去求分布的积分，因此可以采样。

一个比较直观的想法是用判别器去对采样的结果进行分类，如果分类准确率高就说明两者的分布相差大,好分辨。

<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Generative Adversarial Network(GAN)/zhiguanbianbie.png" width="80%", heigt="80%"></div>          
&emsp;

公式化的写法就是我们要训练一个判别器, 而 V(G,D) 推导出来事实上就是两个分布间的 JS 散度。我们想要判别器的效果好, 就是要最大化 V。

<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Generative Adversarial Network(GAN)/bianbieqi.png" width="80%", heigt="80%"></div>

&emsp;

从另一个角度看, V 其实是交叉熵的负数, 因此我们要最大化 V, 也就是减小判别器的 Loss。

因为 V 越大，代表判别的准确度越高, 因此回到上面的式子, 最小化 Diversity, 也就是最小化 V。

<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Generative Adversarial Network(GAN)/zuixiaohua.png" width="40%", heigt="80%"></div>

## GAN 的缺陷

### JS 散度的局限性

可以看到, 我们在上面的公式中使用了 JS 散度来比较两个分布的相似度, 但这会带来问题, 因为 JS 散度没法评估两个不重合分布的相似度。

### 为什么两个分布不重合

#### 1. 数据的性质
图片的分布在高维空间中可能是一条线, 或者只有很小一部分,这是因为大部分的空间中的点并不能组成一个完整的图像。因此两个分布只会有很少的部分相交, 可以忽略不记。

<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Generative Adversarial Network(GAN)/fenbuxiangjiao.png" width="45%", heigt="45%"></div>

#### 2. 采样
从分布中采样出的点是很少一部分, 即使整体分布有重合，采样的稀疏点也很容易被判别成两个不重合的分布。

<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Generative Adversarial Network(GAN)/sample.png" width="45%", heigt="45%"></div>

### JS 散度的问题

只要两个分布不重合, JS 散度恒为 log2, 这就导致判别器没法训练。

<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Generative Adversarial Network(GAN)/jsproblem.png" width="80%", heigt="80%"></div>

## Wasserstein distance(推土机距离)

为了解决这个问题, 我们改用 Wasserstein distance 来判断两个分布的相似度。这个距离的定义就是把一个分布变成另一个分布需要移动的最小平均距离。

可以看到, 此时分布间的距离就会越来越近了。

<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Generative Adversarial Network(GAN)/wdistance.png" width="80%", heigt="80%"></div>

## WGAN

### 距离公式
Wasserstein distance 也有对应的推导式子:
<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Generative Adversarial Network(GAN)/wshizi.png" width="60%", heigt="80%"></div>

&emsp;

可以看到, 这里对 D 有一个限制, 直白的说就是要求 D 是平滑的。

因为如果没有限制, D 就无法收敛了。(两个值会无限的增大和减小)

<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Generative Adversarial Network(GAN)/shoulian.png"width="60%", heigt="60%" ></div>

### Lipschitz Function
事实上 D 是要满足这个 Function 的性质, 也就是下面这个式子:

<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Generative Adversarial Network(GAN)/Lip.png"width="40%", heigt="40%" ></div>

而 D 就是 K = 1 的情况。


### 如何保证 D 是平滑的

有几种方法使 D 拥有平滑的性质:

* Original WGAN → Weight

    可以限制 w 的值在 [-c, c] 之间, 这样相当于限制了 D 的上下界。

* Improved WGAN → Gradient Penalty

    上面的 Lipschitz 性质有一个等价条件, 就是 D 对 x 的梯度都是小于等于 1 的

    <div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Generative Adversarial Network(GAN)/gp.png"width="40%", heigt="40%" ></div>

    很容易理解的方法就是对梯度加一个惩罚项。

    <div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Generative Adversarial Network(GAN)/jifen1.png" width="40%", heigt="40%" ></div>

    有一个问题在于我们没法对所有的X进行积分, 因此一个替代的方法是用 P<sub>data</sub> 和 P<sub>G</sub> 中间的 P<sub>penalty</sub> 的分布下的 X 作为惩罚项。
    
    <div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Generative Adversarial Network(GAN)/ppen.png" width="60%", heigt="60%" ></div>

    这样做看起来也确实是有一定道理的。 因为最终 P<sub>data</sub> 和 P<sub>G</sub> 逐渐靠近, 也只有他们中间的分布才会有影响。

    实验中有一个更好的收敛方法是梯度接近 1 是最好的。
    <div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Generative Adversarial Network(GAN)/improve.png" width="60%", heigt="60%" ></div>

    https://arxiv.org/abs/1704.00028

* Spectral Normalization → Keep gradient norm smaller than 1 everywhere

    这个方法可以保证每个梯度都是小于 1 的

    https://arxiv.org/abs/1802.05957

### WGAN algorithm

WGAN 的算法在原始 GAN 的只需要四步改进:

* 修改 D 的目标函数

* D 的训练添加 Weight clipping / Gradient Penalty

* 修改 G 的目标函数

* D 的输出不要 sigmoid 激活函数

* 优化器不要用 Adam, 用 SGD

<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Generative Adversarial Network(GAN)/wgan.png" width="90%", heigt="90%" ></div>

    