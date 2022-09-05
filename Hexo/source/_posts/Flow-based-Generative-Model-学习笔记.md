---
title: Flow-based Generative Model 学习笔记
author: boss
abbrlink: 77c57936
date: 2022-08-01 13:56:35
tags:
 - Flow
 - 深度学习
catorgories: 
 - 实验室
---

# Flow-based Generative Model
[李宏毅老师视频地址](https://www.bilibili.com/video/av46561029/?p=59)

参考了以下博客:

[【学习笔记】生成模型——流模型（Flow）](http://www.gwylab.com/note-flow_based_model.html)

[Flow-based Deep Generative Models](https://lilianweng.github.io/posts/2018-10-13-flow-models/)


## 生成模型

生成模型的统一目标都是对一个标准分布生成一个复杂分布, 使这个复杂分布和真实的分布尽量接近。

<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Flow Base Generative Model/generative.png" width="80%", height="80%"></div>

Flow 模型可以直接优化这个目标式子

## 数学补充

### Jacobian Matrix

有两个向量&ensp; `$ x, z $`

<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Flow Base Generative Model/Jacoxz.png" width="40%", height="40%"></div>


向量&ensp; `$ z $`&ensp; 经过线性变换&ensp; `$ f $`&ensp; 得到&ensp; `$ x $`&ensp;

<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Flow Base Generative Model/Jacoxt.png" width="20%", height="20%"></div>

因此&ensp;`$ f $`&ensp;的 Jacobian Matrix 为:
<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Flow Base Generative Model/Jacof.png" width="40%", height="40%"></div>

举个例子:
<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Flow Base Generative Model/jacosample1.png" width="30%", height="30%"></div>
<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Flow Base Generative Model/jacosample2.png" width="20%", height="20%"></div>

同理有&ensp;`$ z = f^{-1}(x) $`&ensp;, 此时&ensp;`$ f^{-1} $` &ensp;的 Jacobian Matrix 为:
<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Flow Base Generative Model/jacozt.png" width="40%", height="40%"></div>

有一个性质:
<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Flow Base Generative Model/jacoxingzhi.png" width="20%", height="20%"></div>

### Determinant

行列式的计算方法
2 x 2:
<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Flow Base Generative Model/det22.png" width="40%", height="40%"></div>

3 x 3:
<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Flow Base Generative Model/det33.png" width="40%", height="40%"></div>

行列式的性质:
<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Flow Base Generative Model/detxingzhi.png" width="40%", height="40%"></div>

&nbsp;
行列式的几何意义其实就是二维空间中的面积或三维空间中的体积:
<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Flow Base Generative Model/detxingzhi2.png" width="40%", height="40%"></div>

### Change Of Variable Theorem(变数变换)

首先一个基本问题就是, 对于一个简单的分布&ensp;`$ \pi(z) $`&ensp;和&ensp;`$ p(x) $`&ensp;, 我们希望找到他们之间的变换函数&ensp;`$ f $`
<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Flow Base Generative Model/relation.png" width="80%", height="80%"></div>

我们可以看一个简单的例子:

&ensp;`$ \pi(z) $`&ensp;是一个 `$ [0, 1] $` 上的均匀分布, 而&ensp;`$ p(x) $`&ensp;是一个 `$[1, 3]$` 上的均匀分布.

<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Flow Base Generative Model/z.png"></div>
<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Flow Base Generative Model/p.png"></div>

&ensp;`$ z $`&ensp;与&ensp; `$ x $`&ensp;之间的变换函数其实就是: `$ x = f(x) = 2z + 1 $`
<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Flow Base Generative Model/tran.png"></div>

注意到&ensp;`$ x $`&ensp;的底变宽了，为原来的 2 倍, 因此 x 的高其实是原来的 1/2。

现在我们的分布变成了一个复杂分布:
<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Flow Base Generative Model/complex.png" width="60%", height="60%"></div>

对于一段极小的区间, `$ [z^{'}, z^{'} + \Delta z], [x^{'}, x^{'} + \Delta x] $`&ensp;, 我们可以认为他们在这个区间里是均匀分布的。
<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Flow Base Generative Model/junyun.png" width="80%" height="80%"></div>

而蓝色的方块和绿色的方块面积要相同，因此我们得到式子:
<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Flow Base Generative Model/shizi.png" width="40%", height="40%"></div>

注意最后的式子需要绝对值, 因为&ensp;`$ x $`&ensp;的变化可能是正, 也可能是负的。

<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Flow Base Generative Model/bianhua.png" width="60%", height="60%"></div>

下面是二维的例子:

将一个&ensp;`$ z $`&ensp;上小小的正方形投影到&ensp;`$ x $`&ensp;上变成一个小小的菱形
<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Flow Base Generative Model/erwei.png"></div>

菱形的面积可以用行列式计算, 因此有:
<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Flow Base Generative Model/mianji.png" width="60%", height="60%"></div>

上面的式子可以经过一系列的变换:
<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Flow Base Generative Model/tuidao.png" width="60%", height="60%"></div>

最终得到:
<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Flow Base Generative Model/jieguo.png" width="40%", height="40%"></div>

## Flow-based Model
看看最初的目标:
<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Flow Base Generative Model/generative.png"></div>

我们可以用&ensp;`$ z $`&ensp;的分布写出&ensp;`$ x $`&ensp;的分布了:
<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Flow Base Generative Model/zx.png" width="60%", height="60%"></div>

要最大化这个式子有两个前提, 其一是&ensp;`$ det(J_{G}) $`&ensp;需要能够计算, 因为矩阵维数很大的话行列式的计算量很大, 其二是&ensp;`$ G^{-1} $`&ensp; 需要知道, 这就意味着&ensp;`$ G^{-1} $`&ensp;是可逆的, 因此&ensp;`$ z $`&ensp;和&ensp;`$ x $`&ensp;的维度也要是相同的。
从上面的分析可以发现, &ensp;`$ G $`&ensp;的能力是很有限的, 因此我们可以通过流的方式来构建一个更加复杂的分布。
<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Flow Base Generative Model/flow.png"></div>

对应的式子就是:
<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Flow Base Generative Model/liu.png" width="70%", height="70%"></div>

### 实际训练

在实际的训练中, 我们是对&ensp;`$ G^{-1} $`&ensp;进行训练, 训练好后再用&ensp;`$ G $`&ensp;来生成。
<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Flow Base Generative Model/actual.png"></div>

对应的最大化函数是:
<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Flow Base Generative Model/zuidahua.png" width="60%", height="60%"></div>

注意到对于第一项, 因为&ensp;`$ z ~ \pi(z) $`&ensp;, 因此&ensp;`$ z^{i} $` 为 0 向量时最大。但是此时第二项会趋于 `$ -inf $`。因此最大化是两项的均衡结果。

## 几种 Flow-based Model

### NICE/ Real NVP

[NICE: Non-linear Independent Components Estimation](https://arxiv.org/abs/1410.8516)
[Density estimation using Real NVP](https://arxiv.org/abs/1605.08803)

这里采用了耦合层的设计(Coupling Layer)

#### Coupling Layer

将&ensp;`$z $`&ensp;拆成两组, &ensp;`$ z_{1:d} $`&ensp;一组, &ensp;`$ z_{d+1:D} $`&ensp;一组。
&ensp;`$ z_{1:d} $`&ensp;直接复制变成&ensp;`$ x_{1:d} $`&ensp;
&ensp;`$ z_{1:d} $`&ensp;经过一个函数&ensp;`$ F $`&ensp;变成&ensp;`$ \beta_{d+1:D} $`&ensp;, 经过一个函数&ensp;`$ H $`&ensp;变成&ensp;`$ \gamma_{d+1:D} $`&ensp;
然后有&ensp;`$ x_{i>d} = \beta_{i}z_{i} + \gamma_{i} $`&ensp;
这里的&ensp;`$ F $`, `$ H $`&ensp;可以是任意的函数, 因为之后会发现&ensp;`$ G^{-1} $`&ensp;和他们无关。

<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Flow Base Generative Model/couple.png" width="80%", height="80%"></div>

因此, &ensp;`$z $`&ensp;变换成 &ensp;`$x $`&ensp;的式子可以写为:

<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Flow Base Generative Model/z2x.png"></div>

逆变换的式子则有:

<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Flow Base Generative Model/x2z.png"></div>

#### Jacobian 行列式
Flow 的另一个关键在于 &ensp;`$ G $`&ensp;的 &ensp;`$Det(J) $`&ensp;需要能够快速计算
事实上, &ensp;`$ J_{G} $`&ensp; 就是下面的形式:
<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Flow Base Generative Model/JG.png" width="70%", height="70%"></div>

左上角是直接复制过来的, 因此是单位矩阵。很明显右上角是 0, 因此计算行列式就与左下角无关了。 而右下角可以发现  &ensp;`$ x_{i>d} $`&ensp; 只和  &ensp;`$ z_{i} $`&ensp; 有关, 因此是对角矩阵。
所以有
<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Flow Base Generative Model/JGshizi.png" width="50%", height="50%"></div>

#### Stacking
因为我们的 Flow 是多层堆叠的, 所以我们的 Coupling layer 也要堆叠在一起。
<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Flow Base Generative Model/simple stack.png"></div>
可以发现简单的堆叠前半部分就没有改变了,因此一种思想是我们的堆叠可以是交叉的。第一层复制前半部分,第二层复制后半部分。
<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Flow Base Generative Model/complexstack.png"></div>
更具体的来说, 对于一个图片, 我们可以分为奇偶两种格子, 或者几个 channel 交叉。
<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Flow Base Generative Model/image.png" width="80%", height="80%"></div>

### GLOW

[Glow: Generative Flow with Invertible 1x1 Convolutions](https://arxiv.org/abs/1807.03039)

#### 1x1 Convolution
GLOW 引入了 W 矩阵来决定按什么顺序进行 copy 和 affine 变换，也就是 1x1 convolution。
<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Flow Base Generative Model/wmatrix.png" width="80%", height="80%"></div>

这个 W 矩阵能够对通道进行交换。比如我们将 (1, 2, 3) 的向量替换成 (3, 1, 2) 的向量，只需将 W 矩阵定义如下。
<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Flow Base Generative Model/jiaohuan.png" width="60%", height="60%"></div>

下面就是 Glow 中一步 flow 的模型
<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Flow Base Generative Model/one-glow-step.png" width="80%", height="80%"></div>

#### W 是否可逆?
论文中并没有论述。文中的操作是初始化时用一个可逆矩阵, 之后矩阵很大概率也是可逆的。

#### Jacobian Determinant

&ensp;`$z$`&ensp; 到 &ensp;`$x$`&ensp; 的变换为:
<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Flow Base Generative Model/wz.png" width="20%", height="20%"></div>

因此&ensp;`$ f $`&ensp;的 jacobian 其实就是&ensp;`$ W $`&ensp;
<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Flow Base Generative Model/real.png" width="40%", height="40%"></div>
<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Flow Base Generative Model/jf.png" width="60%", height="60%"></div>

代入到含有 d 个 3 * 3 维的仿射变换矩阵当中, 得到最终的Jacobian行列式的计算结果就为&ensp;`$ (det(W))^{dxd} $`&ensp;, 如下图所示:
<div align="center"><img src="https://cdn.jsdelivr.net/gh/dontnet-wuenze/picbed/Flow Base Generative Model/r11.png" width="80%", height="80%"></div>

W 是 3x3 的矩阵, 因此计算就非常简单了。