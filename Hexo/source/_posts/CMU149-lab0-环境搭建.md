---
title: CMU149 lab0 环境搭建
author: boss
tags:
  - CMU 149
categories:
  - 课程学习
abbrlink: e401529b
date: 2022-07-10 23:33:00
---

# 环境搭建

附上官方地址:

<https://github.com/stanford-cs149/asst1>

## 安装 ISPC

首先需要安装 ISPC

我这里是 win11 使用 wsl 虚拟机 
如果虚拟机或者 linux 系统一样的操作


`wget https://github.com/ispc/ispc/releases/download/v1.16.1/ispc-v1.16.1-linux.tar.gz`

下载后解压

`tar -xvf ispc-v1.16.1-linux.tar.gz`

## 添加环境变量

`export PATH=$PATH:${HOME}/ispc-v1.16.1-linux/bin`

如果只在命令行里输入的话是临时的，下次就没有了
所以我们最好添加永久的环境变量

bash 用户可以添加到 ~/.bashrc 文件中


我这里在 zsh 添加了永久的环境变量

添加后输入

`source $HOME/.zshrc`

使环境变量生效

如果 lab1 的样例程序编译通过就代表安装成功了

[lab1 Parallel Fractal Generation Using Threads](http://bossalex.top/2022/07/10/lab1-Parallel-Fractal-Generation-Using-Threads/)