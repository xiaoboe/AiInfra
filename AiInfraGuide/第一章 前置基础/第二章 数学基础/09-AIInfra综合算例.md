# 9. AI Infra 综合算例

### 9.1 算例一：Transformer 线性层的 shape 与代价

输入：

$$
\mathbf{X}\in\mathbb{R}^{B\times S\times H}
$$

权重：

$$
\mathbf{W}\in\mathbb{R}^{H\times 4H}
$$

把前两维展平：

$$
\mathbf{X}'\in\mathbb{R}^{(BS)\times H}
$$

输出：

$$
\mathbf{Y}'=\mathbf{X}'\mathbf{W}
\in\mathbb{R}^{(BS)\times 4H}
$$

再 reshape 为 $(B,S,4H)$。FLOPs：

$$
2\cdot(BS)\cdot H\cdot4H
=8BSH^2
$$

若 $B=8,S=2048,H=4096$：

$$
8\times8\times2048\times4096^2
\approx 2.20\times10^{12}\ \text{FLOPs}
$$

这只是一次前向线性投影，说明大模型为何高度依赖 Tensor Core GEMM。

### 9.2 算例二：多头 Attention 的完整维度

从：

$$
\mathbf{X}\in\mathbb{R}^{B\times S\times H}
$$

生成 Q/K/V 后 reshape 并转置：

$$
\mathbf{Q},\mathbf{K},\mathbf{V}
\in\mathbb{R}^{B\times N_h\times S\times D_h}
$$

其中 $H=N_hD_h$。

分数矩阵：

$$
\mathbf{S}
=\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{D_h}}
\in\mathbb{R}^{B\times N_h\times S\times S}
$$

应用因果 mask 和 Softmax：

$$
\mathbf{P}=\operatorname{softmax}(\mathbf{S}+\mathbf{M})
$$

其中未来位置的 mask 逻辑上为 $-\infty$，使其 Softmax 概率为 0。

输出：

$$
\mathbf{O}=\mathbf{P}\mathbf{V}
\in\mathbb{R}^{B\times N_h\times S\times D_h}
$$

合并 head 后回到 $(B,S,H)$。

若显式物化 FP16 的 $\mathbf{S}$，仅它就需要：

$$
2BN_hS^2\ \text{bytes}
$$

当 $B=1,N_h=32,S=32768$ 时：

$$
2\times1\times32\times32768^2
=64\ \text{GiB}
$$

这还没算概率、反向中间量和其他层。FlashAttention 的关键就是不把完整 $S\times S$ 中间矩阵写回 HBM，而在片上分块完成稳定 Softmax 与 $PV$ 累积。

### 9.3 算例三：为什么除以 $\sqrt{D_h}$

假设 $q_i,k_i$ 独立、均值 0、方差 1：

$$
\mathbf{q}^\top\mathbf{k}
=\sum_{i=1}^{D_h}q_i k_i
$$

每项 $q_i k_i$ 期望约为 0、方差约为 1，则和的方差约为 $D_h$，标准差约为 $\sqrt{D_h}$。

除以 $\sqrt{D_h}$ 后，分数方差恢复到约 1：

$$
\operatorname{Var}\left(
\frac{\mathbf{q}^\top\mathbf{k}}{\sqrt{D_h}}
\right)\approx1
$$

避免维度增大时 logits 过度扩张，Softmax 进入极端饱和区。

### 9.4 算例四：Online Softmax 的分块合并

一行 logits 被切成多个 block。已处理部分的最大值和指数和为 $(m,l)$：

$$
m=\max_i x_i,\qquad
l=\sum_i e^{x_i-m}
$$

新 block 的状态为 $(m_b,l_b)$。合并最大值：

$$
m_{new}=\max(m,m_b)
$$

调整到同一基准后合并指数和：

$$
l_{new}
= e^{m-m_{new}}l
+e^{m_b-m_{new}}l_b
$$

这使 Softmax 可以分块、流式且数值稳定地计算。若同时维护加权 Value 累加器，还能避免保存完整注意力矩阵，这是 FlashAttention 的数学核心之一。

### 9.5 算例五：LoRA 的参数与计算

完整线性层：

$$
\mathbf{y}=\mathbf{x}\mathbf{W}_0^\top
$$

LoRA：

$$
\mathbf{y}
=\mathbf{x}\mathbf{W}_0^\top
+\frac{\alpha}{r}
  (\mathbf{x}\mathbf{A}^\top)\mathbf{B}^\top
$$

形状：

```text
x:     (..., din)
A^T:   (din, r)       -> (..., r)
B^T:   (r, dout)      -> (..., dout)
```

在线 adapter 会多做两个小 GEMM；若提前把 $\mathbf{B}\mathbf{A}$ 合并进基座权重，推理仍是一个大 GEMM，但失去低成本动态切换 adapter 的便利。

### 9.6 算例六：分布式均值为什么要加权

两个 rank 的局部样本数分别为 $n_1,n_2$，局部均值为 $\mu_1,\mu_2$。全局均值不是简单的 $(\mu_1+\mu_2)/2$，除非 $n_1=n_2$。正确值：

$$
\mu=
\frac{n_1\mu_1+n_2\mu_2}{n_1+n_2}
$$

更一般地，每个 rank 先汇总局部 `sum` 和 `count`，AllReduce 后再相除。分布式 loss、指标和吞吐统计都要明确分母，尤其是最后一个不完整 batch 或序列 mask 不同的场景。

### 9.7 算例七：显存估算

若一个张量 shape 为 $(B,S,H)$，元素字节数为 $b$，其连续存储大小：

$$
M=B\times S\times H\times b
$$

例如 BF16 的 $(8,4096,8192)$：

$$
8\times4096\times8192\times2
=536{,}870{,}912\ \text{bytes}
=512\ \text{MiB}
$$

训练峰值显存还包括参数、梯度、优化器状态、保存的激活、临时 workspace、通信 buffer、内存碎片和框架上下文。单个张量估算只是起点。

### 9.8 从数学式到 Kernel 的固定检查表

看到一个新算子时，按以下顺序拆解：

1. 输入、输出和中间量的 shape 是什么？
2. 哪些维度保留，哪些维度归约？
3. 能否分块？分块状态如何合并？
4. FLOPs 和理论最小访存量是多少？
5. 是否存在广播、转置或非连续 stride？
6. 哪些操作对精度敏感，需要高精度累加？
7. 是否会出现 exp 溢出、除零、消减或长归约误差？
8. 中间张量能否融合消除？
9. 边界 shape 和尾块如何处理？
10. 用什么参考实现、dtype 容差和极端输入验证？

这份检查表把抽象数学转换成可执行的 Infra 工作流。


---

来源：[AIInfraGuide · 第2章：数学基础](https://caomaolufei.github.io/AIInfraGuide/guides/%E6%A8%A1%E5%9D%97%E4%B8%80-%E5%89%8D%E7%BD%AE%E7%9F%A5%E8%AF%86/%E7%AC%AC2%E7%AB%A0-%E6%95%B0%E5%AD%A6%E5%9F%BA%E7%A1%80/)，作者/项目：caomaolufei/AIInfraGuide。按原章节拆分，保留原文、公式与代码；项目 [README 标注 MIT 许可](https://github.com/caomaolufei/AIInfraGuide#license)。

[返回目录](README.md)

