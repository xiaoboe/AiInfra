# 5. Softmax、交叉熵与 KL 散度

### 5.1 Logits 不是概率

模型最后一层输出 $V$ 个任意实数：

$$
\mathbf{z}=(z_1,z_2,\ldots,z_V)
$$

这些值叫 logits。它们可以为负，也不要求和为 1。Softmax 把它们转换为分类分布：

$$
p_i = \operatorname{softmax}(\mathbf{z})_i
= \frac{e^{z_i}}{\sum_{j=1}^{V}e^{z_j}}
$$

显然 $p_i>0$ 且 $\sum_i p_i=1$。

Softmax 对统一平移不敏感：

$$
\operatorname{softmax}(\mathbf{z}+c)
= \operatorname{softmax}(\mathbf{z})
$$

因为分子分母都会乘以 $e^c$。这个性质正是稳定实现的依据。

### 5.2 数值稳定的 Softmax

若直接计算 $e^{1000}$，有限精度浮点数会溢出。令：

$$
m=\max_j z_j
$$

稳定形式为：

$$
p_i=
\frac{e^{z_i-m}}
{\sum_j e^{z_j-m}}
$$

最大的指数输入为 0，因此最大指数值为 1；其他值不大于 1。伪代码：

```python
def stable_softmax(x):
    maximum = max(x)
    exps = [exp(value - maximum) for value in x]
    denominator = sum(exps)
    return [value / denominator for value in exps]
```

减最大值防止正向溢出，但很小的项仍可能下溢到 0。这通常代表它相对最大项确实可以忽略；如果后续还要取对数，则应直接使用稳定的 `log_softmax`，避免先变成 0 再 `log(0)`。

### 5.3 LogSumExp

定义：

$$
\operatorname{LSE}(\mathbf{z})
= \log\sum_j e^{z_j}
$$

稳定形式：

$$
\operatorname{LSE}(\mathbf{z})
= m + \log\sum_j e^{z_j-m}
$$

于是：

$$
\log\operatorname{softmax}(\mathbf{z})_i
= z_i - \operatorname{LSE}(\mathbf{z})
$$

交叉熵实现通常融合 `log_softmax + NLLLoss`，既减少中间张量和访存，也避免不稳定的“先 Softmax 再取 log”。

### 5.4 温度参数

带温度 $T>0$ 的 Softmax：

$$
p_i(T)=
\frac{e^{z_i/T}}{\sum_j e^{z_j/T}}
$$

- $T<1$：放大 logit 差异，分布更尖锐；
- $T>1$：缩小差异，分布更平坦；
- $T\to0^+$：逐渐接近只选择最大 logit；
- $T\to\infty$：逐渐接近均匀分布。

温度改变的是采样分布，不等于修改模型权重。工程实现应先缩放 logits，再应用稳定 Softmax，并谨慎处理非常小的温度。

### 5.5 交叉熵

真实分布 $\mathbf{q}$ 与模型分布 $\mathbf{p}$ 的交叉熵：

$$
H(\mathbf{q},\mathbf{p})
= -\sum_i q_i\log p_i
$$

若真实标签是 one-hot，正确类别为 $y$：

$$
q_y=1,\quad q_{i\ne y}=0
$$

则：

$$
\mathcal{L}
= -\log p_y
$$

模型给正确类别越高概率，损失越小。若 $p_y=1$，损失为 0；若 $p_y$ 接近 0，损失很大。

### 5.6 信息熵

分布自身的熵：

$$
H(\mathbf{p})
= -\sum_i p_i\log p_i
$$

熵衡量不确定性：

- 全部概率集中在一个类别时，熵最小；
- $V$ 个类别均匀分布时，熵最大，为 $\log V$。

对数底决定单位：自然对数对应 nat，以 2 为底对应 bit。机器学习损失通常使用自然对数。

### 5.7 KL 散度

从分布 $\mathbf{q}$ 到 $\mathbf{p}$ 的 KL 散度：

$$
D_{KL}(\mathbf{q}\lVert\mathbf{p})
= \sum_i q_i\log\frac{q_i}{p_i}
$$

它满足：

$$
D_{KL}(\mathbf{q}\lVert\mathbf{p})\ge0
$$

且两分布相同时为 0。但它通常不对称：

$$
D_{KL}(\mathbf{q}\lVert\mathbf{p})
\ne D_{KL}(\mathbf{p}\lVert\mathbf{q})
$$

所以 KL 散度不是严格意义的距离。

交叉熵可分解为：

$$
H(\mathbf{q},\mathbf{p})
= H(\mathbf{q})
+ D_{KL}(\mathbf{q}\lVert\mathbf{p})
$$

训练时真实分布 $\mathbf{q}$ 固定，$H(\mathbf{q})$ 与模型参数无关，所以最小化交叉熵等价于最小化相应 KL 散度。

KL 散度会出现在知识蒸馏、分布匹配、RLHF/PPO 约束和推测解码分析中。实现时要明确 API 接收的是概率、log 概率还是 logits，以及 KL 的方向。

### 5.8 Perplexity

若平均 token 负对数似然为 $\bar{L}$，困惑度：

$$
\operatorname{PPL}=e^{\bar{L}}
$$

它可直觉理解为模型在每一步面对的“有效候选数”。但不同 tokenizer、数据预处理、上下文长度和是否忽略特殊 token 都会影响 PPL，不能脱离评测设置直接横比。

### 5.9 Top-k 与 Top-p 采样

模型输出分布后，解码策略决定如何选 token：

- Greedy：选择最大概率 token；
- Top-k：只在概率最高的 $k$ 个 token 中采样；
- Top-p（nucleus）：选择累计概率至少达到 $p$ 的最小候选集合，再归一化采样。

Top-p 集合大小会随分布尖锐程度动态变化。实现时通常先过滤 logits，把被排除项设为负无穷，再执行稳定 Softmax 和采样。

分布式推理若 vocabulary 被张量并行切分，global top-k/top-p 需要跨设备聚合局部候选或使用等价的分布式算法。


---

来源：[AIInfraGuide · 第2章：数学基础](https://caomaolufei.github.io/AIInfraGuide/guides/%E6%A8%A1%E5%9D%97%E4%B8%80-%E5%89%8D%E7%BD%AE%E7%9F%A5%E8%AF%86/%E7%AC%AC2%E7%AB%A0-%E6%95%B0%E5%AD%A6%E5%9F%BA%E7%A1%80/)，作者/项目：caomaolufei/AIInfraGuide。按原章节拆分，保留原文、公式与代码；项目 [README 标注 MIT 许可](https://github.com/caomaolufei/AIInfraGuide#license)。

[返回目录](README.md)

