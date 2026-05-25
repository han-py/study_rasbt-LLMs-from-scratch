# 📌 PyTorch 中常见 `grad_fn` 对照表

> **先说结论：**
> `grad_fn` 不是“张量本身的值”，而是 **这个张量是通过什么运算得到的**、以及 **反向传播应该怎么回去** 的记录。
>
> 当你看到：
>
> ```python
> tensor(..., grad_fn=<AddmmBackward0>)
> ```
>
> 它的意思就是：这个张量不是直接创建的，而是经过某种可求导运算算出来的。

---

## 🌟 一、`grad_fn` 是什么？

在 PyTorch 里，只要一个张量是通过可求导运算得到的，并且它参与了计算图，那么它通常就会带上 `grad_fn`。

### 你可以这样理解

- **前向传播**：张量是怎么一步步算出来的
- **反向传播**：梯度应该沿着哪条路径回传
- **`grad_fn`**：记录“这一步运算的反向规则”的节点

### 两类张量要先分清

#### 1. 叶子张量（leaf tensor）
通常是你手动创建、并设置了 `requires_grad=True` 的张量：

```python
import torch

w = torch.tensor([2.0], requires_grad=True)
print(w.grad_fn)
```

输出通常是：

```python
None
```

因为它是计算图的起点之一，不是“算出来的中间结果”。

#### 2. 非叶子张量（non-leaf tensor）
比如：

```python
x = torch.tensor([2.0], requires_grad=True)
y = x * 3
print(y.grad_fn)
```

这里 `y` 是通过运算得到的，所以会有 `grad_fn`。

---

## 🌟 二、为什么 `grad_fn` 后面常常带 `0`？

比如：

- `AddmmBackward0`
- `ReluBackward0`
- `SigmoidBackward0`

这里的 `0` 一般表示 **这个反向函数实例的编号或版本标记**，你可以把它先理解成“这一类反向节点的具体对象名”。

> 初学阶段不需要死记这个 `0` 的含义，重点是认出前面的操作名。

---

## 🌟 三、`grad_fn` 常见种类总览

> 说明：PyTorch 的 `grad_fn` 种类非常多，不同版本、不同操作、不同设备上还会有差异。下面整理的是**学习中最常见、最值得掌握的类型**。

---

# 1️⃣ 线性代数 / 矩阵运算类

这类最常见于神经网络层，尤其是 `Linear`、`matmul`、`@` 等操作。

| `grad_fn` | 对应操作 | 初学者理解 | 常见场景 |
|---|---|---|---|
| `AddmmBackward0` | `bias + matrix multiplication` | 线性层最常见：矩阵乘法后加偏置 | `nn.Linear` |
| `MmBackward0` | 矩阵乘法 `mm` | 两个二维矩阵相乘 | 全连接层底层、特征变换 |
| `MatmulBackward0` | 通用矩阵乘法 `matmul` | 比 `mm` 更通用 | `x @ W` |
| `BmmBackward0` | batch 矩阵乘法 | 一批矩阵一起乘 | 注意力机制、多头计算 |
| `AddmvBackward0` | 矩阵向量乘法 + 加法 | 矩阵乘向量 | 较少见的线性代数操作 |
| `MvBackward0` | 矩阵向量乘法 | 矩阵和向量相乘 | 线代基础运算 |

### 例子

```python
import torch

x = torch.randn(2, 3, requires_grad=True)
w = torch.randn(3, 4, requires_grad=True)
y = x @ w
print(y.grad_fn)
```

可能看到：

```python
<MatmulBackward0>
```

如果写成 `nn.Linear`，常见则会看到 `AddmmBackward0`。

---

# 2️⃣ 激活函数类

激活函数会给模型引入非线性，让模型能拟合更复杂的关系。

| `grad_fn` | 对应操作 | 初学者理解 | 常见场景 |
|---|---|---|---|
| `ReluBackward0` | `ReLU` | 小于 0 的部分被截断 | 隐藏层 |
| `SigmoidBackward0` | `sigmoid` | 把值压到 0~1 之间 | 二分类输出 |
| `TanhBackward0` | `tanh` | 把值压到 -1~1 之间 | RNN、传统神经网络 |
| `GeluBackward0` | `GELU` | 更平滑的激活函数 | Transformer、GPT |
| `LeakyReluBackward0` | `LeakyReLU` | 负半轴保留一点梯度 | CNN、MLP |
| `EluBackward0` | `ELU` | 指数型激活 | 深度网络 |
| `SoftplusBackward0` | `Softplus` | ReLU 的平滑版本 | 某些概率模型 |

### 例子

```python
import torch

x = torch.tensor([-1.0, 0.5], requires_grad=True)
y = torch.relu(x)
print(y.grad_fn)
```

可能输出：

```python
<ReluBackward0>
```

---

# 3️⃣ 元素级运算类

这类操作会对张量里的每个元素分别计算。

| `grad_fn` | 对应操作 | 初学者理解 | 常见场景 |
|---|---|---|---|
| `AddBackward0` | 加法 | 逐元素相加 | 张量运算 |
| `SubBackward0` | 减法 | 逐元素相减 | 张量运算 |
| `MulBackward0` | 乘法 | 逐元素相乘 | 张量运算 |
| `DivBackward0` | 除法 | 逐元素相除 | 张量运算 |
| `PowBackward0` | 幂运算 | 平方、立方等 | 损失、变换 |
| `NegBackward0` | 取负 | 前面加负号 | 代数运算 |
| `ExpBackward0` | 指数函数 | `e^x` | 概率、softmax |
| `LogBackward0` | 对数函数 | `log(x)` | 损失函数 |
| `AbsBackward0` | 绝对值 | 取绝对值 | 某些损失 |
| `ClampBackward0` | 截断 | 限制数值范围 | 数值稳定性 |

### 例子

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
print(y.grad_fn)
```

一般会看到：

```python
<PowBackward0>
```

---

# 4️⃣ 损失函数类

训练模型时，最常接触的就是损失函数对应的反向节点。

| `grad_fn` | 对应操作 | 初学者理解 | 常见场景 |
|---|---|---|---|
| `MseLossBackward0` | 均方误差 | 预测值和真实值差距的平方平均 | 回归 |
| `L1LossBackward0` | L1 损失 | 预测值和真实值差距的绝对值平均 | 回归 |
| `BinaryCrossEntropyBackward0` | 二元交叉熵 | 二分类损失 | 二分类 |
| `BinaryCrossEntropyWithLogitsBackward0` | 带 logits 的二元交叉熵 | 更稳定的二分类损失 | 二分类 |
| `NllLossBackward0` | 负对数似然损失 | 分类任务常见损失 | 分类 |
| `CrossEntropyLossBackward0` | 交叉熵损失 | 分类任务最常见损失 | 多分类 |
| `SmoothL1LossBackward0` | Smooth L1 | MSE 和 L1 的折中 | 检测、回归 |

### 例子

```python
import torch
import torch.nn.functional as F

pred = torch.tensor([0.8], requires_grad=True)
target = torch.tensor([1.0])
loss = F.binary_cross_entropy(pred, target)
print(loss.grad_fn)
```

常见输出：

```python
<BinaryCrossEntropyBackward0>
```

---

# 5️⃣ 归一化 / 概率变换类

这类操作常出现在分类模型、语言模型和注意力机制里。

| `grad_fn` | 对应操作 | 初学者理解 | 常见场景 |
|---|---|---|---|
| `SoftmaxBackward0` | softmax | 把分数变成概率 | 分类输出 |
| `LogSoftmaxBackward0` | log-softmax | softmax 的对数形式 | 分类损失 |
| `NormBackward0` | 各种范数 | 向量长度/归一化 | 正则化、特征处理 |
| `NormalizeBackward0` | 归一化 | 统一尺度 | 表示学习 |

### 例子

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = torch.softmax(x, dim=0)
print(y.grad_fn)
```

通常会看到：

```python
<SoftmaxBackward0>
```

---

# 6️⃣ 形状变换类

这类不会改变数值本身，只改变张量的形状、维度顺序或视图。

| `grad_fn` | 对应操作 | 初学者理解 | 常见场景 |
|---|---|---|---|
| `ViewBackward0` | `view()` | 改变视图 | reshape 类操作 |
| `ReshapeAliasBackward0` | `reshape()` | 改变形状 | reshape 类操作 |
| `TransposeBackward0` | `transpose()` | 交换两个维度 | 矩阵、注意力 |
| `TBackward0` | `.t()` | 二维矩阵转置 | 线性代数 |
| `PermuteBackward0` | `permute()` | 重新排列维度 | 图像、NLP |
| `SqueezeBackward0` | `squeeze()` | 去掉长度为 1 的维度 | 批量维处理 |
| `UnsqueezeBackward0` | `unsqueeze()` | 增加一个长度为 1 的维度 | 对齐维度 |
| `FlattenBackward0` | `flatten()` | 压平张量 | MLP 输入 |
| `SliceBackward0` | 切片 | 截取部分数据 | 数据选择 |
| `SelectBackward0` | 选取元素 | 取某个位置 | 索引操作 |
| `CatBackward0` | `torch.cat()` | 拼接张量 | 特征融合 |
| `StackBackward0` | `torch.stack()` | 堆叠张量 | 批处理 |

### 例子

```python
import torch

x = torch.randn(2, 3, requires_grad=True)
y = x.view(3, 2)
print(y.grad_fn)
```

常见输出：

```python
<ViewBackward0>
```

---

# 7️⃣ 拼接 / 拆分 / 选择类

这些操作常用于把多个张量拼起来，或者从张量中取出一部分。

| `grad_fn` | 对应操作 | 初学者理解 | 常见场景 |
|---|---|---|---|
| `CatBackward0` | `cat` | 拼接 | 特征拼接 |
| `StackBackward0` | `stack` | 堆叠 | 批量收集 |
| `SplitBackward0` | `split` | 拆分 | 切分张量 |
| `ChunkBackward0` | `chunk` | 分块 | 切分张量 |
| `GatherBackward0` | `gather` | 按索引取值 | 注意力、检索 |
| `IndexSelectBackward0` | 索引选择 | 按索引取片段 | 取词、取特征 |

> 注意：有些索引类操作在不同版本中显示的名字可能略有不同。

---

# 8️⃣ 类型转换 / 复制类

这类操作通常不直接改变数值计算逻辑，但会影响张量的数据类型、设备或复制方式。

| `grad_fn` | 对应操作 | 初学者理解 | 常见场景 |
|---|---|---|---|
| `CloneBackward0` | `clone()` | 复制一份张量 | 保险起见复制 |
| `CopyBackwards` | 拷贝 | 从一个地方复制到另一个地方 | 张量复制 |
| `ToCopyBackward0` | `.to(...)` | 类型/设备转换 | CPU/GPU、dtype |
| `DetachBackward0` | `detach()` | 断开梯度 | 推理、截断计算图 |

### 例子

```python
import torch

x = torch.tensor([1.0], requires_grad=True)
y = x.clone()
print(y.grad_fn)
```

常见输出：

```python
<CloneBackward0>
```

---

# 9️⃣ 梯度累积 / 参数相关

| `grad_fn` | 对应操作 | 初学者理解 | 常见场景 |
|---|---|---|---|
| `AccumulateGrad` | 梯度累积 | 把梯度存到参数 `.grad` 里 | 叶子参数 |

### 这个很重要

如果一个张量是模型参数，并且 `requires_grad=True`，它本身通常是 **leaf tensor**。这类张量通常不会显示普通的 `grad_fn`，但在反向传播时，会通过 `AccumulateGrad` 把梯度累积到它的 `.grad` 中。

---

## 🌟 四、最常见的几个 `grad_fn`，你先记住这些就够了

如果你刚开始学，不需要背所有名字，先把下面这些记熟：

| `grad_fn` | 你可以把它理解成 |
|---|---|
| `AddmmBackward0` | 线性层：矩阵乘法 + 偏置 |
| `MmBackward0` | 矩阵乘法 |
| `MatmulBackward0` | 通用矩阵乘法 |
| `ReluBackward0` | ReLU 激活 |
| `SigmoidBackward0` | Sigmoid 激活 |
| `TanhBackward0` | Tanh 激活 |
| `GeluBackward0` | GELU 激活 |
| `PowBackward0` | 幂运算 |
| `SoftmaxBackward0` | softmax 概率变换 |
| `BinaryCrossEntropyBackward0` | 二分类交叉熵损失 |
| `MseLossBackward0` | 均方误差损失 |
| `ViewBackward0` | 改变形状 |
| `AccumulateGrad` | 参数梯度累积 |

---

## 🌟 五、结合你的 MLP 例子来理解

你前面的例子里：

```python
model = NeuralNetwork(50, 3)
x = torch.rand(1, 50)
out = model(x)
print(out)
```

输出类似：

```python
tensor([[-0.1262,  0.1080, -0.1792]], grad_fn=<AddmmBackward0>)
```

这说明：

1. `out` 是模型最后一层算出来的结果
2. 最后一层是 `Linear(20, 3)`
3. `Linear` 层底层对应的就是矩阵乘法 + 偏置加法
4. 所以它的反向节点显示为 `AddmmBackward0`

换句话说：

> 这个输出张量不是“静态数字”，而是一个还可以继续求导的中间结果。

---

## 🌟 六、如何自己查看 `grad_fn`？

你可以这样写：

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x * 3
z = y + 4
loss = z ** 2

print("y.grad_fn =", y.grad_fn)
print("z.grad_fn =", z.grad_fn)
print("loss.grad_fn =", loss.grad_fn)
```

可能输出：

```python
y.grad_fn = <MulBackward0>
z.grad_fn = <AddBackward0>
loss.grad_fn = <PowBackward0>
```

这非常适合你用来学习“每一步运算在计算图中对应什么节点”。

---

## 🌟 七、学习 `grad_fn` 的小技巧

### 技巧 1：先认“操作名”，再看后缀
比如：
- `AddmmBackward0` → 先认出 `Addmm`
- `ReluBackward0` → 先认出 `Relu`
- `PowBackward0` → 先认出 `Pow`

### 技巧 2：不要一开始死磕全部名字
PyTorch 的 `grad_fn` 类型很多，版本也会变化。
最重要的是先理解：
- 它是前向运算留下的痕迹
- 它告诉你梯度怎么回传

### 技巧 3：把 `grad_fn` 和具体代码对应起来
例如：
- `x ** 2` → `PowBackward0`
- `torch.relu(x)` → `ReluBackward0`
- `nn.Linear(...)` → 常见 `AddmmBackward0`

---

## 🎯 八、练习题

### 练习 1：猜 `grad_fn`
下面代码会出现什么类型的 `grad_fn`？

```python
import torch
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3
print(y.grad_fn)
```

---

### 练习 2：线性层的 `grad_fn`
下面代码输出的 `grad_fn` 通常是什么？

```python
import torch
import torch.nn as nn

layer = nn.Linear(4, 2)
x = torch.randn(1, 4)
out = layer(x)
print(out.grad_fn)
```

---

### 练习 3：激活函数的 `grad_fn`

```python
import torch
x = torch.tensor([-1.0, 1.0], requires_grad=True)
y = torch.sigmoid(x)
print(y.grad_fn)
```

---

### 练习 4：形状变化的 `grad_fn`

```python
import torch
x = torch.randn(2, 3, requires_grad=True)
y = x.reshape(3, 2)
print(y.grad_fn)
```

---

## ✅ 参考答案

<details>
<summary>点击展开答案</summary>

### 练习 1 答案

```python
<PowBackward0>
```

因为 `x ** 3` 是幂运算。

### 练习 2 答案

通常是：

```python
<AddmmBackward0>
```

因为 `nn.Linear` 底层通常是“矩阵乘法 + 偏置加法”。

### 练习 3 答案

```python
<SigmoidBackward0>
```

### 练习 4 答案

通常是：

```python
<ViewBackward0>
```

或者在某些版本中可能看到与 reshape 相关的其他名字。

</details>

---

## 🌈 九、小结

你可以把 `grad_fn` 理解成：

> **“这个张量是怎么被算出来的，以及反向传播怎么回去”的记录卡。**

最常见的几个名字你先记住：

- `AddmmBackward0`：线性层
- `MmBackward0`：矩阵乘法
- `ReluBackward0`：ReLU
- `SigmoidBackward0`：Sigmoid
- `PowBackward0`：幂运算
- `BinaryCrossEntropyBackward0`：二分类交叉熵
- `ViewBackward0`：形状变换
- `AccumulateGrad`：参数梯度累积

如果你能把这些和具体代码对应起来，后面看计算图、看反向传播、看神经网络训练日志时，就会轻松很多。
# 📌 PyTorch 中常见 `grad_fn` 对照表

> **先说结论：**
> `grad_fn` 不是“张量本身的值”，而是 **这个张量是通过什么运算得到的**、以及 **反向传播应该怎么回去** 的记录。
>
> 当你看到：
>
> ```python
> tensor(..., grad_fn=<AddmmBackward0>)
> ```
>
> 它的意思就是：这个张量不是直接创建的，而是经过某种可求导运算算出来的。

---

## 🌟 一、`grad_fn` 是什么？

在 PyTorch 里，只要一个张量是通过可求导运算得到的，并且它参与了计算图，那么它通常就会带上 `grad_fn`。

### 你可以这样理解

- **前向传播**：张量是怎么一步步算出来的
- **反向传播**：梯度应该沿着哪条路径回传
- **`grad_fn`**：记录“这一步运算的反向规则”的节点

### 两类张量要先分清

#### 1. 叶子张量（leaf tensor）
通常是你手动创建、并设置了 `requires_grad=True` 的张量：

```python
import torch

w = torch.tensor([2.0], requires_grad=True)
print(w.grad_fn)
```

输出通常是：

```python
None
```

因为它是计算图的起点之一，不是“算出来的中间结果”。

#### 2. 非叶子张量（non-leaf tensor）
比如：

```python
x = torch.tensor([2.0], requires_grad=True)
y = x * 3
print(y.grad_fn)
```

这里 `y` 是通过运算得到的，所以会有 `grad_fn`。

---

## 🌟 二、为什么 `grad_fn` 后面常常带 `0`？

比如：

- `AddmmBackward0`
- `ReluBackward0`
- `SigmoidBackward0`

这里的 `0` 一般表示 **这个反向函数实例的编号或版本标记**，你可以把它先理解成“这一类反向节点的具体对象名”。

> 初学阶段不需要死记这个 `0` 的含义，重点是认出前面的操作名。

---

## 🌟 三、`grad_fn` 常见种类总览

> 说明：PyTorch 的 `grad_fn` 种类非常多，不同版本、不同操作、不同设备上还会有差异。下面整理的是**学习中最常见、最值得掌握的类型**。

---

# 1️⃣ 线性代数 / 矩阵运算类

这类最常见于神经网络层，尤其是 `Linear`、`matmul`、`@` 等操作。

| `grad_fn` | 对应操作 | 初学者理解 | 常见场景 |
|---|---|---|---|
| `AddmmBackward0` | `bias + matrix multiplication` | 线性层最常见：矩阵乘法后加偏置 | `nn.Linear` |
| `MmBackward0` | 矩阵乘法 `mm` | 两个二维矩阵相乘 | 全连接层底层、特征变换 |
| `MatmulBackward0` | 通用矩阵乘法 `matmul` | 比 `mm` 更通用 | `x @ W` |
| `BmmBackward0` | batch 矩阵乘法 | 一批矩阵一起乘 | 注意力机制、多头计算 |
| `AddmvBackward0` | 矩阵向量乘法 + 加法 | 矩阵乘向量 | 较少见的线性代数操作 |
| `MvBackward0` | 矩阵向量乘法 | 矩阵和向量相乘 | 线代基础运算 |

### 例子

```python
import torch

x = torch.randn(2, 3, requires_grad=True)
w = torch.randn(3, 4, requires_grad=True)
y = x @ w
print(y.grad_fn)
```

可能看到：

```python
<MatmulBackward0>
```

如果写成 `nn.Linear`，常见则会看到 `AddmmBackward0`。

---

# 2️⃣ 激活函数类

激活函数会给模型引入非线性，让模型能拟合更复杂的关系。

| `grad_fn` | 对应操作 | 初学者理解 | 常见场景 |
|---|---|---|---|
| `ReluBackward0` | `ReLU` | 小于 0 的部分被截断 | 隐藏层 |
| `SigmoidBackward0` | `sigmoid` | 把值压到 0~1 之间 | 二分类输出 |
| `TanhBackward0` | `tanh` | 把值压到 -1~1 之间 | RNN、传统神经网络 |
| `GeluBackward0` | `GELU` | 更平滑的激活函数 | Transformer、GPT |
| `LeakyReluBackward0` | `LeakyReLU` | 负半轴保留一点梯度 | CNN、MLP |
| `EluBackward0` | `ELU` | 指数型激活 | 深度网络 |
| `SoftplusBackward0` | `Softplus` | ReLU 的平滑版本 | 某些概率模型 |

### 例子

```python
import torch

x = torch.tensor([-1.0, 0.5], requires_grad=True)
y = torch.relu(x)
print(y.grad_fn)
```

可能输出：

```python
<ReluBackward0>
```

---

# 3️⃣ 元素级运算类

这类操作会对张量里的每个元素分别计算。

| `grad_fn` | 对应操作 | 初学者理解 | 常见场景 |
|---|---|---|---|
| `AddBackward0` | 加法 | 逐元素相加 | 张量运算 |
| `SubBackward0` | 减法 | 逐元素相减 | 张量运算 |
| `MulBackward0` | 乘法 | 逐元素相乘 | 张量运算 |
| `DivBackward0` | 除法 | 逐元素相除 | 张量运算 |
| `PowBackward0` | 幂运算 | 平方、立方等 | 损失、变换 |
| `NegBackward0` | 取负 | 前面加负号 | 代数运算 |
| `ExpBackward0` | 指数函数 | `e^x` | 概率、softmax |
| `LogBackward0` | 对数函数 | `log(x)` | 损失函数 |
| `AbsBackward0` | 绝对值 | 取绝对值 | 某些损失 |
| `ClampBackward0` | 截断 | 限制数值范围 | 数值稳定性 |

### 例子

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
print(y.grad_fn)
```

一般会看到：

```python
<PowBackward0>
```

---

# 4️⃣ 损失函数类

训练模型时，最常接触的就是损失函数对应的反向节点。

| `grad_fn` | 对应操作 | 初学者理解 | 常见场景 |
|---|---|---|---|
| `MseLossBackward0` | 均方误差 | 预测值和真实值差距的平方平均 | 回归 |
| `L1LossBackward0` | L1 损失 | 预测值和真实值差距的绝对值平均 | 回归 |
| `BinaryCrossEntropyBackward0` | 二元交叉熵 | 二分类损失 | 二分类 |
| `BinaryCrossEntropyWithLogitsBackward0` | 带 logits 的二元交叉熵 | 更稳定的二分类损失 | 二分类 |
| `NllLossBackward0` | 负对数似然损失 | 分类任务常见损失 | 分类 |
| `CrossEntropyLossBackward0` | 交叉熵损失 | 分类任务最常见损失 | 多分类 |
| `SmoothL1LossBackward0` | Smooth L1 | MSE 和 L1 的折中 | 检测、回归 |

### 例子

```python
import torch
import torch.nn.functional as F

pred = torch.tensor([0.8], requires_grad=True)
target = torch.tensor([1.0])
loss = F.binary_cross_entropy(pred, target)
print(loss.grad_fn)
```

常见输出：

```python
<BinaryCrossEntropyBackward0>
```

---

# 5️⃣ 归一化 / 概率变换类

这类操作常出现在分类模型、语言模型和注意力机制里。

| `grad_fn` | 对应操作 | 初学者理解 | 常见场景 |
|---|---|---|---|
| `SoftmaxBackward0` | softmax | 把分数变成概率 | 分类输出 |
| `LogSoftmaxBackward0` | log-softmax | softmax 的对数形式 | 分类损失 |
| `NormBackward0` | 各种范数 | 向量长度/归一化 | 正则化、特征处理 |
| `NormalizeBackward0` | 归一化 | 统一尺度 | 表示学习 |

### 例子

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = torch.softmax(x, dim=0)
print(y.grad_fn)
```

通常会看到：

```python
<SoftmaxBackward0>
```

---

# 6️⃣ 形状变换类

这类不会改变数值本身，只改变张量的形状、维度顺序或视图。

| `grad_fn` | 对应操作 | 初学者理解 | 常见场景 |
|---|---|---|---|
| `ViewBackward0` | `view()` | 改变视图 | reshape 类操作 |
| `ReshapeAliasBackward0` | `reshape()` | 改变形状 | reshape 类操作 |
| `TransposeBackward0` | `transpose()` | 交换两个维度 | 矩阵、注意力 |
| `TBackward0` | `.t()` | 二维矩阵转置 | 线性代数 |
| `PermuteBackward0` | `permute()` | 重新排列维度 | 图像、NLP |
| `SqueezeBackward0` | `squeeze()` | 去掉长度为 1 的维度 | 批量维处理 |
| `UnsqueezeBackward0` | `unsqueeze()` | 增加一个长度为 1 的维度 | 对齐维度 |
| `FlattenBackward0` | `flatten()` | 压平张量 | MLP 输入 |
| `SliceBackward0` | 切片 | 截取部分数据 | 数据选择 |
| `SelectBackward0` | 选取元素 | 取某个位置 | 索引操作 |
| `CatBackward0` | `torch.cat()` | 拼接张量 | 特征融合 |
| `StackBackward0` | `torch.stack()` | 堆叠张量 | 批处理 |

### 例子

```python
import torch

x = torch.randn(2, 3, requires_grad=True)
y = x.view(3, 2)
print(y.grad_fn)
```

常见输出：

```python
<ViewBackward0>
```

---

# 7️⃣ 拼接 / 拆分 / 选择类

这些操作常用于把多个张量拼起来，或者从张量中取出一部分。

| `grad_fn` | 对应操作 | 初学者理解 | 常见场景 |
|---|---|---|---|
| `CatBackward0` | `cat` | 拼接 | 特征拼接 |
| `StackBackward0` | `stack` | 堆叠 | 批量收集 |
| `SplitBackward0` | `split` | 拆分 | 切分张量 |
| `ChunkBackward0` | `chunk` | 分块 | 切分张量 |
| `GatherBackward0` | `gather` | 按索引取值 | 注意力、检索 |
| `IndexSelectBackward0` | 索引选择 | 按索引取片段 | 取词、取特征 |

> 注意：有些索引类操作在不同版本中显示的名字可能略有不同。

---

# 8️⃣ 类型转换 / 复制类

这类操作通常不直接改变数值计算逻辑，但会影响张量的数据类型、设备或复制方式。

| `grad_fn` | 对应操作 | 初学者理解 | 常见场景 |
|---|---|---|---|
| `CloneBackward0` | `clone()` | 复制一份张量 | 保险起见复制 |
| `CopyBackwards` | 拷贝 | 从一个地方复制到另一个地方 | 张量复制 |
| `ToCopyBackward0` | `.to(...)` | 类型/设备转换 | CPU/GPU、dtype |
| `DetachBackward0` | `detach()` | 断开梯度 | 推理、截断计算图 |

### 例子

```python
import torch

x = torch.tensor([1.0], requires_grad=True)
y = x.clone()
print(y.grad_fn)
```

常见输出：

```python
<CloneBackward0>
```

---

# 9️⃣ 梯度累积 / 参数相关

| `grad_fn` | 对应操作 | 初学者理解 | 常见场景 |
|---|---|---|---|
| `AccumulateGrad` | 梯度累积 | 把梯度存到参数 `.grad` 里 | 叶子参数 |

### 这个很重要

如果一个张量是模型参数，并且 `requires_grad=True`，它本身通常是 **leaf tensor**。这类张量通常不会显示普通的 `grad_fn`，但在反向传播时，会通过 `AccumulateGrad` 把梯度累积到它的 `.grad` 中。

---

## 🌟 四、最常见的几个 `grad_fn`，你先记住这些就够了

如果你刚开始学，不需要背所有名字，先把下面这些记熟：

| `grad_fn` | 你可以把它理解成 |
|---|---|
| `AddmmBackward0` | 线性层：矩阵乘法 + 偏置 |
| `MmBackward0` | 矩阵乘法 |
| `MatmulBackward0` | 通用矩阵乘法 |
| `ReluBackward0` | ReLU 激活 |
| `SigmoidBackward0` | Sigmoid 激活 |
| `TanhBackward0` | Tanh 激活 |
| `GeluBackward0` | GELU 激活 |
| `PowBackward0` | 幂运算 |
| `SoftmaxBackward0` | softmax 概率变换 |
| `BinaryCrossEntropyBackward0` | 二分类交叉熵损失 |
| `MseLossBackward0` | 均方误差损失 |
| `ViewBackward0` | 改变形状 |
| `AccumulateGrad` | 参数梯度累积 |

---

## 🌟 五、结合你的 MLP 例子来理解

你前面的例子里：

```python
model = NeuralNetwork(50, 3)
x = torch.rand(1, 50)
out = model(x)
print(out)
```

输出类似：

```python
tensor([[-0.1262,  0.1080, -0.1792]], grad_fn=<AddmmBackward0>)
```

这说明：

1. `out` 是模型最后一层算出来的结果
2. 最后一层是 `Linear(20, 3)`
3. `Linear` 层底层对应的就是矩阵乘法 + 偏置加法
4. 所以它的反向节点显示为 `AddmmBackward0`

换句话说：

> 这个输出张量不是“静态数字”，而是一个还可以继续求导的中间结果。

---

## 🌟 六、如何自己查看 `grad_fn`？

你可以这样写：

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x * 3
z = y + 4
loss = z ** 2

print("y.grad_fn =", y.grad_fn)
print("z.grad_fn =", z.grad_fn)
print("loss.grad_fn =", loss.grad_fn)
```

可能输出：

```python
y.grad_fn = <MulBackward0>
z.grad_fn = <AddBackward0>
loss.grad_fn = <PowBackward0>
```

这非常适合你用来学习“每一步运算在计算图中对应什么节点”。

---

## 🌟 七、学习 `grad_fn` 的小技巧

### 技巧 1：先认“操作名”，再看后缀
比如：
- `AddmmBackward0` → 先认出 `Addmm`
- `ReluBackward0` → 先认出 `Relu`
- `PowBackward0` → 先认出 `Pow`

### 技巧 2：不要一开始死磕全部名字
PyTorch 的 `grad_fn` 类型很多，版本也会变化。
最重要的是先理解：
- 它是前向运算留下的痕迹
- 它告诉你梯度怎么回传

### 技巧 3：把 `grad_fn` 和具体代码对应起来
例如：
- `x ** 2` → `PowBackward0`
- `torch.relu(x)` → `ReluBackward0`
- `nn.Linear(...)` → 常见 `AddmmBackward0`

---

## 🎯 八、练习题

### 练习 1：猜 `grad_fn`
下面代码会出现什么类型的 `grad_fn`？

```python
import torch
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3
print(y.grad_fn)
```

---

### 练习 2：线性层的 `grad_fn`
下面代码输出的 `grad_fn` 通常是什么？

```python
import torch
import torch.nn as nn

layer = nn.Linear(4, 2)
x = torch.randn(1, 4)
out = layer(x)
print(out.grad_fn)
```

---

### 练习 3：激活函数的 `grad_fn`

```python
import torch
x = torch.tensor([-1.0, 1.0], requires_grad=True)
y = torch.sigmoid(x)
print(y.grad_fn)
```

---

### 练习 4：形状变化的 `grad_fn`

```python
import torch
x = torch.randn(2, 3, requires_grad=True)
y = x.reshape(3, 2)
print(y.grad_fn)
```

---

## ✅ 参考答案

<details>
<summary>点击展开答案</summary>

### 练习 1 答案

```python
<PowBackward0>
```

因为 `x ** 3` 是幂运算。

### 练习 2 答案

通常是：

```python
<AddmmBackward0>
```

因为 `nn.Linear` 底层通常是“矩阵乘法 + 偏置加法”。

### 练习 3 答案

```python
<SigmoidBackward0>
```

### 练习 4 答案

通常是：

```python
<ViewBackward0>
```

或者在某些版本中可能看到与 reshape 相关的其他名字。

</details>

---

## 🌈 九、小结

你可以把 `grad_fn` 理解成：

> **“这个张量是怎么被算出来的，以及反向传播怎么回去”的记录卡。**

最常见的几个名字你先记住：

- `AddmmBackward0`：线性层
- `MmBackward0`：矩阵乘法
- `ReluBackward0`：ReLU
- `SigmoidBackward0`：Sigmoid
- `PowBackward0`：幂运算
- `BinaryCrossEntropyBackward0`：二分类交叉熵
- `ViewBackward0`：形状变换
- `AccumulateGrad`：参数梯度累积

如果你能把这些和具体代码对应起来，后面看计算图、看反向传播、看神经网络训练日志时，就会轻松很多。


