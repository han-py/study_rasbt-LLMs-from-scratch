# 📌 PyTorch `grad_fn` 速查版

> 这是一个适合快速查阅的简明版：
> 看到 `tensor(..., grad_fn=<...>)` 时，先认出这个张量是由什么运算生成的。

---

## 1. `grad_fn` 是什么？

`grad_fn` 表示这个张量对应的**反向传播节点**。

- 叶子张量：通常是你直接创建、`requires_grad=True` 的张量，`grad_fn` 常为 `None`
- 非叶子张量：由运算得到，通常会有 `grad_fn`

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x * 3
print(x.grad_fn)  # None
print(y.grad_fn)  # <MulBackward0>
```

---

## 2. 最常见的 `grad_fn` 对照表

| `grad_fn` | 对应操作 | 常见场景 |
|---|---|---|
| `AddmmBackward0` | 线性层：`bias + X @ W` | `nn.Linear` |
| `MmBackward0` | 矩阵乘法 | 二维矩阵运算 |
| `MatmulBackward0` | 通用矩阵乘法 | `x @ W` |
| `BmmBackward0` | batch 矩阵乘法 | 注意力、多头 |
| `AddBackward0` | 加法 | `a + b` |
| `SubBackward0` | 减法 | `a - b` |
| `MulBackward0` | 乘法 | `a * b` |
| `DivBackward0` | 除法 | `a / b` |
| `PowBackward0` | 幂运算 | `x ** 2` |
| `ReluBackward0` | ReLU | 隐藏层激活 |
| `SigmoidBackward0` | Sigmoid | 二分类 / 门控 |
| `TanhBackward0` | Tanh | RNN / 门控 |
| `GeluBackward0` | GELU | Transformer / GPT |
| `SoftmaxBackward0` | softmax | 分类概率、注意力 |
| `LogSoftmaxBackward0` | log-softmax | 分类损失 |
| `BinaryCrossEntropyBackward0` | 二元交叉熵 | 二分类损失 |
| `CrossEntropyLossBackward0` | 交叉熵 | 多分类损失 |
| `MseLossBackward0` | 均方误差 | 回归 |
| `L1LossBackward0` | L1 损失 | 回归 |
| `ViewBackward0` | `view()` | 改变形状 |
| `ReshapeAliasBackward0` | `reshape()` | 改变形状 |
| `TransposeBackward0` | `transpose()` | 维度交换 |
| `PermuteBackward0` | `permute()` | 维度重排 |
| `SqueezeBackward0` | `squeeze()` | 去掉 1 维 |
| `UnsqueezeBackward0` | `unsqueeze()` | 增加 1 维 |
| `CatBackward0` | `torch.cat()` | 拼接张量 |
| `StackBackward0` | `torch.stack()` | 堆叠张量 |
| `SliceBackward0` | 切片 | `x[a:b]` |
| `SelectBackward0` | 索引选择 | `x[i]` |
| `IndexBackward0` | 高级索引 | 索引运算 |
| `GatherBackward0` | `gather` | 注意力、检索 |
| `ConvolutionBackward0` | 卷积 | CNN |
| `MaxPool2DWithIndicesBackward0` | 最大池化 | CNN 下采样 |
| `NativeBatchNormBackward0` | BatchNorm | CNN / MLP |
| `NativeLayerNormBackward0` | LayerNorm | Transformer |
| `CloneBackward0` | `clone()` | 复制张量 |
| `CopyBackwards` | 拷贝 | 张量复制 |
| `ToCopyBackward0` | `.to()` | 设备 / dtype 转换 |
| `DetachBackward0` | `detach()` | 断开梯度 |
| `AccumulateGrad` | 梯度累积 | 叶子参数 `.grad` |

---

## 3. 一眼识别小技巧

### 3.1 先看前半部分

- `AddmmBackward0` → 先认 `Addmm`
- `ReluBackward0` → 先认 `Relu`
- `PowBackward0` → 先认 `Pow`

### 3.2 常见代码对应

```python
x ** 2        # PowBackward0
x @ W         # MatmulBackward0 / MmBackward0
torch.relu(x) # ReluBackward0
torch.softmax(x, dim=0) # SoftmaxBackward0
x.view(...)   # ViewBackward0
nn.Linear(...)# AddmmBackward0
```

### 3.3 叶子参数通常不显示 `grad_fn`

```python
w = torch.tensor([2.0], requires_grad=True)
print(w.grad_fn)  # None
```

---

## 4. 典型模型里会看到什么？

### MLP
```text
Linear -> ReLU -> Linear -> ReLU -> Linear
```
常见：
- `AddmmBackward0`
- `ReluBackward0`
- `AddmmBackward0`
- `ReluBackward0`
- `AddmmBackward0`

### 分类模型
```text
Linear -> logits -> CrossEntropyLoss
```
常见：
- `AddmmBackward0`
- `CrossEntropyLossBackward0`

### Transformer / GPT
常见组合：
- `AddmmBackward0`
- `ViewBackward0`
- `TransposeBackward0`
- `BmmBackward0`
- `SoftmaxBackward0`
- `DropoutBackward0`
- `NativeLayerNormBackward0`
- `CatBackward0`
- `MaskedFillBackward0`

---

## 5. 迷你示例

```python
import torch
import torch.nn as nn

model = nn.Linear(4, 2)
x = torch.randn(1, 4)
out = model(x)
print(out.grad_fn)  # 常见 <AddmmBackward0>
```

---

## 6. 最后记住这几个就够快查

- `AddmmBackward0`：线性层
- `MulBackward0`：乘法
- `PowBackward0`：幂运算
- `ReluBackward0`：ReLU
- `SigmoidBackward0`：Sigmoid
- `SoftmaxBackward0`：softmax
- `ViewBackward0`：形状变换
- `ConvolutionBackward0`：卷积
- `AccumulateGrad`：参数梯度累积

---

## 7. 一句话总结

`grad_fn` 就是：

> **“这个张量是怎么算出来的，以及梯度怎么回传”的记录。**

