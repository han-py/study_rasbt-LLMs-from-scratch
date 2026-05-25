# 📌 PyTorch 中常见 `grad_fn` 全量速查与深入说明
> **核心结论：**
> `grad_fn` 记录的是：**当前张量是由哪一步可求导运算得到的**，以及 **反向传播时梯度应该如何沿着计算图回传**。
>
> 例如：
>
> ```python
> tensor(..., grad_fn=<AddmmBackward0>)
> ```
>
> 这说明这个张量不是“凭空出现”的，而是由某个算子经过前向计算生成的。
---
## 目录
1. `grad_fn`、`is_leaf`、`requires_grad` 的关系
2. 计算图与 `next_functions`
3. `grad_fn` 命名规律与版本差异
4. 常见 `grad_fn` 全量分类表
5. 典型网络中的 `grad_fn` 组合模式
6. 如何调试计算图
7. 自定义 `autograd.Function` 的 `grad_fn`
8. 练习题与参考答案
---
## 1. `grad_fn`、`is_leaf`、`requires_grad` 的关系
### 1.1 `requires_grad`
`requires_grad=True` 表示：
> 这个张量需要参与梯度计算。
它通常用于：
- 模型参数
- 需要求导的输入
- 中间可训练变量
### 1.2 `grad_fn`
当一个张量是通过可求导运算得到时，它会带上 `grad_fn`。
```python
import torch
x = torch.tensor(2.0, requires_grad=True)
y = x * 3
print(y.grad_fn)
```
`y` 是通过乘法得到的，所以它有 `grad_fn`。
### 1.3 `is_leaf`
- **leaf tensor**：由用户直接创建、通常是图的起点。
- **non-leaf tensor**：由运算得到的中间结果。
```python
import torch
x = torch.tensor(2.0, requires_grad=True)
y = x * 3
print(x.is_leaf)  # True
print(y.is_leaf)  # False
print(x.grad_fn)  # None
print(y.grad_fn)  # <MulBackward0>
```
### 1.4 `.grad` 存在哪儿？
- 叶子张量在反向传播后，梯度通常累积到 `.grad`
- 非叶子张量默认不保留 `.grad`，除非你手动 `retain_grad()`
```python
import torch
x = torch.tensor(2.0, requires_grad=True)
y = x * 3
y.retain_grad()
loss = y ** 2
loss.backward()
print(x.grad)  # 36
print(y.grad)  # 12
```
---
## 2. 计算图与 `next_functions`
`grad_fn` 不只是一个名字，它还是计算图中的一个节点。
### 2.1 查看 `next_functions`
```python
import torch
x = torch.tensor(2.0, requires_grad=True)
y = x * 3
z = y + 4
loss = z ** 2
print(loss.grad_fn)
print(loss.grad_fn.next_functions)
```
你会看到当前节点往前连接的上游节点。
### 2.2 一个简单的链条
```text
x --MulBackward0--> y --AddBackward0--> z --PowBackward0--> loss
```
### 2.3 `AccumulateGrad`
叶子参数的梯度回收通常会通过 `AccumulateGrad` 节点完成。
```python
import torch
w = torch.tensor([2.0], requires_grad=True)
y = w * 5
print(y.grad_fn.next_functions)
```
其中通常会包含指向叶子参数的 `AccumulateGrad` 相关信息。
---
## 3. `grad_fn` 命名规律与版本差异
### 3.1 命名规律
一般格式是：
```text
<算子名>Backward0
```
例如：
- `AddmmBackward0`
- `ReluBackward0`
- `PowBackward0`
- `SoftmaxBackward0`
### 3.2 为什么有时不一样？
不同 PyTorch 版本、不同设备、不同后端实现，显示名可能略有差异，例如：
- CPU / CUDA 可能不同
- `reshape()` 有时表现得像 `ViewBackward0`
- 某些融合算子会显示更具体的后端名
### 3.3 记忆原则
不要只背“名字”，要把名字和“前向算子”对应起来。
---
## 4. 常见 `grad_fn` 全量分类表
> 说明：PyTorch 可见的 `grad_fn` 非常多，下表覆盖的是**训练、调试、阅读模型代码时最常见的类别**，并进一步补充了更底层、更完整的相关节点。
---
### 4.1 线性代数 / 矩阵运算类
| `grad_fn` | 对应操作 | 典型公式 | 常见场景 |
|---|---|---|---|
| `AddmmBackward0` | `addmm` | `bias + X @ W` | `nn.Linear` |
| `MmBackward0` | `mm` | `A @ B` | 二维矩阵乘法 |
| `MatmulBackward0` | `matmul` | 通用矩阵乘法 | `x @ W` |
| `BmmBackward0` | `bmm` | batch 矩阵乘法 | 注意力、多头 |
| `AddmvBackward0` | `addmv` | `bias + A @ v` | 线性代数 |
| `MvBackward0` | `mv` | `A @ v` | 矩阵向量乘法 |
| `DotBackward0` | `dot` | `a · b` | 向量内积 |
| `ChainMatmulBackward0` | 连乘 | 多个矩阵链式相乘 | 线性变换链 |
| `EigBackward0` / `LinalgEighBackward0` | 特征分解 | 矩阵特征值/特征向量 | 数值线代 |
| `SvdBackward0` / `LinalgSvdBackward0` | 奇异值分解 | `A = UΣVᵀ` | 数值线代 |
#### 例子
```python
import torch
x = torch.randn(2, 3, requires_grad=True)
w = torch.randn(3, 4, requires_grad=True)
out = x @ w
print(out.grad_fn)
```
---
### 4.2 元素级算术运算类
| `grad_fn` | 对应操作 | 典型含义 | 说明 |
|---|---|---|---|
| `AddBackward0` | 加法 | `a + b` | 逐元素相加 |
| `SubBackward0` | 减法 | `a - b` | 逐元素相减 |
| `MulBackward0` | 乘法 | `a * b` | 逐元素相乘 |
| `DivBackward0` | 除法 | `a / b` | 逐元素相除 |
| `PowBackward0` | 幂运算 | `x ** p` | 平方、立方等 |
| `NegBackward0` | 取负 | `-x` | 符号翻转 |
| `RsubBackward0` | 反向减法 | `b - a` | 常见于广播表达式 |
| `ExpBackward0` | 指数 | `e^x` | softmax、概率模型 |
| `LogBackward0` | 对数 | `log(x)` | 损失函数、概率 |
| `Log1pBackward0` | `log(1+x)` | 数值稳定对数 | 稳定计算 |
| `SqrtBackward0` | 平方根 | `sqrt(x)` | 归一化、几何 |
| `RsqrtBackward0` | 倒数平方根 | `1/sqrt(x)` | LayerNorm、BatchNorm |
| `ReciprocalBackward0` | 倒数 | `1/x` | 数值变换 |
| `SinBackward0` | 正弦 | `sin(x)` | 数学函数 |
| `CosBackward0` | 余弦 | `cos(x)` | 数学函数 |
| `TanBackward0` | 正切 | `tan(x)` | 数学函数 |
| `FloorBackward0` | 向下取整 | `floor(x)` | 离散化 |
| `CeilBackward0` | 向上取整 | `ceil(x)` | 离散化 |
| `RoundBackward0` | 四舍五入 | `round(x)` | 离散化 |
| `ClampBackward0` | 截断 | `clamp(min,max)` | 数值稳定 |
| `AbsBackward0` | 绝对值 | `abs(x)` | 稀疏/稳健损失 |
#### 例子
```python
import torch
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
print(y.grad_fn)
```
---
### 4.3 激活函数类
| `grad_fn` | 对应操作 | 典型用途 | 说明 |
|---|---|---|---|
| `ReluBackward0` | ReLU | 隐藏层 | 最常见激活 |
| `LeakyReluBackward0` | LeakyReLU | 隐藏层 | 负半轴保留梯度 |
| `ThresholdBackward0` | threshold | 阈值激活 | 早期网络 |
| `SigmoidBackward0` | sigmoid | 二分类/门控 | 压到 0~1 |
| `TanhBackward0` | tanh | 门控/RNN | 压到 -1~1 |
| `GeluBackward0` | GELU | Transformer | 平滑激活 |
| `SiluBackward0` / `SwishBackward0` | SiLU/Swish | 现代网络 | 平滑门控 |
| `EluBackward0` | ELU | 深层网络 | 指数型激活 |
| `SeluBackward0` | SELU | 自归一化网络 | 统计稳定 |
| `SoftplusBackward0` | Softplus | 平滑 ReLU | 数学更平滑 |
| `HardtanhBackward0` | Hardtanh | 截断激活 | 限幅 |
| `HardsigmoidBackward0` | Hardsigmoid | 近似 sigmoid | 低成本近似 |
| `HardswishBackward0` | Hardswish | 近似 swish | 轻量网络 |
| `RreluWithNoiseBackward0` | RReLU | 随机化激活 | 训练正则 |
#### 例子
```python
import torch
x = torch.tensor([-1.0, 0.5], requires_grad=True)
y = torch.relu(x)
print(y.grad_fn)
```
---
### 4.4 归一化 / 概率变换类
| `grad_fn` | 对应操作 | 典型用途 | 说明 |
|---|---|---|---|
| `SoftmaxBackward0` | softmax | 概率分布 | 分类、注意力 |
| `LogSoftmaxBackward0` | log-softmax | 对数概率 | 交叉熵 |
| `NativeBatchNormBackward0` | BatchNorm | 训练稳定 | CNN、MLP |
| `NativeLayerNormBackward0` | LayerNorm | 训练稳定 | Transformer |
| `NormalizeBackward0` | normalize | 特征归一化 | 表示学习 |
| `NormBackward0` | norm | 向量/矩阵范数 | 正则化 |
| `LinalgVectorNormBackward0` | 向量范数 | 长度计算 | 数学运算 |
| `LinalgMatrixNormBackward0` | 矩阵范数 | 矩阵尺度 | 数学运算 |
| `FrobeniusNormBackward0` | Frobenius 范数 | 矩阵长度 | 线代 |
| `RmsNormBackward0` | RMSNorm | Transformer 稳定化 | 现代大模型 |
#### 例子
```python
import torch
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = torch.softmax(x, dim=0)
print(y.grad_fn)
```
---
### 4.5 损失函数类
| `grad_fn` | 对应操作 | 典型用途 | 说明 |
|---|---|---|---|
| `MseLossBackward0` | MSE | 回归 | 均方误差 |
| `L1LossBackward0` | L1 | 回归 | 绝对误差 |
| `SmoothL1LossBackward0` | Smooth L1 | 回归/检测 | Huber 风格 |
| `HuberLossBackward0` | Huber | 回归/检测 | 稳健损失 |
| `BinaryCrossEntropyBackward0` | BCE | 二分类 | 概率输出 |
| `BinaryCrossEntropyWithLogitsBackward0` | BCE with logits | 二分类 | 数值稳定 |
| `NllLossBackward0` | NLL | 分类 | 对数似然 |
| `CrossEntropyLossBackward0` | CE | 多分类 | 组合损失 |
| `KLDivBackward0` | KL 散度 | 分布对齐 | 蒸馏、概率模型 |
| `PoissonNLLLossBackward0` | Poisson NLL | 计数建模 | 统计模型 |
| `MarginRankingLossBackward0` | 排序损失 | ranking | 检索/排序 |
| `TripletMarginLossBackward0` | 三元组损失 | 表示学习 | 度量学习 |
#### 例子
```python
import torch
import torch.nn.functional as F
pred = torch.tensor([0.8], requires_grad=True)
target = torch.tensor([1.0])
loss = F.binary_cross_entropy(pred, target)
print(loss.grad_fn)
```
---
### 4.6 形状、视图与广播类
| `grad_fn` | 对应操作 | 典型用途 | 说明 |
|---|---|---|---|
| `ViewBackward0` | `view()` | 改变视图 | 共享内存视图 |
| `ReshapeAliasBackward0` | `reshape()` | 改变形状 | 可能返回视图 |
| `TransposeBackward0` | `transpose()` | 交换维度 | 矩阵、注意力 |
| `TBackward0` | `.t()` | 二维转置 | 线代 |
| `PermuteBackward0` | `permute()` | 维度重排 | 图像/NLP |
| `SqueezeBackward0` | `squeeze()` | 删除 1 维 | 去掉 batch 维 |
| `UnsqueezeBackward0` | `unsqueeze()` | 增加 1 维 | 对齐维度 |
| `FlattenBackward0` | `flatten()` | 压平 | MLP 输入 |
| `ExpandBackward0` | `expand()` | 广播视图 | 扩维 |
| `RepeatBackward0` | `repeat()` | 重复复制 | 展开张量 |
| `UnfoldBackward0` | `unfold()` | 滑动窗口展开 | 卷积、patch |
| `FoldBackward0` | `fold()` | 反向折叠 | 图像重建 |
| `AsStridedBackward0` | `as_strided()` | 任意视图 | 底层高级操作 |
| `BroadcastBackward0` | 广播相关 | 自动扩展 | 形状对齐 |
#### 例子
```python
import torch
x = torch.randn(2, 3, requires_grad=True)
y = x.view(3, 2)
print(y.grad_fn)
```
---
### 4.7 拼接、拆分、索引与选择类
| `grad_fn` | 对应操作 | 典型用途 | 说明 |
|---|---|---|---|
| `CatBackward0` | `cat` | 拼接 | 特征融合 |
| `StackBackward0` | `stack` | 堆叠 | 批量收集 |
| `SplitBackward0` | `split` | 拆分 | 分块 |
| `ChunkBackward0` | `chunk` | 分块 | 分块 |
| `SelectBackward0` | `select` | 选择单个位置 | 索引 |
| `SliceBackward0` | 切片 | 截取区间 | 索引 |
| `IndexBackward0` | 索引 | 高级索引 | 张量切片 |
| `IndexSelectBackward0` | index_select | 按索引选择 | 向量/词表 |
| `GatherBackward0` | gather | 按索引收集 | 注意力、检索 |
| `ScatterBackward0` | scatter | 按索引散射 | 反向索引写入 |
| `ScatterAddBackward0` | scatter_add | 按索引累加 | 图神经网络 |
| `MaskedSelectBackward0` | masked_select | 按掩码选择 | 稀疏选择 |
| `TakeBackward0` | take | 拉平后取值 | 索引选择 |
| `PutBackward0` | put | 按位置写入 | 稀疏更新 |
| `EmbeddingBackward0` | embedding lookup | 查表 | NLP 词向量 |
| `EmbeddingDenseBackward0` | 稠密 embedding | embedding 变体 | NLP/推荐 |
#### 例子
```python
import torch
x = torch.arange(6.0, requires_grad=True)
y = x[2:5]
print(y.grad_fn)
```
---
### 4.8 卷积、池化与视觉算子类
| `grad_fn` | 对应操作 | 典型用途 | 说明 |
|---|---|---|---|
| `ConvolutionBackward0` | 卷积 | CNN 主干 | 卷积反传 |
| `CudnnConvolutionBackward0` | cuDNN 卷积 | GPU 卷积 | 后端实现 |
| `ThnnConv2DBackward0` | 旧式卷积实现 | 历史实现 | 版本相关 |
| `MaxPool2DWithIndicesBackward0` | 最大池化 | 视觉下采样 | 带索引回传 |
| `AvgPool2DBackward0` | 平均池化 | 下采样 | 平均池化 |
| `AdaptiveAvgPool2DBackward0` | 自适应平均池化 | 固定输出大小 | CNN 头部 |
| `AdaptiveMaxPool2DBackward0` | 自适应最大池化 | 固定输出大小 | CNN 头部 |
| `MaxUnpool2DBackward0` | 反池化 | 还原空间 | 解码器 |
| `PixelShuffleBackward0` | 像素重排 | 超分辨率 | 维度重排 |
| `ReflectionPad2DBackward0` | 反射填充 | CNN padding | 图像边界 |
| `ReplicationPad2DBackward0` | 复制填充 | CNN padding | 图像边界 |
#### 例子
```python
import torch
import torch.nn as nn
conv = nn.Conv2d(3, 8, 3)
x = torch.randn(1, 3, 32, 32)
y = conv(x)
print(y.grad_fn)
```
---
### 4.9 序列、RNN、Transformer 相关类
很多序列模型不会只显示一个单独的“TransformerBackward”，而是由多个基础算子组合而成。
| `grad_fn` | 对应操作 | 典型用途 | 说明 |
|---|---|---|---|
| `EmbeddingBackward0` | embedding lookup | 词嵌入 | NLP |
| `BmmBackward0` | batch matmul | 注意力中的批量矩阵乘法 | 注意力 |
| `SoftmaxBackward0` | softmax | 注意力权重 | 注意力 |
| `DropoutBackward0` | dropout | 随机失活 | 正则化 |
| `NativeLayerNormBackward0` | layer norm | Transformer 归一化 | 编码器/解码器 |
| `PermuteBackward0` | 维度换位 | `batch, seq, hidden` 转换 | 形状调整 |
| `TransposeBackward0` | 转置 | Q/K/V 重排 | 注意力 |
| `ViewBackward0` | reshape | 头数拆分 | 多头注意力 |
| `CatBackward0` | 拼接 | 多头合并 | 多头注意力 |
| `AddmmBackward0` | 线性投影 | Q/K/V 投影、输出投影 | Transformer |
| `MaskedFillBackward0` | masked_fill | 因果掩码 | causal mask |
| `WhereBackward0` | where | 条件选择 | 掩码逻辑 |
| `MulBackward0` | 乘法 | attention score scaling | 缩放点积 |
#### 典型注意力链条
```text
Linear -> View/Transpose -> Bmm -> Div/Mul -> MaskedFill/Where -> Softmax -> Dropout -> Bmm -> Linear
```
---
### 4.10 类型转换、复制、设备与存储类
| `grad_fn` | 对应操作 | 典型用途 | 说明 |
|---|---|---|---|
| `CloneBackward0` | clone | 复制张量 | 保留原值 |
| `CopyBackwards` | copy | 复制 | 张量拷贝 |
| `ToCopyBackward0` | `.to(...)` | 设备/类型转换 | CPU/GPU/dtype |
| `DetachBackward0` | detach | 断开图 | 停止回传 |
| `CopySlices` | 切片复制 | 局部写入 | 索引赋值 |
| `DetachBackward0` | detach | 切断梯度 | 图截断 |
#### 例子
```python
import torch
x = torch.tensor([1.0], requires_grad=True)
y = x.clone()
print(y.grad_fn)
```
---
### 4.11 梯度累计、Hook 与图边界类
| `grad_fn` | 对应内容 | 说明 |
|---|---|---|
| `AccumulateGrad` | 梯度累积 | 叶子张量把梯度写入 `.grad` |
| `BackwardHookFunctionBackward` | 反向 hook | 与 hook/回调相关 |
| `CheckpointFunctionBackward` | checkpoint | 激活重计算 |
| `AutogradMeta` 相关节点 | 元信息 | 不是直接显示的用户节点，但影响行为 |
### 4.12 自定义 `autograd.Function`
如果你自己写了自定义函数，`grad_fn` 往往会显示成你定义的名字加 `Backward`。
```python
import torch
class MySquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * x
    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * 2 * x
x = torch.tensor(3.0, requires_grad=True)
y = MySquare.apply(x)
print(y.grad_fn)
```
可能看到类似：
```python
<torch.autograd.function.MySquareBackward>
```
---
## 5. 典型网络中的 `grad_fn` 组合模式
### 5.1 多层感知机（MLP）
```text
Linear -> ReLU -> Linear -> ReLU -> Linear
```
可能对应：
- `AddmmBackward0`
- `ReluBackward0`
- `AddmmBackward0`
- `ReluBackward0`
- `AddmmBackward0`
### 5.2 分类模型
```text
Linear -> logits -> CrossEntropyLoss
```
常见节点：
- `AddmmBackward0`
- `CrossEntropyLossBackward0`
- 内部常包含 `LogSoftmaxBackward0` / `NllLossBackward0`
### 5.3 Transformer / GPT
典型会看到：
- `AddmmBackward0`
- `ViewBackward0`
- `TransposeBackward0`
- `BmmBackward0`
- `SoftmaxBackward0`
- `DropoutBackward0`
- `NativeLayerNormBackward0`
- `CatBackward0`
- `MulBackward0`
- `MaskedFillBackward0`
---
## 6. 如何调试计算图
### 6.1 打印 `grad_fn`
```python
print(tensor.grad_fn)
```
### 6.2 看上游节点
```python
print(tensor.grad_fn.next_functions)
```
### 6.3 看叶子标记
```python
print(tensor.is_leaf)
print(tensor.requires_grad)
```
### 6.4 保留非叶子梯度
```python
intermediate.retain_grad()
```
### 6.5 追踪多次反向
```python
loss.backward(retain_graph=True)
```
### 6.6 用 `autograd.grad`
```python
from torch.autograd import grad
g = grad(loss, [w, b], retain_graph=True)
print(g)
```
---
## 7. 梯度图中的几个关键概念
### 7.1 `retain_graph=True`
当你还要对同一张计算图再次求导时，需要保留图。
### 7.2 `create_graph=True`
如果你要对“梯度再求导”，要构建高阶导数图。
```python
import torch
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3
first_grad = torch.autograd.grad(y, x, create_graph=True)[0]
second_grad = torch.autograd.grad(first_grad, x)[0]
print(first_grad, second_grad)
```
### 7.3 `detach()` 与 `no_grad()`
- `detach()`：切断当前张量的梯度链
- `torch.no_grad()`：整个上下文不记录图
---
## 8. 综合示例：从 `grad_fn` 读出模型结构
```python
import torch
import torch.nn as nn
model = nn.Sequential(
    nn.Linear(50, 30),
    nn.ReLU(),
    nn.Linear(30, 20),
    nn.ReLU(),
    nn.Linear(20, 3)
)
x = torch.rand(1, 50)
out = model(x)
print(out)
print(out.grad_fn)
```
你可以据此推断：
- 最后一层是线性层
- 所以输出常见 `grad_fn` 是 `AddmmBackward0`
- 前面可能还有 `ReluBackward0`、其他 `AddmmBackward0`
---
## 9. `grad_fn` 速查总表（浓缩版）
| 类别 | 常见 `grad_fn` |
|---|---|
| 线性代数 | `AddmmBackward0`、`MmBackward0`、`MatmulBackward0`、`BmmBackward0` |
| 元素运算 | `AddBackward0`、`SubBackward0`、`MulBackward0`、`DivBackward0`、`PowBackward0` |
| 激活函数 | `ReluBackward0`、`SigmoidBackward0`、`TanhBackward0`、`GeluBackward0`、`SiluBackward0` |
| 归一化/概率 | `SoftmaxBackward0`、`LogSoftmaxBackward0`、`NativeBatchNormBackward0`、`NativeLayerNormBackward0` |
| 损失函数 | `MseLossBackward0`、`L1LossBackward0`、`BinaryCrossEntropyBackward0`、`CrossEntropyLossBackward0`、`KLDivBackward0` |
| 形状变换 | `ViewBackward0`、`ReshapeAliasBackward0`、`TransposeBackward0`、`PermuteBackward0`、`ExpandBackward0` |
| 索引/拼接 | `SelectBackward0`、`SliceBackward0`、`IndexBackward0`、`GatherBackward0`、`CatBackward0`、`StackBackward0` |
| 卷积/池化 | `ConvolutionBackward0`、`MaxPool2DWithIndicesBackward0`、`AvgPool2DBackward0` |
| 设备/复制 | `CloneBackward0`、`CopyBackwards`、`ToCopyBackward0`、`DetachBackward0` |
| 参数梯度 | `AccumulateGrad` |
---
## 10. 练习题
### 练习 1：预测 `grad_fn`
```python
import torch
x = torch.tensor(2.0, requires_grad=True)
y = x * 4 + 1
print(y.grad_fn)
```
### 练习 2：线性层输出
```python
import torch
import torch.nn as nn
layer = nn.Linear(4, 2)
x = torch.randn(1, 4)
out = layer(x)
print(out.grad_fn)
```
### 练习 3：归一化
```python
import torch
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = torch.softmax(x, dim=0)
print(y.grad_fn)
```
### 练习 4：reshape / view
```python
import torch
x = torch.randn(2, 3, requires_grad=True)
y = x.reshape(3, 2)
print(y.grad_fn)
```
### 练习 5：索引操作
```python
import torch
x = torch.arange(10.0, requires_grad=True)
y = x[2:7]
print(y.grad_fn)
```
### 练习 6：卷积
```python
import torch
import torch.nn as nn
conv = nn.Conv2d(3, 8, 3)
x = torch.randn(1, 3, 32, 32)
y = conv(x)
print(y.grad_fn)
```
---
## 11. 参考答案
<details>
<summary>点击展开答案</summary>
### 练习 1
常见是：
```python
<AddBackward0>
```
### 练习 2
通常是：
```python
<AddmmBackward0>
```
### 练习 3
通常是：
```python
<SoftmaxBackward0>
```
### 练习 4
通常是：
```python
<ViewBackward0>
```
### 练习 5
通常是：
```python
<SliceBackward0>
```
### 练习 6
通常是：
```python
<ConvolutionBackward0>
```
</details>
---
## 12. 小结
你可以把 `grad_fn` 理解成：
> **“这个张量从哪里来、怎么回传梯度、图上游连接了谁”** 的记录。
在实际学习和调试里，最有价值的是把它和前向代码一一对应起来：
- `nn.Linear` → 常见 `AddmmBackward0`
- `relu` → `ReluBackward0`
- `sigmoid` → `SigmoidBackward0`
- `softmax` → `SoftmaxBackward0`
- `x ** 2` → `PowBackward0`
- `view / reshape` → `ViewBackward0` / `ReshapeAliasBackward0`
- `cat / stack` → `CatBackward0` / `StackBackward0`
- `conv2d` → `ConvolutionBackward0`
- 参数梯度累积 → `AccumulateGrad`
如果你把这些节点和实际代码结合起来看，后面读复杂模型、看计算图、定位梯度问题会非常顺手。
