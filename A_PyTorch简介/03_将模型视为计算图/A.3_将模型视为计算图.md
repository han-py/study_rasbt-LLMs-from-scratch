# 🧠 PyTorch 中的“将模型视为计算图”完全指南

> **一句话先理解：**
> 在 PyTorch 里，模型不是一坨“黑盒代码”，而是一个由很多**运算节点**连接起来的**计算图（Computational Graph）**。
> 
> 你可以把它想象成：
> - **节点**：每一步运算，比如加法、乘法、矩阵乘法、激活函数。
> - **边**：数据在这些运算之间流动的路径。
> - **方向**：数据通常从输入流向输出，训练时梯度则反向传播回来。
>
> 这也是为什么 PyTorch 能很自然地支持 **自动求导（autograd）**：因为它会在你做运算的时候，自动帮你记录“这一步是怎么来的”。

---

## 📚 目录

- [一、什么是计算图？](#一什么是计算图)
- [二、为什么要把模型看成计算图？](#二为什么要把模型看成计算图)
- [三、PyTorch 是怎么“记录”计算图的？](#三pytorch-是怎么记录计算图的)
- [四、计算图中的 `requires_grad` 是什么？](#四计算图中的-requires_grad-是什么)
- [五、计算图里的叶子节点、梯度和 `.grad`](#五计算图里的叶子节点梯度和-grad)
- [六、前向传播：计算图是怎么往前算的？](#六前向传播计算图是怎么往前算的)
- [七、反向传播：梯度是怎么回来的？](#七反向传播梯度是怎么回来的)
- [八、把模型拆成计算图来理解](#八把模型拆成计算图来理解)
- [九、`.detach()`：如何从计算图中“切断”张量](#九detach如何从计算图中切断张量)
- [十、`no_grad()`：推理时为什么常用它？](#十no_grad推理时为什么常用它)
- [十一、一个完整的小例子：从前向到反向](#十一一个完整的小例子从前向到反向)
- [十二、为什么“模型 = 计算图”这个理解很重要？](#十二为什么模型--计算图这个理解很重要)
- [十三、练习题：检查你是否真正理解了计算图](#十三练习题检查你是否真正理解了计算图)
- [参考答案](#参考答案)
- [小结](#小结)

---

## 🌟 一、什么是计算图？

如果把神经网络看成一个“工厂流水线”，那么：

- **输入数据** 就像原材料
- **每一层神经网络** 就像流水线中的机器
- **每个运算** 就像机器内部的一道加工工序
- **最终输出** 就像成品

计算图就是这条流水线的“地图”。它记录了：
1. 哪些输入参与了运算
2. 运算是怎么连接的
3. 输出是如何一步步算出来的
4. 反向传播时梯度应该怎么回传

### 一个非常小的例子

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

z = x * y + y
print(z)
```

这里的运算顺序是：
- 先算 `x * y`
- 再加上 `y`
- 最后得到 `z`

如果从计算图角度看，它其实长这样：

```text
x ──┐
    ├─(*)──┐
 y ──┘     │
           (+)── z
 y ────────┘
```

也就是说，`z` 不是凭空出现的，它是由前面的运算一步一步构建出来的。

---

## 🌟 二、为什么要把模型看成计算图？

把模型看成计算图，有三个非常重要的好处：

### 1. 方便自动求导
训练神经网络时，我们需要知道：
- 参数应该往哪个方向调整
- 调整多少才会让损失更小

PyTorch 会利用计算图，自动计算梯度。

### 2. 方便理解前向传播和反向传播
- **前向传播**：数据从输入流向输出，得到预测结果
- **反向传播**：梯度从输出倒着传回去，更新参数

### 3. 方便调试复杂模型
如果某个地方输出不对，你可以顺着计算图往回查：
- 是不是形状不匹配？
- 是不是数据类型不对？
- 是不是某一步把梯度断掉了？

---

## 🌟 三、PyTorch 是怎么“记录”计算图的？

PyTorch 的一个核心特点是：**动态图（Dynamic Graph）**。

意思是：
- 你在运行代码的时候，PyTorch 才临时构建计算图
- 你每运行一次，图就重新构建一次
- 这让模型更加灵活，特别适合研究和调试

### 示例：动态图的感觉

```python
import torch

x = torch.tensor(2.0, requires_grad=True)

if x.item() > 1:
    y = x * 2
else:
    y = x + 2

print(y)
```

这里的计算路径会随着条件变化而变化。也就是说，PyTorch 不要求你提前把整个流程写死，而是可以根据代码执行过程动态构建图。

---

## 🌟 四、计算图中的 `requires_grad` 是什么？

`requires_grad=True` 的意思是：

> “请 PyTorch 记住这个张量参与了哪些运算，因为我后面可能要对它求导。”

### 示例

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
print(y)
```

如果我们对 `y` 调用反向传播：

```python
y.backward()
print(x.grad)
```

那么 PyTorch 会自动帮我们算出：

- `y = x^2`
- 所以 `dy/dx = 2x`
- 当 `x = 2` 时，梯度就是 `4`

这说明 PyTorch 不只是会算结果，还会帮你算“结果对输入有多敏感”。

---

## 🌟 五、计算图里的叶子节点、梯度和 `.grad`

### 1. 什么是叶子节点？

在 PyTorch 中，通常我们自己手动创建、并且 `requires_grad=True` 的张量，常被看作计算图的“叶子节点”。

```python
x = torch.tensor(3.0, requires_grad=True)
```

这里的 `x` 就是一个很重要的起点。

### 2. `.grad` 是什么？

当你执行完 `backward()` 之后，PyTorch 会把梯度保存在 `.grad` 里：

```python
import torch

x = torch.tensor(3.0, requires_grad=True)
y = x * x + 2 * x

# 反向传播

y.backward()

print(x.grad)
```

这里：
- `y = x^2 + 2x`
- `dy/dx = 2x + 2`
- 当 `x = 3` 时，梯度就是 `8`

所以你会看到 `x.grad` 大约是 `8`

### 3. 为什么梯度会累积？

PyTorch 默认不会自动清空梯度，而是会“累加”。

```python
x = torch.tensor(2.0, requires_grad=True)
y1 = x * 2
y1.backward()
print(x.grad)  # 第一次梯度

y2 = x * 3
y2.backward()
print(x.grad)  # 梯度会累加
```

所以在训练神经网络时，一般每轮更新前都会先：

```python
optimizer.zero_grad()
```

这一步就是把旧梯度清空，避免越加越多。

---

## 🌟 六、前向传播：计算图是怎么往前算的？

前向传播就是从输入到输出的计算过程。

### 一个简单的神经网络风格例子

```python
import torch

x = torch.tensor([1.0, 2.0], requires_grad=True)
W = torch.tensor([[0.1, 0.2],
                  [0.3, 0.4]], requires_grad=True)
b = torch.tensor([0.5, 0.6], requires_grad=True)

# 线性变换：x @ W + b
z = x @ W + b
print(z)
```

这里的计算图可以理解为：

```text
x ──┐
    ├─ matmul ──┐
W ──┘           │
                (+) ── z
b ──────────────┘
```

这就是神经网络里最常见的基本结构之一：
- 输入
- 矩阵乘法
- 加偏置
- 输出

---

## 🌟 七、反向传播：梯度是怎么回来的？

反向传播的核心目标是：

> 计算“损失函数对每个参数的影响程度”

### 一个最简单的例子

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x ** 3

# dy/dx = 3x^2 = 12

y.backward()
print(x.grad)
```

这表示：
- 当 `x` 发生微小变化时，`y` 会变化多少
- 梯度越大，说明这个参数对结果影响越大

### 为什么训练需要梯度？

因为我们要“沿着让损失下降的方向”更新参数。
这就是优化器（比如 SGD、Adam）做的事情。

---

## 🌟 八、把模型拆成计算图来理解

假设一个模型做的事情很简单：

```python
import torch

x = torch.tensor(4.0, requires_grad=True)

# 模型：先乘 2，再加 1，再平方
h = x * 2 + 1
out = h ** 2

print(out)
```

从计算图角度看：

```text
x ──(*)2──┐
          (+)1── h ──(**2)── out
```

如果你想知道 `out` 对 `x` 的梯度：

```python
out.backward()
print(x.grad)
```

PyTorch 会自动沿着图反向计算：
- `out = (2x + 1)^2`
- 先求外层导数，再乘以内层导数
- 不需要你手算链式法则，PyTorch 会帮你完成

---

## 🌟 九、`.detach()`：如何从计算图中“切断”张量

有时候我们不想让某个张量继续参与求导，可以使用 `.detach()`。

### 示例

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x * 3
z = y.detach()

print(z)
print(z.requires_grad)
```

`detach()` 之后：
- 这个张量的后续操作不会再被记录到原来的计算图里
- 常用于推理、日志记录、或者某些特殊训练技巧

### 小提醒

如果不小心把需要求导的路径 detach 掉了，梯度就会断掉，模型就学不动了。

---

## 🌟 十、`no_grad()`：推理时为什么常用它？

在做验证或者推理时，我们通常不需要梯度。
这时可以使用：

```python
with torch.no_grad():
    ...
```

### 示例

```python
import torch

x = torch.tensor(2.0, requires_grad=True)

with torch.no_grad():
    y = x * 10

print(y)
print(y.requires_grad)
```

它的作用是：
- 节省显存
- 加快推理速度
- 避免无意义地构建计算图

这在大模型推理中非常常见。

---

## 🌟 十一、一个完整的小例子：从前向到反向

下面我们做一个完整流程，看看计算图是如何工作的。

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(5.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# 前向传播
z = x * w + b
loss = (z - 11) ** 2

print("z =", z.item())
print("loss =", loss.item())

# 反向传播
loss.backward()

print("x.grad =", x.grad)
print("w.grad =", w.grad)
print("b.grad =", b.grad)
```

### 这段代码在做什么？

1. `x * w + b` 得到预测值 `z`
2. 把 `z` 和目标值 `11` 做差，得到误差
3. 对误差平方，得到损失 `loss`
4. 调用 `backward()` 之后，PyTorch 会自动算出每个参数的梯度

这就是训练神经网络最核心的流程。

---

## 🌟 十二、为什么“模型 = 计算图”这个理解很重要？

如果你以后要学习：
- 卷积神经网络（CNN）
- 循环神经网络（RNN）
- Transformer
- GPT
- 微调
- LoRA
- RLHF

你都会不断遇到一个共同点：

> **所有复杂模型，本质上都是由很多基础运算组合而成的计算图。**

你越早理解这个概念，后面学复杂模型就越轻松。

---

## 🎯 十三、练习题：检查你是否真正理解了计算图

### 练习 1：判断是否会求梯度

下面这段代码中，哪些张量会记录梯度，哪些不会？请先自己判断，再运行验证。

```python
import torch

a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0)
c = a * b
print(c)
```

**问题：**
1. `a.requires_grad` 是什么？
2. `b.requires_grad` 是什么？
3. `c.requires_grad` 是什么？
4. 为什么？

---

### 练习 2：手算梯度

```python
import torch

x = torch.tensor(4.0, requires_grad=True)
y = x ** 2 + 3 * x + 1
```

**问题：**
1. `y` 对 `x` 的导数是多少？
2. 当 `x = 4` 时，梯度是多少？
3. 调用 `y.backward()` 后，`x.grad` 应该是多少？

---

### 练习 3：理解 detach

```python
import torch

x = torch.tensor(5.0, requires_grad=True)
y = x * 2
z = y.detach() * 3
```

**问题：**
1. `z` 还会和 `x` 保持梯度关系吗？
2. 为什么？
3. 如果把 `detach()` 去掉，会发生什么变化？

---

### 练习 4：构造一个简单计算图

请自己写一个表达式，满足下面条件：
- 至少有两个可求导张量
- 至少包含一次乘法和一次加法
- 最后调用 `backward()`
- 打印每个张量的 `.grad`

**提示：** 可以尝试类似下面的结构：

```python
loss = (a * b + c) ** 2
```

---

## ✅ 参考答案

<details>
<summary>点击展开参考答案</summary>

### 练习 1 参考答案

```python
import torch

a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0)
c = a * b

print(a.requires_grad)  # True
print(b.requires_grad)  # False
print(c.requires_grad)  # True
```

**解释：**
- `a` 需要求导，所以会被记录
- `b` 不需要求导
- `c` 由 `a` 和 `b` 运算得到，只要其中一个需要求导，结果就会进入计算图

### 练习 2 参考答案

```python
import torch

x = torch.tensor(4.0, requires_grad=True)
y = x ** 2 + 3 * x + 1

y.backward()
print(x.grad)
```

手算：
- `dy/dx = 2x + 3`
- 当 `x = 4` 时，`dy/dx = 11`

所以 `x.grad` 应该是 `11`

### 练习 3 参考答案

```python
import torch

x = torch.tensor(5.0, requires_grad=True)
y = x * 2
z = y.detach() * 3

print(z.requires_grad)  # False
```

因为 `detach()` 把张量从原计算图里切断了，所以后面不会再向 `x` 回传梯度。

### 练习 4 参考答案

```python
import torch

a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)
c = torch.tensor(1.0, requires_grad=True)

loss = (a * b + c) ** 2
loss.backward()

print(a.grad)
print(b.grad)
print(c.grad)
```

</details>

---

## 🌈 小结

这一节最重要的概念是：

- PyTorch 会把模型中的运算自动组织成**计算图**
- 前向传播负责算结果
- 反向传播负责算梯度
- `requires_grad=True` 让张量参与求导
- `backward()` 会自动把梯度回传到图中的参数
- `detach()` 和 `no_grad()` 则是常见的“切断计算图”手段

如果你能把“模型 = 计算图”真正理解透，那么后面学习神经网络、优化器、Transformer 和 GPT 会轻松很多。

