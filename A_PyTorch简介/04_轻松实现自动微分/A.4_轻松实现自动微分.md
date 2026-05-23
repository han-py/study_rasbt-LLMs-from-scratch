# ⚡ PyTorch 中如何轻松实现自动微分（Autograd）——超详细入门指南

> **一句话理解自动微分：**
> 自动微分就是：**你写好前向计算，PyTorch 自动帮你算梯度**。
>
> 如果把训练神经网络比作“开车”，那么：
> - **前向传播** 是你看前面路况、正常行驶
> - **损失函数** 是你判断“现在开得好不好”
> - **自动微分** 是车上的导航系统，自动告诉你“往哪边打方向盘更合适”
> - **反向传播** 是把这个方向建议一层层传回去
>
> 也就是说，自动微分是深度学习训练中最核心、最省事、最神奇的功能之一。

---

## 🌟 一、为什么需要自动微分？

在深度学习里，我们经常要做一件事：

> **根据损失函数，计算每个参数的梯度，再更新参数。**

如果没有自动微分，我们就得：
- 自己手推导数公式
- 自己写每一层的梯度计算
- 自己处理链式法则
- 一旦网络变复杂，就很容易出错

而 PyTorch 的自动微分可以帮我们：
1. 自动记录运算过程
2. 自动构建计算图
3. 自动计算梯度
4. 自动把梯度保存在 `.grad` 中

这就是 PyTorch 训练神经网络时最重要的“省心神器”。

---

## 🌟 二、自动微分到底在做什么？

我们可以把它理解为：

### 1. 先做前向计算
比如：
```python
x -> y -> z -> loss
```

### 2. 再做反向传播
从 `loss` 开始，沿着计算图往回走：
```python
loss -> z -> y -> x
```

### 3. 计算梯度
PyTorch 会计算：
- `loss` 对 `x` 的影响
- `loss` 对 `y` 的影响
- `loss` 对每个参数的影响

这些影响程度就是梯度。

---

## 🌟 三、最基础的自动微分示例

下面我们来看一个最简单的例子。

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2

y.backward()
print("x.grad =", x.grad)
```

### 这段代码发生了什么？
- `x = 2.0`
- `y = x^2`
- 由于 `requires_grad=True`，PyTorch 会记录这个运算
- 调用 `y.backward()` 后，PyTorch 会自动计算导数

因为：
- `y = x^2`
- 所以 `dy/dx = 2x`
- 当 `x = 2` 时，梯度就是 `4`

所以输出结果应该接近：
```python
x.grad = tensor(4.)
```

---

## 🌟 四、`requires_grad=True` 是什么意思？

这个参数非常重要。

### 它的作用是：
告诉 PyTorch：

> “这个张量我要参与求导，请把它记进计算图。”

### 示例

```python
import torch

x = torch.tensor(3.0, requires_grad=True)
y = x * 5
print(y)
```

如果张量没有 `requires_grad=True`，PyTorch 就不会帮你记录梯度。

```python
import torch

x = torch.tensor(3.0)
y = x * 5
print(y)
```

虽然这也能算出结果，但它不会参与自动微分。

---

## 🌟 五、梯度是怎么来的？——链式法则

自动微分本质上依赖于数学里的**链式法则**。

比如：

```python
x -> y -> z -> loss
```

如果：
- `y = f(x)`
- `z = g(y)`
- `loss = h(z)`

那么：

```text
 dloss/dx = dloss/dz * dz/dy * dy/dx
```

PyTorch 会自动帮我们把这些链式关系拼起来，不需要我们手工推导。

---

## 🌟 六、一个稍微复杂一点的例子

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x * 3
z = y + 4
loss = z ** 2

loss.backward()
print("x.grad =", x.grad)
```

### 我们来手算一下：

- `y = 3x`
- `z = y + 4 = 3x + 4`
- `loss = z^2 = (3x + 4)^2`

对 `x` 求导：

```text
 dloss/dx = 2(3x + 4) * 3
```

当 `x = 2`：

```text
 dloss/dx = 2(3*2 + 4) * 3 = 2(10) * 3 = 60
```

所以 `x.grad` 应该是 `60`。

---

## 🌟 七、`backward()` 到底做了什么？

`backward()` 的意思是：

> 从当前张量开始，向后计算梯度。

### 注意：
它通常用于**标量张量**。

比如：

```python
loss.backward()
```

这里 `loss` 一般是一个单独的数值。

如果你的张量不是标量，通常就要先把它变成一个标量，比如：
- 求和 `sum()`
- 求平均 `mean()`

例如：

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x * 2
loss = y.sum()
loss.backward()

print(x.grad)
```

这里：
- `y = [2, 4, 6]`
- `loss = 12`
- `loss` 是标量，所以可以直接 `backward()`

---

## 🌟 八、`.grad` 是什么？

当你调用 `backward()` 后，梯度会被保存在张量的 `.grad` 属性里。

```python
import torch

x = torch.tensor(4.0, requires_grad=True)
y = x ** 3

y.backward()
print(x.grad)
```

这里：
- `y = x^3`
- `dy/dx = 3x^2`
- 当 `x = 4` 时，梯度是 `48`

所以 `x.grad` 应该是 `48`。

---

## 🌟 九、为什么梯度会累积？

PyTorch 默认不会自动清空梯度，而是会累加。

### 示例

```python
import torch

x = torch.tensor(2.0, requires_grad=True)

y1 = x * 2
y1.backward()
print("第一次梯度:", x.grad)

# 第二次反向传播之前不清空梯度
y2 = x * 3
y2.backward()
print("累积后的梯度:", x.grad)
```

### 为什么会这样？
因为在训练循环里，可能需要多次反向传播或累积梯度，所以 PyTorch 默认采用累加方式。

### 所以训练时通常要先清空梯度：
```python
optimizer.zero_grad()
```

或者：
```python
x.grad.zero_()
```

---

## 🌟 十、如何断开计算图？——`detach()`

有时候我们不希望某个张量继续参与求导，可以使用：

```python
.detach()
```

### 示例

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x * 3
z = y.detach()

print(z)
print(z.requires_grad)
```

`detach()` 之后，`z` 不再属于原来的计算图。

这意味着：
- 后面基于 `z` 的运算不会继续回传梯度
- 常用于推理、保存中间结果、或者某些特殊训练技巧

---

## 🌟 十一、`no_grad()` 的作用

在验证或推理阶段，我们通常不需要计算梯度。

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

### `no_grad()` 的好处：
- 节省内存
- 加快推理速度
- 避免构建不必要的计算图

---

## 🌟 十二、一个完整的自动微分流程示例

下面我们做一个完整的小流程。

```python
import torch

# 定义参数
w = torch.tensor(5.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# 输入
x = torch.tensor(2.0)

# 前向计算
z = w * x + b
loss = (z - 11) ** 2

print("z =", z.item())
print("loss =", loss.item())

# 反向传播
loss.backward()

print("w.grad =", w.grad)
print("b.grad =", b.grad)
```

### 这段代码的逻辑是：
1. 用参数 `w` 和 `b` 计算输出 `z`
2. 和目标值 `11` 比较，得到损失 `loss`
3. 调用 `loss.backward()` 后，自动计算 `w` 和 `b` 的梯度
4. 梯度会告诉我们参数应该怎么更新

这就是神经网络训练的核心流程。

---

## 🌟 十三、为什么自动微分这么重要？

自动微分让我们可以轻松训练复杂模型，比如：
- 线性回归
- 逻辑回归
- MLP
- CNN
- RNN
- Transformer
- GPT

如果没有自动微分，这些模型的训练会变得非常繁琐。

所以可以说：

> **自动微分是 PyTorch 的灵魂功能之一。**

---

## 🌟 十四、常见错误和小提醒

### 1. 忘记设置 `requires_grad=True`
如果不设置，梯度不会被记录。

### 2. 忘记清空梯度
梯度会累加，训练前要清空。

### 3. 非标量直接 `backward()`
如果输出不是标量，通常需要先 `sum()` 或 `mean()`。

### 4. 不小心 `detach()` 掉梯度路径
会导致梯度断开，模型学不动。

---

## 🎯 十五、练习题：巩固自动微分

### 练习 1：最简单的平方函数
创建一个 `requires_grad=True` 的张量 `x=3`，计算：

```python
y = x ** 2
```

然后调用 `backward()`，查看梯度。

**问题：**
1. `y` 等于多少？
2. `dy/dx` 等于多少？
3. `x.grad` 应该是多少？

---

### 练习 2：链式法则

```python
x = torch.tensor(2.0, requires_grad=True)
y = x * 4
z = y + 1
loss = z ** 2
```

**问题：**
1. `loss` 的表达式是什么？
2. `loss` 对 `x` 的梯度是多少？
3. 用 PyTorch 验证结果。

---

### 练习 3：梯度累积

```python
x = torch.tensor(1.0, requires_grad=True)
y1 = x * 2
y1.backward()
y2 = x * 3
y2.backward()
```

**问题：**
1. 第一次 `x.grad` 是多少？
2. 第二次 `x.grad` 是多少？
3. 为什么会累加？

---

### 练习 4：detach 的影响

```python
x = torch.tensor(2.0, requires_grad=True)
y = x * 3
z = y.detach() * 5
```

**问题：**
1. `z` 是否还和 `x` 有梯度关系？
2. 如果去掉 `detach()` 会怎样？

---

## ✅ 参考答案

<details>
<summary>点击展开答案</summary>

### 练习 1 参考答案

```python
import torch

x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)
```

- `y = 9`
- `dy/dx = 2x = 6`
- `x.grad = 6`

### 练习 2 参考答案

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x * 4
z = y + 1
loss = z ** 2
loss.backward()
print(x.grad)
```

手算：
- `loss = (4x + 1)^2`
- `d(loss)/dx = 2(4x + 1) * 4`
- 当 `x = 2` 时：`2(9)*4 = 72`

所以 `x.grad = 72`

### 练习 3 参考答案

```python
import torch

x = torch.tensor(1.0, requires_grad=True)
y1 = x * 2
y1.backward()
print(x.grad)  # 2

y2 = x * 3
y2.backward()
print(x.grad)  # 5
```

因为梯度是累加的，所以第二次会变成 `2 + 3 = 5`

### 练习 4 参考答案

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x * 3
z = y.detach() * 5
print(z.requires_grad)  # False
```

`detach()` 会切断梯度关系，所以 `z` 不再和 `x` 连接。

</details>

---

## 🌈 小结

这一节最重要的内容是：

- 自动微分让 PyTorch 可以自动算梯度
- `requires_grad=True` 是求导的开关
- `backward()` 会触发反向传播
- `.grad` 保存梯度结果
- `detach()` 和 `no_grad()` 可以切断或关闭梯度跟踪
- 梯度会默认累加，训练时要记得清空

如果你能真正理解自动微分，后面学习深度学习模型训练就会轻松很多。

