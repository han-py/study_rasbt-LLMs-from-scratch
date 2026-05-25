# 🧠 PyTorch 中如何实现多层神经网络（MLP）——超详细入门指南

> **一句话理解多层神经网络：**
> 多层神经网络（Multi-Layer Perceptron, MLP）就是把很多个“线性变换 + 非线性激活函数”叠在一起形成的模型。
>
> 如果把它比作一个工厂：
> - **输入层** 是原材料入口
> - **隐藏层** 是一条条加工流水线
> - **输出层** 是最终成品出口
> - **激活函数** 像是每道工序里的“筛选器”和“开关”
>
> 它的核心思想是：
> - 先用线性层做“组合与变换”
> - 再用非线性函数打破“只能画直线”的限制
> - 多层叠加后，就能表示更复杂的关系

---

## 🌟 一、什么是多层神经网络？

### 1. 单层模型的局限
如果只有一层线性变换，那么模型本质上只能做一些比较简单的线性映射。

比如：
```python
output = x @ W + b
```

这种形式虽然很常见，但它的表达能力有限。

### 2. 为什么要加“多层”？
因为现实世界的数据关系往往很复杂：
- 图像不是简单直线能分开的
- 文本语义不是一条公式就能完全描述的
- 用户行为、推荐系统也有很多非线性模式

所以我们需要：
- 多层线性变换
- 中间穿插非线性激活函数
- 让模型逐步提取更抽象、更复杂的特征

---

## 🌟 二、MLP 的基本结构

一个最经典的多层神经网络通常长这样：

```text
输入 x -> 线性层 -> 激活函数 -> 线性层 -> 激活函数 -> 线性层 -> 输出
```

例如：

```python
x -> Linear -> ReLU -> Linear -> ReLU -> Linear -> output
```

### 这里每一部分是什么意思？

#### 1. Linear（线性层）
负责做矩阵乘法和加偏置：

```python
output = x @ W + b
```

#### 2. 激活函数（Activation Function）
负责引入非线性。
常见激活函数有：
- ReLU
- Sigmoid
- Tanh
- GELU

#### 3. 输出层
根据任务不同，输出层的形式也不同：
- 回归任务：输出一个连续值
- 分类任务：输出类别分数

---

## 🌟 三、为什么激活函数这么重要？

如果神经网络里只有线性层，那么无论叠多少层，整体效果仍然可以“合并”为一个线性变换。

这就像：
- 你把很多次“乘法 + 加法”叠在一起
- 结果仍然还是一个线性关系

### 加入非线性后会怎样？
模型就能表示复杂的函数，比如：
- 曲线
- 分段关系
- 更复杂的分类边界

### 常见激活函数简述

#### ReLU
```python
f(x) = max(0, x)
```

优点：
- 简单
- 训练快
- 常用于隐藏层

#### Sigmoid
```python
f(x) = 1 / (1 + e^-x)
```

优点：
- 输出范围在 0 到 1
- 可用于概率解释

缺点：
- 深层网络里容易梯度消失

#### Tanh
输出范围是 -1 到 1。

#### GELU
Transformer 和 GPT 中非常常见。

---

## 🌟 四、用 PyTorch 搭建一个最简单的 MLP

下面我们用 `torch.nn` 来实现一个多层神经网络。

```python
import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleMLP(input_size=4, hidden_size=8, output_size=2)
print(model)
```

### 这段代码在做什么？

#### `class SimpleMLP(nn.Module)`
表示我们正在定义一个 PyTorch 模型。

#### `self.fc1`
第一层线性层，把输入特征变换到隐藏空间。

#### `self.relu`
激活函数，让模型变得更灵活。

#### `self.fc2`
第二层线性层，把隐藏表示映射到最终输出。

#### `forward()`
定义数据是如何在模型中前向流动的。

---

## 🌟 五、一个完整前向传播示例

```python
import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleMLP(4, 8, 2)

# 假设我们有 3 条样本，每条样本 4 个特征
x = torch.randn(3, 4)
out = model(x)

print("输入形状:", x.shape)
print("输出形状:", out.shape)
print("输出内容:\n", out)
```

### 这一段和“`grad_fn` 是什么”的关系

上面的 `out` 是一个**由模型前向传播算出来的中间结果**，因此它不是纯粹的“静态数值”，而是带着计算图信息的张量。

所以如果你在交互式环境里打印类似输出：

```python
tensor([[-0.1262,  0.1080, -0.1792]], grad_fn=<AddmmBackward0>)
```

就说明：

- 这个张量来自一个可求导运算
- 最后一步通常是线性层的矩阵乘法加偏置
- 反向传播节点是 `AddmmBackward0`

如果你想快速对照这些名字，可以查看同目录下的 `grad_fn速查版.md`。

---

## 🌟 六、把多层神经网络写得更像你在 notebook 里的代码

下面我们直接用你给出的 `NeuralNetwork` 结构，逐行解释它在做什么。

```python
import torch

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):  # 将输入和输出的数量编码为变量，使我们可以在具有不同特征数量和类别数量的数据集上重复相同的代码
        super().__init__()

        self.layers = torch.nn.Sequential(
            # 第一个隐藏层
            torch.nn.Linear(num_inputs, 30),  # 线性层将输入结点和输出结点的数量作为参数
            torch.nn.ReLU(),  # 非线性激活函数被放置在隐藏层之间

            # 第二个隐藏层
            torch.nn.Linear(30, 20),  # 下一个隐藏层的输出节点数量必须与下一层的输入节点数量相匹配
            torch.nn.ReLU(),

            # 输出层
            torch.nn.Linear(20, num_outputs)
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits  # 最后一层的输出称为 logits

model = NeuralNetwork(50, 3)
print(model)  # 查看模型结构的摘要
```

### 这段代码的结构含义

#### `NeuralNetwork(torch.nn.Module)`
这是在定义一个 PyTorch 模型类。

#### `num_inputs` 和 `num_outputs`
- `num_inputs`：输入特征数量
- `num_outputs`：输出类别数量或输出维度数量

这样写的好处是：
- 代码更通用
- 不同数据集可以重复使用
- 输入维度和类别数可以通过参数灵活传入

#### `self.layers = torch.nn.Sequential(...)`
表示把多层网络按顺序串起来：

```text
输入 -> Linear -> ReLU -> Linear -> ReLU -> Linear -> 输出
```

#### 为什么是 `30 -> 20 -> num_outputs`
这里人为设计了两个隐藏层：
- 第一隐藏层 30 个神经元
- 第二隐藏层 20 个神经元
- 最后一层输出 `num_outputs` 个分数

这不是唯一写法，只是一个典型的 MLP 结构。

#### `forward(self, x)`
定义数据前向流动方式。
所有输入张量都会先经过 `self.layers(x)`，最后返回 `logits`。

---

## 🌟 七、检查可训练参数总数

你可以这样统计模型里的训练参数：

```python
# 检查模型的可训练参数总数
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total number of trainable model parameters: {num_params}')
```

### 这里发生了什么？

- `model.parameters()`：取出模型所有参数
- `p.requires_grad`：只统计需要训练的参数
- `p.numel()`：统计每个参数张量里的元素总数
- `sum(...)`：把所有参数加起来

### 为什么要看参数总数？

因为它能帮助你：
- 判断模型规模
- 看自己搭的网络是否太大或太小
- 了解每一层到底有多少权重需要更新

---

## 🌟 八、查看权重和偏置

### 1. 访问第一个线性层的权重

```python
print(model.layers[0].weight)  # 访问第一个隐藏层的权重参数矩阵
```

这里的 `model.layers[0]` 就是第一层 `Linear(num_inputs, 30)`。

权重矩阵的形状通常是：

```python
print(model.layers[0].weight.shape)
```

对于 `Linear(num_inputs, 30)` 来说，形状一般是：

```text
(30, num_inputs)
```

这是因为 PyTorch 内部存储线性层参数时，通常按“输出维度 × 输入维度”保存。

### 2. 访问偏置

```python
print(model.layers[0].bias)  # 访问第一个隐藏层的偏置参数
```

偏置的长度一般等于输出神经元数量，也就是 30。

---

## 🌟 九、随机种子 `torch.manual_seed()` 的作用

```python
# 使用 .manual_seed() 函数设置随机数种子，从而确保每次运行代码时，结果都相同
torch.manual_seed(123)
model = NeuralNetwork(50, 3)
print(model.layers[0].weight)
```

### 这一步为什么重要？

神经网络层的参数初始化通常带随机性。
如果不固定随机种子：
- 每次运行模型初始化结果都不同
- 你看到的权重值也会不同
- 调试时不容易复现结果

固定随机种子后：
- 代码更容易复现
- 教学演示更稳定
- 你可以更好地对比前后结果

---

## 🌟 十、前向传播并查看输出

```python
# 通过前向传播使用 NeuralNetwork 实例
torch.manual_seed(123)
x = torch.rand(1, 50)
out = model(x)
print(out)
```

### 这里的输入代表什么？

- `x = torch.rand(1, 50)`
- 表示一个 batch 中有 1 条样本
- 每条样本有 50 个特征

### `out` 是什么？

`out` 是网络最后输出的 **logits**。

logits 的意思是：
- 还没有经过 softmax 的原始分数
- 不是概率
- 只是各类别的打分

### 为什么输出里会出现 `grad_fn=<AddmmBackward0>`？

因为最后一层 `Linear(20, num_outputs)` 底层就是：

```text
X @ W + b
```

这个操作在 PyTorch 的反向图里通常会显示为 `AddmmBackward0`。

### 输出示例

```python
tensor([[-0.1262,  0.1080, -0.1792]], grad_fn=<AddmmBackward0>)
```

这里的含义是：
- 这是一个 1×3 的输出张量
- 3 个数分别是 3 个类别的原始分数
- 它仍然在计算图中，所以带着 `grad_fn`

---

## 🌟 十一、`grad_fn` 的含义进一步展开

如果你想把这类输出和 `grad_fn` 对上号，可以这样理解：

- `AddmmBackward0`：最后一步是线性层
- `ReluBackward0`：中间经过了 ReLU
- `SoftmaxBackward0`：如果你显式做了 softmax，就会出现这个节点

所以你看到：

```python
tensor(..., grad_fn=<AddmmBackward0>)
```

就说明：
- 这个张量来自一个线性变换
- 它还可以继续参与反向传播
- 它不是一个“已经和梯度无关”的常量

---

## 🌟 十二、使用 `torch.no_grad()` 关闭梯度计算

如果你只是想做预测，而不需要训练，就可以关闭梯度记录：

```python
# 使用 torch.no_grad() 函数来禁用梯度计算
with torch.no_grad():
    out = model(x)
print(out)
```

### 这有什么意义？

- 节省显存
- 计算更快
- 不会构建不必要的计算图

### 这时输出会怎样？

通常你会看到：

```python
tensor([...])
```

而不再有 `grad_fn=...`。

---

## 🌟 十三、把输出变成类别概率

如果你要把模型输出变成“属于每个类别的概率”，就要用 softmax：

```python
# 为预测结果计算类别成员概率，就需要显示调用 softmax 函数
with torch.no_grad():
    out = torch.softmax(model(x), dim=1)  # 通过指定 dim=1 来沿着正确的维度计算 softmax; dim=0 是批次维度，dim=1 是类别维度
print(out)
```

### 这里为什么要写 `dim=1`？

因为你的输出形状是：

```text
(batch_size, num_classes)
```

比如 `(1, 3)`：
- `dim=0` 是 batch 维度
- `dim=1` 是类别维度

softmax 应该沿着类别维度计算，这样每一行才会变成概率分布。

### softmax 的结果满足什么条件？

- 每个值都在 0 到 1 之间
- 同一行加起来等于 1

比如：

```python
tensor([[0.31, 0.38, 0.31]])
```

这就可以解释成：
- 第 1 类概率 0.31
- 第 2 类概率 0.38
- 第 3 类概率 0.31

---

## 🌟 十四、输出、logits、概率三者的区别

### 1. logits
模型直接输出的原始分数。

### 2. softmax 后的概率
把 logits 变成归一化概率。

### 3. 预测类别
通常取概率最大的那个类别：

```python
pred_class = torch.argmax(out, dim=1)
```

---

## 🌟 十五、和 `grad_fn对照表.md` 的关系

你现在这个例子和 `grad_fn对照表.md` 可以直接对应起来：

- `Linear` 层输出 → 常见 `AddmmBackward0`
- `ReLU` → `ReluBackward0`
- `softmax` → `SoftmaxBackward0`
- `view / reshape` 等形状变换 → 对应各自的视图类 `grad_fn`

如果你对输出中看到的 `grad_fn` 不熟，可以直接去查同目录下的 `grad_fn对照表.md` 或 `grad_fn速查版.md`。

### 这里的形状是怎么变化的？

- 输入 `x` 的形状是 `(3, 4)`
  - 3 表示 batch size（批量大小）
  - 4 表示每个样本的特征数
- 经过第一层后变成 `(3, 8)`
- 经过第二层后变成 `(3, 2)`

也就是说，网络在逐层压缩或变换特征表示。

---

## 🌟 六、训练一个多层神经网络需要什么？

训练流程通常包括：

1. 准备数据
2. 定义模型
3. 定义损失函数
4. 定义优化器
5. 前向传播
6. 计算损失
7. 反向传播
8. 更新参数

---

## 🌟 七、一个完整的训练示例

下面我们使用一个随机生成的数据集，演示如何训练 MLP。

```python
import torch
import torch.nn as nn
from torch.optim import SGD

# 1. 定义模型
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleMLP(4, 8, 2)

# 2. 构造假数据
x = torch.randn(10, 4)
y = torch.randint(0, 2, (10,))

# 3. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.1)

# 4. 训练一步
optimizer.zero_grad()
logits = model(x)
loss = criterion(logits, y)
loss.backward()
optimizer.step()

print("loss =", loss.item())
```

### 这段代码里的关键点

#### `nn.CrossEntropyLoss()`
常用于分类任务。
它会自动帮我们处理 logits 和标签之间的差异。

#### `optimizer.zero_grad()`
清空上一轮的梯度。

#### `loss.backward()`
自动计算梯度。

#### `optimizer.step()`
更新模型参数。

---

## 🌟 八、分类任务里输出层应该注意什么？

### 1. 二分类
如果是二分类问题，通常可以：
- 输出 2 个类别分数
- 配合 `CrossEntropyLoss`

### 2. 多分类
如果有 10 类、100 类，都可以输出对应数量的分数。

### 3. 回归任务
如果任务是预测连续值，输出层通常只有 1 个神经元。

---

## 🌟 九、MLP 的参数是什么？

以这个模型为例：

```python
model = SimpleMLP(4, 8, 2)
```

它的可学习参数包括：
- 第一层的权重 `fc1.weight`
- 第一层的偏置 `fc1.bias`
- 第二层的权重 `fc2.weight`
- 第二层的偏置 `fc2.bias`

### 查看参数

```python
for name, param in model.named_parameters():
    print(name, param.shape)
```

你会看到每个参数的名字和形状。

---

## 🌟 十、为什么多层网络比单层更强？

因为它可以逐步提取特征：

- 第一层可能学到简单模式
- 第二层学到组合模式
- 第三层学到更抽象的表示

这就像人类看图：
- 先看边缘
- 再看纹理
- 再看轮廓
- 最后识别出具体物体

多层神经网络就是在模拟这种“逐层抽象”的过程。

---

## 🌟 十一、和大语言模型有什么关系？

虽然 GPT 不是简单的 MLP，但它里面也有很多类似“线性层 + 激活函数”的结构。

比如 Transformer 里的前馈网络（Feed Forward Network, FFN）就可以看成一种特殊的多层神经网络。

所以先学会 MLP，后面理解 Transformer 会轻松很多。

---

## 🌟 十二、一个更像真实任务的小例子

假设我们有 2 个输入特征：
- 身高
- 体重

我们希望预测一个人属于某个类别，比如：
- 偏瘦
- 正常
- 偏胖

可以这样设计一个 MLP：

```python
import torch
import torch.nn as nn

class BodyTypeClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )

    def forward(self, x):
        return self.net(x)

model = BodyTypeClassifier()

x = torch.tensor([
    [170.0, 60.0],
    [180.0, 80.0],
    [160.0, 45.0]
])

out = model(x)
print(out.shape)
```

这里：
- 输入是 2 个特征
- 中间有两层隐藏层
- 最后输出 3 个类别分数

---

## 🌟 十三、用 `nn.Sequential` 简化写法

如果网络结构很简单，可以直接用 `nn.Sequential`。

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 2)
)
```

这种写法非常简洁，适合快速实验。

如果模型逻辑更复杂，通常再使用自定义 `class`。

---

## 🌟 十四、训练多层神经网络时常见问题

### 1. 输入维度不对
比如模型期待 `(batch, 4)`，但你却传了 `(4,)`。

### 2. 标签格式不对
分类任务中，`CrossEntropyLoss` 要求标签是类别索引，而不是 one-hot 向量。

### 3. 忘记清空梯度
会导致梯度累积错误。

### 4. 学习率太大或太小
- 太大学不稳
- 太小训练太慢

### 5. 激活函数选错
比如深层网络里如果用不合适的激活函数，可能会影响训练效果。

---

## 🎯 十五、练习题：巩固多层神经网络

### 练习 1：补全模型
请补全下面的模型，让它成为一个两层 MLP：

```python
import torch
import torch.nn as nn

class MyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 10)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(10, 3)

    def forward(self, x):
        # TODO: 补全前向传播
        pass
```

---

### 练习 2：观察形状变化
创建一个输入张量 `x`，形状为 `(4, 5)`，传入一个 `5 -> 10 -> 3` 的 MLP，观察输出形状。

**问题：** 输出张量的形状是多少？

---

### 练习 3：查看参数
使用 `named_parameters()` 打印模型中的所有参数名和形状。

**问题：** 你看到了哪些参数？

---

### 练习 4：训练一步
自己构造一个小批量数据，完成一次前向传播、计算损失、反向传播和参数更新。

**提示：** 可以用 `nn.CrossEntropyLoss()`。

---

## ✅ 参考答案

<details>
<summary>点击展开参考答案</summary>

### 练习 1 参考答案

```python
import torch
import torch.nn as nn

class MyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 10)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(10, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
```

### 练习 2 参考答案

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(5, 10),
    nn.ReLU(),
    nn.Linear(10, 3)
)

x = torch.randn(4, 5)
out = model(x)
print(out.shape)
```

输出形状是：`(4, 3)`

### 练习 3 参考答案

```python
for name, param in model.named_parameters():
    print(name, param.shape)
```

### 练习 4 参考答案

```python
import torch
import torch.nn as nn
from torch.optim import SGD

model = nn.Sequential(
    nn.Linear(5, 10),
    nn.ReLU(),
    nn.Linear(10, 3)
)

x = torch.randn(8, 5)
y = torch.randint(0, 3, (8,))

criterion = nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.1)

optimizer.zero_grad()
logits = model(x)
loss = criterion(logits, y)
loss.backward()
optimizer.step()

print(loss.item())
```

</details>

---

## 🌈 小结

这一节最重要的是理解：

- 多层神经网络 = 多个线性层 + 激活函数的组合
- 线性层负责变换特征
- 激活函数负责引入非线性
- `nn.Module` 是构建模型的基础
- `forward()` 定义数据如何流过网络
- 训练流程包括前向传播、损失计算、反向传播和参数更新

如果你已经理解了这一节，那么后面学习更复杂的神经网络结构时就会非常顺畅。

