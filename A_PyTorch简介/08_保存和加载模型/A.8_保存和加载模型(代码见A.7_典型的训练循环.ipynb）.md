# 💾 PyTorch 中如何保存和加载模型——超详细入门指南

> **一句话理解：**
> 保存和加载模型，就是把训练好的模型“拍照存档”，以后想继续训练、部署推理、换电脑使用时，都可以把它再“读回来”。
>
> 如果把训练神经网络比作培养一个学生：
> - **模型参数** 就像学生脑子里学到的知识
> - **checkpoint** 就像阶段性考试的成绩单和学习记录
> - **保存** 就是把学习成果存档
> - **加载** 就是把以前学到的东西恢复出来

---

## 📚 目录

- <a href="#sec-1">一、为什么要保存和加载模型？</a>
- <a href="#sec-2">二、PyTorch 里到底能保存什么？</a>
- <a href="#sec-3">三、最常见的保存方式：state_dict()</a>
- <a href="#sec-4">四、保存整个 checkpoint：模型 + 优化器 + 训练状态</a>
- <a href="#sec-5">五、加载模型参数：load_state_dict()</a>
- <a href="#sec-6">六、断点恢复训练：从 checkpoint 继续跑</a>
- <a href="#sec-7">七、保存和加载到不同设备（CPU / GPU）</a>
- <a href="#sec-8">八、保存自定义模型时的注意事项</a>
- <a href="#sec-9">九、一个完整可运行的示例</a>
- <a href="#sec-10">十、保存和加载时常见错误</a>
- <a href="#sec-11">十一、练习题：巩固模型保存与加载</a>
- <a href="#sec-12">参考答案</a>
- <a href="#sec-13">小结</a>

---

<a id="sec-1"></a>
## 一、为什么要保存和加载模型？

训练模型通常不是一次就结束的，我们经常需要：

1. **中断后继续训练**
   - 电脑关机
   - 训练任务被打断
   - 想第二天接着跑

2. **部署推理**
   - 训练完后，想把模型拿去做预测
   - 不想每次都重新训练

3. **对比不同实验**
   - 保存多个版本模型
   - 比较不同学习率、不同结构的结果

4. **分享给别人**
   - 让别人直接加载你的模型权重
   - 不必重新训练

所以，保存和加载模型是深度学习工作流里非常基础、也非常重要的一环。

---

<a id="sec-2"></a>
## 二、PyTorch 里到底能保存什么？

PyTorch 里我们通常会保存以下几类内容：

### 1. 模型参数
也就是模型学到的“知识”，通常通过 `state_dict()` 保存。

### 2. 优化器状态
比如 Adam 的动量、二阶矩等信息。

### 3. 学习率调度器状态
如果你使用了 scheduler，也可以一并保存。

### 4. 当前训练轮次（epoch）
方便断点恢复。

### 5. 随机数状态、AMP scaler 状态等
在更严谨的训练恢复中也常常会保存。

---

<a id="sec-3"></a>
## 三、最常见的保存方式：`state_dict()`

在 PyTorch 中，最推荐的保存方式通常不是直接保存整个模型对象，而是保存它的 `state_dict()`。

### 什么是 `state_dict()`？

`state_dict()` 可以理解为：

> 模型里所有可学习参数和缓冲区的“字典表”。

例如，模型可能包含：
- 各层的权重
- 各层的偏置
- BatchNorm 的均值方差等缓冲值

### 示例：保存模型参数

```python
import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
	def __init__(self):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(4, 8),
			nn.ReLU(),
			nn.Linear(8, 2)
		)

	def forward(self, x):
		return self.net(x)

model = SimpleMLP()
torch.save(model.state_dict(), "simple_mlp_weights.pth")
```

### 这一步保存了什么？

- 不是模型类本身
- 不是 forward 函数代码
- 只是模型里可训练的参数和相关状态

这也是为什么加载时，你仍然需要先“重新定义模型结构”。

---

<a id="sec-4"></a>
## 四、保存整个 checkpoint：模型 + 优化器 + 训练状态

如果你希望以后不仅能加载模型，还能**从上一次训练中继续跑**，那么通常会保存一个 checkpoint。

### checkpoint 里常见内容

```python
import torch
import torch.nn as nn
from torch.optim import SGD

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(4, 2)

    def forward(self, x):
        return self.net(x)

model = SimpleMLP()
optimizer = SGD(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

checkpoint = {
	"epoch": 3,
	"model_state_dict": model.state_dict(),
	"optimizer_state_dict": optimizer.state_dict(),
	"scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
	"loss": 0.456,
}

```

### 示例：保存 checkpoint

```python
import torch
import torch.nn as nn
from torch.optim import SGD

class SimpleMLP(nn.Module):
	def __init__(self):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(4, 8),
			nn.ReLU(),
			nn.Linear(8, 2)
		)

	def forward(self, x):
		return self.net(x)

model = SimpleMLP()
optimizer = SGD(model.parameters(), lr=0.1)
epoch = 3
loss = 0.456

checkpoint = {
	"epoch": epoch,
	"model_state_dict": model.state_dict(),
	"optimizer_state_dict": optimizer.state_dict(),
	"loss": loss,
}

torch.save(checkpoint, "checkpoint.pth")
```

---

<a id="sec-5"></a>
## 五、加载模型参数：`load_state_dict()`

加载模型时，一般流程是：

1. 先重新定义模型结构
2. 再加载保存下来的参数

### 示例：加载模型

```python
import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
	def __init__(self):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(4, 8),
			nn.ReLU(),
			nn.Linear(8, 2)
		)

	def forward(self, x):
		return self.net(x)

model = SimpleMLP()
state_dict = torch.load("simple_mlp_weights.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()
```

### 为什么要先定义模型结构？

因为 `state_dict()` 只包含参数值，不包含整个网络“长什么样”。

所以加载时必须让 PyTorch 知道：
- 哪些层存在
- 每层的输入输出维度是什么

---

<a id="sec-6"></a>
## 六、断点恢复训练：从 checkpoint 继续跑

如果你要从中断处继续训练，通常要恢复：

- 模型参数
- 优化器状态
- 当前 epoch
- 学习率调度器状态（如果有）

### 示例：恢复训练

```python
import torch
import torch.nn as nn
from torch.optim import SGD

class SimpleMLP(nn.Module):
	def __init__(self):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(4, 8),
			nn.ReLU(),
			nn.Linear(8, 2)
		)

	def forward(self, x):
		return self.net(x)

model = SimpleMLP()
optimizer = SGD(model.parameters(), lr=0.1)

checkpoint = torch.load("checkpoint.pth", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
start_epoch = checkpoint["epoch"] + 1

print("从第", start_epoch, "轮继续训练")
```

### 为什么优化器也要恢复？

因为优化器内部可能保存了：
- 动量
- 自适应学习率历史信息
- 一些累积统计量

如果只恢复模型参数，不恢复优化器状态，训练轨迹可能会变。

---

<a id="sec-7"></a>
## 七、保存和加载到不同设备（CPU / GPU）

PyTorch 支持把模型从 GPU 保存，再在 CPU 上加载，或者反过来。

### 示例：在 GPU 上训练，CPU 上加载

```python
import torch

checkpoint = torch.load("checkpoint.pth", map_location="cpu")
```

### 示例：在 GPU 上加载

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("checkpoint.pth", map_location=device)
```

### 为什么要用 `map_location`？

因为保存时的数据可能在某个设备上，而加载时设备不一定相同。

`map_location` 可以帮我们把张量映射到目标设备。

---

<a id="sec-8"></a>
## 八、保存自定义模型时的注意事项

### 1. 尽量保存 `state_dict()`，不要轻易直接保存整个模型对象

虽然你也可以：

```python
import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
	def __init__(self):
		super().__init__()
		self.net = nn.Linear(4, 2)

	def forward(self, x):
		return self.net(x)

model = SimpleMLP()
torch.save(model, "model.pth")
```

但更推荐保存：

```python
import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
	def __init__(self):
		super().__init__()
		self.net = nn.Linear(4, 2)

	def forward(self, x):
		return self.net(x)

model = SimpleMLP()
torch.save(model.state_dict(), "model_weights.pth")
```

### 为什么推荐 `state_dict()`？

- 更稳妥
- 更通用
- 更容易跨环境迁移
- 不容易受到类定义路径变化影响

### 2. 保存和加载时模型结构必须一致

例如，你保存时模型是：

```text
4 -> 8 -> 2
```

那加载时也必须定义成同样的结构。

### 3. 自定义类要保证代码可复现

如果你把模型对象本身保存下来，环境变化、文件路径变化都可能带来麻烦。

---

<a id="sec-9"></a>
## 九、一个完整可运行的示例

下面我们做一个最小的“训练 → 保存 → 加载 → 推理”流程。

```python
import torch
import torch.nn as nn
from torch.optim import SGD

class SimpleMLP(nn.Module):
	def __init__(self):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(4, 8),
			nn.ReLU(),
			nn.Linear(8, 2)
		)

	def forward(self, x):
		return self.net(x)

# 1. 创建模型与优化器
model = SimpleMLP()
optimizer = SGD(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

# 2. 假数据
x = torch.randn(10, 4)
y = torch.randint(0, 2, (10,))

# 3. 训练一步
optimizer.zero_grad()
logits = model(x)
loss = criterion(logits, y)
loss.backward()
optimizer.step()

# 4. 保存 checkpoint
checkpoint = {
	"epoch": 0,
	"model_state_dict": model.state_dict(),
	"optimizer_state_dict": optimizer.state_dict(),
	"loss": loss.item(),
}
torch.save(checkpoint, "demo_checkpoint.pth")

# 5. 新建一个模型，加载参数
new_model = SimpleMLP()
new_optimizer = SGD(new_model.parameters(), lr=0.1)

loaded = torch.load("demo_checkpoint.pth", map_location="cpu")
new_model.load_state_dict(loaded["model_state_dict"])
new_optimizer.load_state_dict(loaded["optimizer_state_dict"])

# 6. 推理
new_model.eval()
with torch.no_grad():
	pred = new_model(x)
	print(pred.shape)
```

### 这段代码展示了什么？

- 模型是可以训练的
- 训练后的参数可以保存
- 保存后的参数可以重新加载
- 加载后可以继续训练，也可以直接推理

---

<a id="sec-10"></a>
## 十、保存和加载时常见错误

### 1. 模型结构不一致

如果保存时和加载时的层数、维度不同，会报错。

### 2. 忘记 `map_location`

如果你在 GPU 上保存，在 CPU 上加载，而没指定 `map_location`，有时会出问题。

### 3. 只加载模型，不加载优化器

如果你想继续训练，优化器状态也应该恢复。

### 4. 忘记 `model.eval()`

推理时如果不切换到评估模式，Dropout、BatchNorm 的行为可能不对。

### 5. 没有使用 `torch.no_grad()`

推理时不需要梯度，最好关闭梯度跟踪，节省显存。

---

<a id="sec-11"></a>
## 十一、练习题：巩固模型保存与加载

### 练习 1：保存模型参数
写一个简单的 MLP，并把它的 `state_dict()` 保存到本地。

**问题：** 为什么推荐保存 `state_dict()`，而不是直接保存整个模型？

---

### 练习 2：加载模型参数
新建一个结构相同的模型，把保存的参数加载进去，并打印模型输出。

**问题：** 加载前为什么必须先定义模型结构？

---

### 练习 3：保存 checkpoint
把模型参数、优化器状态、当前 epoch 和 loss 一起保存成 checkpoint。

**问题：** 为什么断点恢复训练时要保存优化器状态？

---

### 练习 4：断点恢复
从 checkpoint 中恢复模型和优化器，然后继续训练一轮。

**问题：** 如果只加载模型参数，不加载优化器状态，会有什么影响？

---

<a id="sec-12"></a>
## ✅ 参考答案

<details>
<summary>点击展开答案</summary>

### 练习 1 参考答案

```python
import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
	def __init__(self):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(4, 8),
			nn.ReLU(),
			nn.Linear(8, 2)
		)

	def forward(self, x):
		return self.net(x)

model = SimpleMLP()
torch.save(model.state_dict(), "simple_mlp_weights.pth")
```

推荐保存 `state_dict()`，因为它更通用，也更容易跨环境恢复。

### 练习 2 参考答案

```python
import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
	def __init__(self):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(4, 8),
			nn.ReLU(),
			nn.Linear(8, 2)
		)

	def forward(self, x):
		return self.net(x)

model = SimpleMLP()
state_dict = torch.load("simple_mlp_weights.pth", map_location="cpu")
model.load_state_dict(state_dict)
```

必须先定义模型结构，因为 `state_dict` 只保存参数值，不保存网络结构本身。

### 练习 3 参考答案

```python
import torch
import torch.nn as nn
from torch.optim import SGD

class SimpleMLP(nn.Module):
	def __init__(self):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(4, 8),
			nn.ReLU(),
			nn.Linear(8, 2)
		)

	def forward(self, x):
		return self.net(x)

model = SimpleMLP()
optimizer = SGD(model.parameters(), lr=0.1)

checkpoint = {
	"epoch": 0,
	"model_state_dict": model.state_dict(),
	"optimizer_state_dict": optimizer.state_dict(),
	"loss": 0.123,
}

torch.save(checkpoint, "checkpoint.pth")
```

优化器状态中可能包含动量等信息，所以继续训练时建议一起恢复。

### 练习 4 参考答案

```python
import torch
import torch.nn as nn
from torch.optim import SGD

class SimpleMLP(nn.Module):
	def __init__(self):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(4, 8),
			nn.ReLU(),
			nn.Linear(8, 2)
		)

	def forward(self, x):
		return self.net(x)

model = SimpleMLP()
optimizer = SGD(model.parameters(), lr=0.1)

checkpoint = torch.load("checkpoint.pth", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

print("恢复到 epoch:", checkpoint["epoch"])
```

如果只加载模型参数，不加载优化器状态，训练轨迹可能会变，尤其在使用 Adam、带动量的 SGD 时更明显。

</details>

---

<a id="sec-13"></a>
## 🌈 小结

这一节最重要的是理解：

- `state_dict()` 是保存模型参数的标准方式
- `checkpoint` 可以保存更多训练状态，方便断点恢复
- 加载模型时要先定义结构，再加载参数
- 继续训练时通常要把优化器状态也恢复回来
- 使用 `map_location` 可以让模型在不同设备间灵活加载

如果你能熟练掌握保存和加载模型，训练大模型、做实验和部署推理都会方便很多。




