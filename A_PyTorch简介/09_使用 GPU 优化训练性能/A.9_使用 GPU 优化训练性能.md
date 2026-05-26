# ⚡ PyTorch 中如何使用 GPU 优化训练性能——超详细入门指南

> **一句话理解 GPU 优化：**
> GPU 优化的核心思路，就是把模型、数据和计算尽量搬到 GPU 上，让它利用大量并行计算能力，加速训练和推理。
>
> 如果把训练比作搬砖：
> - **CPU** 像一个很聪明但一次只能搬少量砖的人
> - **GPU** 像一个有成百上千只手的工厂
> - **把数据送上 GPU** 就像把砖提前搬到工厂里
> - **减少 CPU 和 GPU 之间来回传输** 就像减少搬运损耗

---

## 📚 目录

- 一、为什么 GPU 能加速训练？
- 二、PyTorch 中的设备是什么？
- 三、如何判断当前是否有 GPU 可用？
- 四、把模型和数据搬到 GPU 上
- 五、一个完整的 CPU / GPU 通用训练示例
- 六、`to(device)`、`cuda()`、`cpu()` 的区别
- 七、为什么要减少 CPU 和 GPU 之间的数据传输？
- 八、`pin_memory=True` 与 `non_blocking=True`
- 九、混合精度（AMP）与 GPU 加速
- 十、如何查看 GPU 占用与训练是否真的更快？
- 十一、常见错误和小提醒
- 十二、练习题：巩固 GPU 训练优化
- 参考答案
- 小结

---

## 一、为什么 GPU 能加速训练？

GPU 之所以适合深度学习，最重要的原因是：

### 1. 大量并行计算
神经网络中最常见的运算是：
- 矩阵乘法
- 向量加法
- 卷积
- 激活函数

这些运算可以被拆成大量小任务同时进行，而 GPU 正擅长这种并行处理。

### 2. 适合大批量数值运算
深度学习训练时，一次往往不是处理一个数字，而是一大批张量。
GPU 在这种场景里比 CPU 更有优势。

### 3. 对 Transformer / GPT 尤其重要
大语言模型里有大量矩阵乘法和注意力计算，GPU 几乎是必需品。

---

## 二、PyTorch 中的设备（device）是什么？

在 PyTorch 里，`device` 表示张量或模型所在的计算设备。

常见设备有：

- `cpu`
- `cuda`

### 例子

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])
print(x.device)
```

如果输出是 `cpu`，说明这个张量现在在 CPU 上。

---

## 三、如何判断当前是否有 GPU 可用？

在写代码时，第一步通常是检查 GPU 是否可用：

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
```

### 这几个函数的作用

- `torch.cuda.is_available()`：是否有可用 CUDA GPU
- `torch.cuda.device_count()`：有几块 GPU
- `torch.cuda.get_device_name(0)`：查看第 0 块 GPU 的名字

---

## 四、把模型和数据搬到 GPU 上

GPU 加速的关键，不只是“有 GPU”，而是**模型和数据都必须在同一个设备上**。

### 1. 定义 device

```python
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
```

### 2. 把张量搬到 device

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x = x.to(device)
print(x.device)
```

### 3. 把模型搬到 device

```python
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = nn.Linear(4, 2)
model = model.to(device)
```

### 为什么要一起搬？

如果模型在 GPU 上，而数据还在 CPU 上，会报错：

> Expected all tensors to be on the same device

---

## 五、一个完整的 CPU / GPU 通用训练示例

下面给出一个可以在 CPU 或 GPU 上运行的完整训练例子。

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimpleModel(nn.Module):
	def __init__(self, in_dim, out_dim):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(in_dim, 64),
			nn.ReLU(),
			nn.Linear(64, out_dim)
		)

	def forward(self, x):
		return self.net(x)

# 假数据
x = torch.randn(200, 20)
y = torch.randint(0, 3, (200,))
loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)

model = SimpleModel(20, 3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(3):
	model.train()
	total_loss = 0.0
	for batch_x, batch_y in loader:
		batch_x = batch_x.to(device)
		batch_y = batch_y.to(device)

		optimizer.zero_grad()
		logits = model(batch_x)
		loss = criterion(logits, batch_y)
		loss.backward()
		optimizer.step()

		total_loss += loss.item() * batch_x.size(0)

	avg_loss = total_loss / len(loader.dataset)
	print(f"epoch {epoch}: loss={avg_loss:.4f}")
```

### 这段代码里 GPU 优化体现在哪？

- 模型搬到了 `device`
- batch 数据每次也搬到了 `device`
- 训练过程中模型和数据一直在同一设备上计算

---

## 六、`to(device)`、`cuda()`、`cpu()` 的区别

### 1. `to(device)`

最通用、最推荐的写法。

比如：把张量写成 `x.to(device)`，把模型写成 `model.to(device)`。

### 2. `.cuda()`

把张量或模型直接送到 CUDA GPU 上。

比如：`x.cuda()`。

缺点是：
- 写法不够通用
- 不适合 CPU / GPU 自动切换

### 3. `.cpu()`

把张量或模型搬回 CPU。

比如：`x.cpu()`。

### 推荐写法

日常开发中，最好优先用：

`to(device)`。

因为它可以自动适配 CPU 和 GPU。

---

## 七、为什么要减少 CPU 和 GPU 之间的数据传输？

GPU 很快，但如果数据总是在 CPU 和 GPU 之间来回搬运，速度会被拖慢。

### 常见损耗来源

1. 数据还在 CPU，模型在 GPU
2. 每个 batch 都频繁复制
3. 训练中间反复把 tensor 拿回 CPU 做不必要的计算

### 正确思路

- 尽量让计算在 GPU 上连续完成
- 只有在必须打印、保存、画图时才把数据搬回 CPU

---

## 八、`pin_memory=True` 与 `non_blocking=True`

这两个参数常常一起出现在 GPU 训练优化里。

### 1. `pin_memory=True`

如果 DataLoader 使用：

`DataLoader(..., pin_memory=True)`。

它可以帮助 CPU 内存更快地拷贝到 GPU。

### 2. `non_blocking=True`

当张量从 CPU 传到 GPU 时，可以写成：

`batch_x = batch_x.to(device, non_blocking=True)`。

### 什么时候有用？

- 使用 GPU 训练时
- 数据传输成为瓶颈时

### 小提醒

这类优化不是“开了就一定快”，要结合你的数据加载和硬件情况实际测试。

---

## 九、混合精度（AMP）与 GPU 加速

AMP 是现代 GPU 训练里非常常见的加速方法。

### 它的思想

- 有些操作使用较低精度计算（如 float16）
- 保持整体训练效果的同时提升速度、减少显存占用

### 典型写法

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 3)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler()

x = torch.randn(32, 20).to(device)
y = torch.randint(0, 3, (32,)).to(device)

optimizer.zero_grad()
with autocast():
	logits = model(x)
	loss = criterion(logits, y)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### AMP 的优点

- 更省显存
- 通常更快
- 对大模型很有帮助

---

## 十、如何查看 GPU 占用与训练是否真的更快？

GPU 优化不是“感觉快”，而是要能观察到证据。

### 1. 看设备

你可以打印模型参数所在的设备，例如查看模型参数是否真的在 GPU 上。

### 2. 看是否在使用 GPU

你可以调用 `torch.cuda.is_available()` 判断 CUDA 是否可用。

### 3. 看训练时间

可以比较：
- CPU 训练耗时
- GPU 训练耗时

### 4. 看显存占用

在终端里可以使用：

```bash
nvidia-smi
```

它会显示：
- 当前 GPU 使用率
- 显存占用
- 哪个进程在占用 GPU

---

## 十一、常见错误和小提醒

### 1. 模型在 GPU，数据还在 CPU
会报设备不一致错误。

### 2. 忘记把标签也搬到 GPU
训练时不仅输入要搬，标签也要搬。

### 3. 推理时忘记 `no_grad()`
会多占显存，推理也更慢。

### 4. 认为 GPU 一定比 CPU 快
如果模型很小、数据很少，GPU 反而可能因为搬运开销而不明显更快。

### 5. 忽略数据加载瓶颈
有时 GPU 很强，但 DataLoader 太慢，GPU 还是会“吃不饱”。

---

## 十二、练习题：巩固 GPU 训练优化

### 练习 1：检查 GPU 可用性
请写代码检查你的环境是否支持 CUDA，并打印设备名称。

**问题：** `torch.cuda.is_available()` 和 `torch.cuda.device_count()` 分别表示什么？

---

### 练习 2：把模型和数据搬到 GPU
定义一个简单的线性模型，把模型和输入张量都搬到 `device` 上。

**问题：** 为什么模型和数据必须在同一设备上？

---

### 练习 3：使用 `pin_memory` 和 `non_blocking`
在 DataLoader 中开启 `pin_memory=True`，并在把 batch 搬到 GPU 时使用 `non_blocking=True`。

**问题：** 这两个参数分别解决什么问题？

---

### 练习 4：尝试 AMP
把一个简单的训练步骤改写成 AMP 版本。

**问题：** 为什么 AMP 在 GPU 上通常能提升训练性能？

---

## ✅ 参考答案

<details>
<summary>点击展开答案</summary>

### 练习 1 参考答案

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
```

### 练习 2 参考答案

```python
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = nn.Linear(4, 2).to(device)
x = torch.randn(3, 4).to(device)

out = model(x)
print(out)
```

模型和数据必须在同一设备上，否则会报设备不一致错误。

### 练习 3 参考答案

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

x = torch.randn(100, 4)
y = torch.randint(0, 2, (100,))
loader = DataLoader(TensorDataset(x, y), batch_size=16, pin_memory=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for batch_x, batch_y in loader:
	batch_x = batch_x.to(device, non_blocking=True)
	batch_y = batch_y.to(device, non_blocking=True)
	break
```

### 练习 4 参考答案

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 3)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler()

x = torch.randn(32, 20).to(device)
y = torch.randint(0, 3, (32,)).to(device)

optimizer.zero_grad()
with autocast():
	logits = model(x)
	loss = criterion(logits, y)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

</details>

---

## 🌈 小结

这一节最重要的是理解：

- GPU 训练的核心是把计算和数据尽量放到同一设备上
- `to(device)` 是最常见的设备迁移方式
- `pin_memory=True` 和 `non_blocking=True` 可以帮助减少数据传输开销
- AMP 可以进一步提升 GPU 训练性能
- GPU 是否真的更快，最终要看模型大小、数据规模和数据加载效率

如果你能熟练处理设备迁移、数据传输和 AMP，PyTorch 训练效率会提升很多。



