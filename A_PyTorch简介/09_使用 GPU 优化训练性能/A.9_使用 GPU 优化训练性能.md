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

- <a href="#sec-1">一、为什么 GPU 能加速训练？</a>
- <a href="#sec-2">二、PyTorch 中的设备是什么？</a>
- <a href="#sec-3">三、如何判断当前是否有 GPU 可用？</a>
- <a href="#sec-4">四、把模型和数据搬到 GPU 上</a>
- <a href="#sec-5">五、一个完整的 CPU / GPU 通用训练示例</a>
- <a href="#sec-6">六、`to(device)`、`cuda()`、`cpu()` 的区别</a>
- <a href="#sec-7">七、为什么要减少 CPU 和 GPU 之间的数据传输？</a>
- <a href="#sec-8">八、`pin_memory=True` 与 `non_blocking=True`</a>
- <a href="#sec-9">九、混合精度（AMP）与 GPU 加速</a>
- <a href="#sec-10">十、如何查看 GPU 占用与训练是否真的更快？</a>
- <a href="#sec-11">十一、常见错误和小提醒</a>
- <a href="#sec-12">十二、练习题：巩固 GPU 训练优化</a>
- <a href="#sec-13">参考答案</a>
- <a href="#sec-14">小结</a>
 - <a href="#sec-15">十五、多 GPU 训练（DistributedDataParallel / DDP）</a>

---

<a id="sec-1"></a>
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

<a id="sec-2"></a>
## 二、PyTorch 中的设备是什么？

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

<a id="sec-3"></a>
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

<a id="sec-4"></a>
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

<a id="sec-5"></a>
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

<a id="sec-6"></a>
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

<a id="sec-7"></a>
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

<a id="sec-8"></a>
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

<a id="sec-9"></a>
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

<a id="sec-10"></a>
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

<a id="sec-11"></a>
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

<a id="sec-12"></a>
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

<a id="sec-13"></a>
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

---

<a id="sec-15"></a>
## 十五、多 GPU 训练（DistributedDataParallel / DDP）

当系统中有多块 GPU 时，推荐使用 PyTorch 的 DistributedDataParallel（DDP）来进行并行训练。DDP 通过数据并行的方式，让每张 GPU 处理不同的数据批次，然后在每个训练步骤后同步梯度，从而实现加速训练。

---

### 📖 15.1 什么是分布式数据并行（DDP）？

**核心思想：**
- 把同一个模型复制到多张 GPU 上
- 每张 GPU 处理不同的数据子集
- 每次反向传播后，自动在所有 GPU 之间同步梯度
- 这样相当于用多个 GPU 同时训练，速度更快

**类比理解：**
如果把训练比作批改作业：
- **单 GPU**：一个老师批改所有学生的作业
- **多 GPU (DDP)**：多个老师每人批改一部分作业，然后互相交流评分标准（梯度同步），最后大家一起进步

---

### 📖 15.2 DDP 工作流程图

```
┌─────────────────────────────────────────────────────┐
│                  主进程 (Main Process)                │
│         启动多个子进程，每个子进程对应一张 GPU          │
└─────────────────────────────────────────────────────┘
                        ↓
        ┌───────────────┬───────────────┐
        ↓               ↓               ↓
   ┌─────────┐    ┌─────────┐    ┌─────────┐
   │ GPU 0   │    │ GPU 1   │    │ GPU 2   │
   │ Rank 0  │    │ Rank 1  │    │ Rank 2  │
   ├─────────┤    ├─────────┤    ├─────────┤
   │ Model   │    │ Model   │    │ Model   │
   │ (副本)   │    │ (副本)   │    │ (副本)   │
   ├─────────┤    ├─────────┤    ├─────────┤
   │ Batch 0 │    │ Batch 1 │    │ Batch 2 │
   │ (数据子集)│    │ (数据子集)│    │ (数据子集)│
   └─────────┘    └─────────┘    └─────────┘
        ↓               ↓               ↓
   前向传播 + 反向传播   前向传播 + 反向传播   前向传播 + 反向传播
        ↓               ↓               ↓
   计算梯度            计算梯度            计算梯度
        ↓               ↓               ↓
   ┌─────────────────────────────────────┐
   │     All-Reduce: 梯度同步与平均        │
   │   (所有 GPU 之间通信，同步梯度信息)     │
   └─────────────────────────────────────┘
        ↓               ↓               ↓
   更新模型参数         更新模型参数         更新模型参数
   (所有副本保持一致)    (所有副本保持一致)    (所有副本保持一致)
```

---

### 📖 15.3 A.13.py 代码逐行详解

下面我们将 `A.13.py` 中的关键部分拆解讲解，帮助你理解每一行代码的作用。

#### 🔑 第一步：导入必要的库

```python
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# NEW imports for DDP:
import os
import platform
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
```

**新增导入说明：**

| 导入项 | 作用 |
|--------|------|
| `os` | 设置环境变量（如 MASTER_ADDR、MASTER_PORT） |
| `platform` | 检测操作系统（Windows/Linux），选择合适的后端 |
| `torch.multiprocessing as mp` | 多进程管理，为每张 GPU 启动独立进程 |
| `DistributedSampler` | 分布式数据采样器，确保不同 GPU 处理不同数据 |
| `DistributedDataParallel as DDP` | DDP 包装器，实现多 GPU 并行训练 |
| `init_process_group` | 初始化分布式进程组，建立 GPU 间通信 |
| `destroy_process_group` | 训练结束后清理进程组资源 |

---

#### 🔑 第二步：初始化分布式进程组

```python
def ddp_setup(rank, world_size):
    """
    初始化分布式训练环境
    
    参数:
        rank: 当前进程的编号（唯一 ID），对应某一张 GPU
        world_size: 总共有多少个进程（通常等于 GPU 数量）
    """
    # 设置主节点地址（所有进程通过这个地址通信）
    # 这里假设所有 GPU 在同一台机器上
    os.environ["MASTER_ADDR"] = "localhost"
    
    # 设置主节点端口（任意空闲端口即可）
    os.environ["MASTER_PORT"] = "12345"

    # 根据操作系统选择不同的通信后端
    if platform.system() == "Windows":
        # Windows 系统：禁用 libuv（PyTorch for Windows 不支持）
        os.environ["USE_LIBUV"] = "0"
        # Windows 使用 "gloo" 后端（Facebook 开发的集体通信库）
        init_process_group(backend="gloo", rank=rank, world_size=world_size)
    else:
        # Linux/macOS 使用 "nccl" 后端（NVIDIA 优化的通信库，速度更快）
        init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # 将当前进程绑定到对应的 GPU
    torch.cuda.set_device(rank)
```

**关键点解释：**

1. **rank（进程编号）**：
   - 每个 GPU 对应一个进程，每个进程有唯一的 rank
   - 例如：2 张 GPU 时，rank 分别为 0 和 1

2. **world_size（世界大小）**：
   - 表示总共有多少个进程（通常等于 GPU 数量）
   - 例如：2 张 GPU 时，world_size = 2

3. **MASTER_ADDR 和 MASTER_PORT**：
   - 所有进程通过这个地址和端口进行通信
   - 类似"指挥中心"，协调各个 GPU 之间的数据同步

4. **后端选择（backend）**：
   - **nccl**：NVIDIA Collective Communication Library，专为 NVIDIA GPU 优化，速度最快（Linux/macOS 推荐）
   - **gloo**：Facebook Collective Communication Library，跨平台支持更好（Windows 必须用这个）

5. **torch.cuda.set_device(rank)**：
   - 确保当前进程只使用指定的 GPU
   - 例如：rank=0 的进程只用 GPU 0，rank=1 的进程只用 GPU 1

---

#### 🔑 第三步：定义数据集和数据加载器

```python
class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self):
        return self.labels.shape[0]
```

这部分是标准的 PyTorch Dataset 定义，和普通训练没有区别。

**关键在 DataLoader 的配置：**

```python
def prepare_dataset():
    X_train = torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5]
    ])
    y_train = torch.tensor([0, 0, 0, 1, 1])

    X_test = torch.tensor([
        [-0.8, 2.8],
        [2.6, -1.6],
    ])
    y_test = torch.tensor([0, 1])

    # 如果要用更多 GPU，可以取消下面的注释来增加数据量
    # factor = 4
    # X_train = torch.cat([X_train + torch.randn_like(X_train) * 0.1 for _ in range(factor)])
    # y_train = y_train.repeat(factor)
    # X_test = torch.cat([X_test + torch.randn_like(X_test) * 0.1 for _ in range(factor)])
    # y_test = y_test.repeat(factor)

    train_ds = ToyDataset(X_train, y_train)
    test_ds = ToyDataset(X_test, y_test)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=False,  # ⚠️ 重要：设为 False，因为 DistributedSampler 会负责打乱
        pin_memory=True,  # 加速 CPU 到 GPU 的数据传输
        drop_last=True,   # 丢弃最后一个不完整的 batch
        sampler=DistributedSampler(train_ds)  # ⭐ 关键：使用分布式采样器
    )
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=2,
        shuffle=False,
    )
    return train_loader, test_loader
```

**DistributedSampler 的作用：**

| 特性 | 说明 |
|------|------|
| **数据分片** | 自动将数据集分成多份，每张 GPU 拿到不同的数据子集 |
| **避免重复** | 确保同一条数据不会在多个 GPU 上重复处理 |
| **Epoch 打乱** | 每个 epoch 可以有不同的数据分配顺序（通过 `set_epoch()` 实现） |

**为什么 shuffle=False？**
- 因为 `DistributedSampler` 内部已经实现了打乱功能
- 如果同时开启 shuffle，会导致冲突和不可预期的行为

---

#### 🔑 第四步：定义神经网络模型

```python
class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(
            # 第 1 个隐藏层
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),

            # 第 2 个隐藏层
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # 输出层
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits
```

这是一个简单的三层神经网络，和普通训练完全一样，不需要特殊修改。

---

#### 🔑 第五步：主训练函数（核心部分）

```python
def main(rank, world_size, num_epochs):
    # ① 初始化分布式进程组
    ddp_setup(rank, world_size)

    # ② 准备数据加载器
    train_loader, test_loader = prepare_dataset()
    
    # ③ 创建模型并移动到当前 GPU
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    model.to(rank)  # 将模型放到 rank 对应的 GPU 上
    
    # ④ 创建优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    # ⑤ ⭐ 关键步骤：用 DDP 包装模型
    model = DDP(model, device_ids=[rank])
    # 包装后，原始模型可以通过 model.module 访问

    # ⑥ 开始训练循环
    for epoch in range(num_epochs):
        # ⑦ 设置当前 epoch 的采样器（保证每个 epoch 数据打乱顺序不同）
        train_loader.sampler.set_epoch(epoch)

        model.train()
        for features, labels in train_loader:
            # ⑧ 将数据移动到当前 GPU
            features, labels = features.to(rank), labels.to(rank)
            
            # ⑨ 前向传播
            logits = model(features)
            loss = F.cross_entropy(logits, labels)

            # ⑩ 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ⑪ 打印日志（带上 rank 标识，方便区分不同 GPU 的输出）
            print(f"[GPU{rank}] Epoch: {epoch+1:03d}/{num_epochs:03d}"
                  f" | Batchsize {labels.shape[0]:03d}"
                  f" | Train/Val Loss: {loss:.2f}")

    # ⑫ 训练结束后评估模型
    model.eval()

    try:
        train_acc = compute_accuracy(model, train_loader, device=rank)
        print(f"[GPU{rank}] Training accuracy", train_acc)
        test_acc = compute_accuracy(model, test_loader, device=rank)
        print(f"[GPU{rank}] Test accuracy", test_acc)

    except ZeroDivisionError as e:
        # 处理小数据集在多 GPU 下某些进程没有数据的情况
        raise ZeroDivisionError(
            f"{e}\n\nThis script is designed for 2 GPUs. You can run it as:\n"
            "CUDA_VISIBLE_DEVICES=0,1 python DDP-script.py\n"
            f"Or, to run it on {torch.cuda.device_count()} GPUs, uncomment the code on lines 103 to 107."
        )

    # ⑬ 清理分布式进程组资源
    destroy_process_group()
```

**关键步骤详解：**

##### ⑤ DDP 包装模型

```python
model = DDP(model, device_ids=[rank])
```

这行代码做了什么？
- 将普通模型包装成 DDP 模型
- 自动处理梯度同步（All-Reduce 操作）
- 确保所有 GPU 上的模型副本保持一致

**注意：**
- 包装后，如果要访问原始模型的属性，需要用 `model.module`
- 例如：`model.module.layers` 而不是 `model.layers`

##### ⑦ set_epoch 的作用

```python
train_loader.sampler.set_epoch(epoch)
```

为什么要调用这个？
- `DistributedSampler` 在每个 epoch 需要不同的数据打乱顺序
- `set_epoch(epoch)` 告诉采样器当前是第几个 epoch
- 这样每个 epoch 的数据分配都会不同，提高训练效果

##### ⑧ 数据移动到 GPU

```python
features, labels = features.to(rank), labels.to(rank)
```

- 这里的 `rank` 就是 GPU 编号
- 确保数据和模型在同一张 GPU 上
- 等价于 `features.to(f'cuda:{rank}')`

##### ⑬ 清理资源

```python
destroy_process_group()
```

- 训练结束后必须调用，释放分布式训练占用的资源
- 否则可能导致进程无法正常退出

---

#### 🔑 第六步：准确率计算函数

```python
def compute_accuracy(model, dataloader, device):
    model = model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)

        with torch.no_grad():
            logits = model(features)
        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)
    return (correct / total_examples).item()
```

这是标准的准确率计算函数，和普通训练没有区别。

**注意：**
- 使用 `torch.no_grad()` 关闭梯度计算，节省显存
- `device` 参数传入的是 `rank`，确保数据移动到正确的 GPU

---

#### 🔑 第七步：程序入口（启动多进程）

```python
if __name__ == "__main__":
    # 检查 PyTorch 版本和 GPU 可用性
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Number of GPUs available:", torch.cuda.device_count())
    
    torch.manual_seed(123)  # 设置随机种子，保证可复现性

    # 启动多进程训练
    num_epochs = 3
    world_size = torch.cuda.device_count()  # GPU 数量 = 进程数量
    
    # ⭐ 关键：mp.spawn 会为每个 GPU 启动一个进程
    # 每个进程会自动获得一个 rank 参数（0, 1, 2, ...）
    mp.spawn(main, args=(world_size, num_epochs), nprocs=world_size)
    # nprocs=world_size 表示启动 world_size 个进程（每张 GPU 一个）
```

**mp.spawn 的工作原理：**

```python
mp.spawn(main, args=(world_size, num_epochs), nprocs=world_size)
```

这行代码做了什么？
1. 启动 `world_size` 个子进程（例如 2 张 GPU 就启动 2 个进程）
2. 每个子进程自动获得一个唯一的 `rank`（从 0 开始编号）
3. 每个子进程都会调用 `main(rank, world_size, num_epochs)` 函数
4. 每个进程独立运行，但通过进程组进行通信和同步

**执行流程示意：**

```
主进程 (if __name__ == "__main__")
    ↓
检测到 2 张 GPU → world_size = 2
    ↓
mp.spawn 启动 2 个子进程
    ↓
┌────────────────────────┬────────────────────────┐
│   子进程 0 (rank=0)    │   子进程 1 (rank=1)    │
│   使用 GPU 0           │   使用 GPU 1           │
│                        │                        │
│   main(0, 2, 3)        │   main(1, 2, 3)        │
│   - ddp_setup          │   - ddp_setup          │
│   - 创建模型 → GPU 0   │   - 创建模型 → GPU 1   │
│   - DDP 包装           │   - DDP 包装           │
│   - 训练循环           │   - 训练循环           │
│   - 梯度同步 ←——→      │   - 梯度同步 ←——→      │
│   - 评估模型           │   - 评估模型           │
│   - destroy_process    │   - destroy_process    │
└────────────────────────┴────────────────────────┘
```

---

### 📖 15.4 如何运行 DDP 脚本

#### ✅ 推荐运行方式（终端执行）

**Linux / macOS:**
```bash
# 指定使用前 2 张 GPU
CUDA_VISIBLE_DEVICES=0,1 python A.13.py
```

**Windows (PowerShell):**
```powershell
# 设置环境变量
$env:CUDA_VISIBLE_DEVICES="0,1"
python A.13.py
```

**Windows (CMD):**
```cmd
set CUDA_VISIBLE_DEVICES=0,1
python A.13.py
```

#### ⚠️ 重要提醒

1. **必须在脚本中运行，不能在 Jupyter Notebook 中运行**
   - `mp.spawn` 需要在独立的 Python 脚本中执行
   - Jupyter 的多进程支持不完善，容易报错

2. **确保 `if __name__ == "__main__":` 存在**
   - 这是 Python 多进程的入口保护
   - 防止子进程重复执行主程序代码

3. **GPU 数量要匹配**
   - 如果只有 1 张 GPU，可以修改代码降级为单进程运行
   - 见下方"回退方案"

---

### 📖 15.5 常见错误与排查

#### ❌ 错误 1：ProcessExitedException

**错误信息：**
```
ProcessExitedException: process 0 terminated with exit code 1
```

**原因：**
- 子进程启动失败或抛出异常
- 通常在 Jupyter/Notebook 中运行 `mp.spawn` 时出现

**解决方法：**
1. 把代码保存为独立脚本（如 `A.13.py`）
2. 在终端运行，不要在 Jupyter 中运行
3. 确保有 `if __name__ == "__main__":` 保护

---

#### ❌ 错误 2：ZeroDivisionError

**错误信息：**
```
ZeroDivisionError: division by zero
```

**原因：**
- 数据集太小，分到某些 GPU 上没有数据
- 例如：5 条数据分给 2 张 GPU，batch_size=2，可能某个 GPU 拿不到完整 batch

**解决方法：**
1. 增加数据量（取消代码中 103-107 行的注释）
2. 减少 GPU 数量（只用 1 张 GPU 测试）
3. 调整 batch_size

---

#### ❌ 错误 3：RuntimeError - Backend not available

**错误信息：**
```
RuntimeError: Distributed package doesn't have NCCL built in
```

**原因：**
- 在 Windows 上使用了 `nccl` 后端（Windows 不支持）

**解决方法：**
- 代码中已经做了平台检测，Windows 会自动使用 `gloo`
- 如果手动修改了 backend，请改回来

---

#### ❌ 错误 4：CUDA out of memory

**错误信息：**
```
RuntimeError: CUDA out of memory
```

**原因：**
- 显存不足
- 可能是 batch_size 太大或模型太大

**解决方法：**
1. 减小 batch_size
2. 使用混合精度训练（AMP）
3. 减少模型大小

---

### 📖 15.6 调试建议

#### 💡 技巧 1：先单 GPU 测试

在迁移到多 GPU 之前，先在单 GPU 或 CPU 上确保代码能正常运行：

```python
# 临时测试：单进程模式
if __name__ == '__main__':
    torch.manual_seed(123)
    num_epochs = 3
    world_size = max(1, torch.cuda.device_count())
    
    if world_size > 1:
        # 多 GPU 模式
        mp.spawn(main, args=(world_size, num_epochs), nprocs=world_size)
    else:
        # 单 GPU/CPU 模式（fallback）
        main(rank=0, world_size=1, num_epochs=num_epochs)
```

#### 💡 技巧 2：带 rank 的日志

在每个进程中打印带 rank 标识的日志，方便区分不同 GPU 的输出：

```python
print(f"[GPU{rank}] Epoch: {epoch+1:03d}/{num_epochs:03d} | Loss: {loss:.2f}")
```

#### 💡 技巧 3：查看 GPU 使用情况

在另一个终端窗口运行：

```bash
# Linux/macOS
watch -n 1 nvidia-smi

# Windows
nvidia-smi -l 1
```

可以看到：
- 哪些 GPU 正在被使用
- 显存占用情况
- GPU 利用率

#### 💡 技巧 4：写入日志文件

对于复杂的多 GPU 训练，可以将每个进程的日志写入不同文件：

```python
import logging

def main(rank, world_size, num_epochs):
    # 为每个 rank 创建独立的日志文件
    logging.basicConfig(
        filename=f'train_rank{rank}.log',
        level=logging.INFO,
        format=f'[Rank {rank}] %(message)s'
    )
    
    # 使用 logging.info 代替 print
    logging.info(f"Starting training on GPU {rank}")
```

---

### 📖 15.7 DDP vs DataParallel 对比

PyTorch 提供了两种多 GPU 训练方式：

| 特性 | DataParallel (DP) | DistributedDataParallel (DDP) |
|------|-------------------|-------------------------------|
| **并行方式** | 单进程多线程 | 多进程 |
| **速度** | 较慢（GIL 限制） | 更快（无 GIL 限制） |
| **推荐程度** | ❌ 不推荐 | ✅ 强烈推荐 |
| **适用场景** | 快速原型开发 | 生产环境、大规模训练 |
| **通信效率** | 低（主 GPU 瓶颈） | 高（All-Reduce 优化） |
| **代码复杂度** | 简单 | 稍复杂 |

**结论：**
- **优先使用 DDP**，它是 PyTorch 官方推荐的多 GPU 训练方式
- DP 只在快速实验时使用，不适合生产环境

---

### 📖 15.8 完整代码回退方案

如果你的环境没有多张 GPU，可以使用以下回退方案：

```python
if __name__ == '__main__':
    torch.manual_seed(123)
    num_epochs = 3
    world_size = max(1, torch.cuda.device_count())
    
    if world_size > 1:
        # 多 GPU 模式
        print(f"Starting DDP training with {world_size} GPUs")
        mp.spawn(main, args=(world_size, num_epochs), nprocs=world_size)
    else:
        # 单 GPU/CPU 模式（fallback）
        print("No multiple GPUs detected, running in single-process mode")
        main(rank=0, world_size=1, num_epochs=num_epochs)
```

这样可以：
- 有多张 GPU 时自动使用 DDP
- 只有一张 GPU 或没有 GPU 时也能正常运行
- 方便在不同环境下测试代码

---

### 📖 15.9 小结：DDP 核心要点

✅ **必须做的：**
1. 使用 `init_process_group` 初始化进程组
2. 使用 `DistributedSampler` 分配数据
3. 每个 epoch 调用 `sampler.set_epoch(epoch)`
4. 用 `DDP()` 包装模型
5. 训练结束后调用 `destroy_process_group()`

✅ **推荐的：**
1. 在脚本中运行，不在 Jupyter 中运行
2. 使用 `if __name__ == "__main__":` 保护入口
3. 打印带 rank 的日志，方便调试
4. 先用单 GPU 测试，再迁移到多 GPU
5. 使用 `pin_memory=True` 加速数据传输

❌ **避免的：**
1. 不要在 DDP 中使用 `shuffle=True`（会让 DistributedSampler 失效）
2. 不要忘记把数据移动到正确的 GPU（`to(rank)`）
3. 不要在 Jupyter 中运行 `mp.spawn`
4. 不要在小数据集上使用太多 GPU（会导致某些 GPU 没有数据）

---

### 🎯 练习题：巩固 DDP 知识

**练习 1：** 解释 `rank` 和 `world_size` 的含义，以及它们在 DDP 中的作用。

**练习 2：** 为什么在使用 `DistributedSampler` 时，DataLoader 的 `shuffle` 参数要设为 `False`？

**练习 3：** `DDP(model, device_ids=[rank])` 这行代码做了什么？包装后的模型如何访问原始模型的属性？

**练习 4：** 为什么每个 epoch 都要调用 `train_loader.sampler.set_epoch(epoch)`？

**练习 5：** 如果你在 Windows 上运行 DDP 代码，应该使用哪个 backend？为什么？

<details>
<summary>点击查看答案解析</summary>

**练习 1 答案：**
- `rank`：当前进程的编号（唯一 ID），对应某一张 GPU。例如：2 张 GPU 时，rank 分别为 0 和 1
- `world_size`：总共有多少个进程（通常等于 GPU 数量）。例如：2 张 GPU 时，world_size = 2
- 作用：`rank` 用于区分不同进程，确保每个进程使用不同的 GPU；`world_size` 用于初始化进程组，确定通信范围

**练习 2 答案：**
- 因为 `DistributedSampler` 内部已经实现了数据打乱功能
- 如果同时开启 `shuffle=True`，会导致冲突和不可预期的行为
- `DistributedSampler` 通过 `set_epoch()` 来控制每个 epoch 的打乱顺序

**练习 3 答案：**
- `DDP(model, device_ids=[rank])` 将普通模型包装成 DDP 模型，自动处理梯度同步
- 包装后，原始模型可以通过 `model.module` 访问
- 例如：`model.module.layers` 而不是 `model.layers`

**练习 4 答案：**
- `set_epoch(epoch)` 告诉 `DistributedSampler` 当前是第几个 epoch
- 这样每个 epoch 的数据分配顺序会不同，提高训练效果
- 如果不调用，每个 epoch 的数据分配顺序都一样，影响模型泛化能力

**练习 5 答案：**
- Windows 上应该使用 `gloo` backend
- 因为 `nccl` 是 NVIDIA 为 Linux 优化的通信库，Windows 不支持
- `gloo` 是 Facebook 开发的跨平台通信库，Windows 和 Linux 都支持

</details>

---
---

<a id="sec-14"></a>
## 🌈 小结

这一节最重要的是理解：

- GPU 训练的核心是把计算和数据尽量放到同一设备上
- `to(device)` 是最常见的设备迁移方式
- `pin_memory=True` 和 `non_blocking=True` 可以帮助减少数据传输开销
- AMP 可以进一步提升 GPU 训练性能
- GPU 是否真的更快，最终要看模型大小、数据规模和数据加载效率

如果你能熟练处理设备迁移、数据传输和 AMP，PyTorch 训练效率会提升很多。



