# 🔁 PyTorch 中的典型训练循环（Training Loop）——详尽指南与实践

下面的文档目标是：把 PyTorch 中训练模型所需的**全流程**与**细节要点**讲清楚、讲透，并给出可直接运行的示例与练习题。

> **阅读建议：**
> 如果你已经会写一个最简单的训练循环，可以直接先看“完整训练/验证函数”和“关键细节与常见陷阱”；如果你想系统学习，建议从头顺着看下去。

## 📚 快速导航

- [一、训练循环的核心步骤（一句话版）](#一训练循环的核心步骤一句话版)
- [二、完整训练/验证函数（简单版，可直接运行）](#二完整训练验证函数简单版可直接运行)
- [三、关键细节与常见陷阱](#三关键细节与常见陷阱)
- [四、混合精度（AMP）训练示例](#四混合精度amp训练示例)
- [五、多 GPU 训练（简要）](#五多-gpu-训练简要)
- [六、诊断与性能分析](#六诊断与性能分析)
- [七、进阶技巧（速查）](#七进阶技巧速查)
- [八、练习题（动手实践）](#八练习题动手实践)
- [九、小结（要牢记的 10 点）](#九小结要牢记的-10-点)

## 🧭 本节速览

| 关键词 | 作用 | 你什么时候会用到 |
|---|---|---|
| `DataLoader` | 按批加载数据 | 几乎所有训练任务 |
| `model.train()` | 切换到训练模式 | 开始训练前 |
| `model.eval()` | 切换到评估模式 | 验证、测试、推理 |
| `torch.no_grad()` | 关闭梯度跟踪 | 验证/推理阶段 |
| `loss.backward()` | 反向传播 | 每个训练 batch |
| `optimizer.step()` | 更新参数 | 每个训练 batch |
| `scheduler.step()` | 调整学习率 | 按 epoch 或按 step |
| AMP | 混合精度训练 | GPU 训练加速 |
| `checkpoint` | 保存训练状态 | 断点恢复、继续训练 |

---

执行清单（本节会覆盖这些内容）：

- [x] 训练循环的核心步骤与伪代码
- [x] 完整可运行的训练/验证函数示例
- [x] 设备（CPU/GPU）与随机性控制（seed）
- [x] 保存/加载 checkpoint 与断点恢复
- [x] 学习率调度（scheduler）与早停（early stopping）策略
- [x] Mixed Precision（AMP）训练示例
- [x] 多卡训练（DataParallel 与 DistributedDataParallel）概览
- [x] 梯度累积、梯度裁剪、累积步骤示例
- [x] 日志、可视化与性能诊断建议
- [x] 练习题与参考答案

---

## 一、训练循环的核心步骤（一句话版）

训练一次模型通常包含：

1. 数据按 batch 从 DataLoader 中读取
2. 将数据移到目标设备（CPU/GPU）
3. 前向传播（model(x)）得到 logits
4. 计算损失（criterion）
5. 反向传播（loss.backward()）得到参数梯度
6. 更新参数（optimizer.step()）并清空梯度（optimizer.zero_grad()）
7. （可选）更新学习率（scheduler.step()）

> **你可以把它记成一个固定模板：**
> **取数据 → 送设备 → 前向 → 算损失 → 反向 → 更新参数 → 清梯度 → 记录日志**

伪代码：

```text
for epoch in range(num_epochs):
	for batch in train_loader:
		optimizer.zero_grad()
		outputs = model(inputs)
		loss = criterion(outputs, targets)
		loss.backward()
		optimizer.step()
	validate()
```

---

## 二、完整训练/验证函数（简单版，可直接运行）

下面的示例演示一个小型训练循环（CPU/GPU 自动选择），适用于分类任务：

### 代码结构说明

在真正看代码前，先记住这几个角色：

- `SimpleModel`：模型本体
- `train_one_epoch()`：训练一个 epoch
- `evaluate()`：在验证集上评估
- `DataLoader`：负责按 batch 提供数据
- `CrossEntropyLoss`：负责计算分类损失
- `SGD`：负责参数更新

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import SGD

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

def train_one_epoch(model, loader, optimizer, criterion, device):
	model.train()
	total_loss = 0.0
	total_samples = 0
	for x, y in loader:
		x = x.to(device)
		y = y.to(device)

		optimizer.zero_grad()
		logits = model(x)
		loss = criterion(logits, y)
		loss.backward()
		optimizer.step()

		batch_size = x.size(0)
		total_loss += loss.item() * batch_size
		total_samples += batch_size

	return total_loss / total_samples

def evaluate(model, loader, criterion, device):
	model.eval()
	total_loss = 0.0
	total_samples = 0
	correct = 0
	with torch.no_grad():
		for x, y in loader:
			x = x.to(device)
			y = y.to(device)
			logits = model(x)
			loss = criterion(logits, y)
			total_loss += loss.item() * x.size(0)
			total_samples += x.size(0)
			preds = logits.argmax(dim=1)
			correct += (preds == y).sum().item()

	return total_loss / total_samples, correct / total_samples

if __name__ == '__main__':
	# toy dataset
	x = torch.randn(200, 20)
	y = torch.randint(0, 3, (200,))
	ds = TensorDataset(x, y)
	loader = DataLoader(ds, batch_size=32, shuffle=True)

	model = SimpleModel(20, 3).to(device)
	optimizer = SGD(model.parameters(), lr=1e-2)
	criterion = nn.CrossEntropyLoss()

	for epoch in range(5):
		train_loss = train_one_epoch(model, loader, optimizer, criterion, device)
		val_loss, val_acc = evaluate(model, loader, criterion, device)
		print(f'epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}')
```

### 运行时你应该观察什么？

- `train_loss` 是否逐渐下降
- `val_loss` 是否同步下降或者至少不明显恶化
- `val_acc` 是否逐步上升
- 是否存在过拟合：训练集越来越好，验证集反而变差

---

## 三、关键细节与常见陷阱

下面是实战中经常会碰到的问题与需要注意的细节：

1. model.train() / model.eval()：
   - 在训练时调用 `model.train()`，启用 Dropout、BatchNorm 的训练行为
   - 在验证/推理时调用 `model.eval()`，并用 `torch.no_grad()` 关闭梯度记录

2. optimizer.zero_grad()：
   - 在每次反向传播前清空上一次累计的梯度
   - 推荐用 `optimizer.zero_grad()` 而不是 `for p in model.parameters(): p.grad = None`（两者各有优缺点，PyTorch 新版推荐 None 来减少内存开销）

   - 如果你追求更现代的写法，也可以了解 `optimizer.zero_grad(set_to_none=True)`，它能在某些情况下减少内存开销并加快训练。

3. loss.backward() 和 torch.autograd.grad：
   - `loss.backward()` 会把梯度累加到叶子张量的 `.grad`
   - `torch.autograd.grad()` 可以计算指定标量相对于某些张量的梯度，并返回它们，不会把结果保存到 `.grad`（除非需要）

4. 非标量 backward：
   - 只有标量张量（shape=()）可以直接 `backward()`；若是向量，需要提供 `gradient=` 或先 `.sum()` / `.mean()`

5. 梯度累积（Gradient Accumulation）：
   - 当 batch 太大导致内存不足时，可以用小 batch 累积多次梯度再 step，例如：

```python
import torch

# 下面这些对象在真实训练脚本里通常已经在上文定义好了；这里给出一个自包含的示意环境。
device = torch.device('cpu')
model = torch.nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()
loader = [(torch.randn(2, 1), torch.randn(2, 1))]

accum_steps = 4
optimizer.zero_grad()
for i, (x,y) in enumerate(loader):
	logits = model(x)
	loss = criterion(logits, y) / accum_steps
	loss.backward()
	if (i+1) % accum_steps == 0:
		optimizer.step()
		optimizer.zero_grad()
```

6. 梯度裁剪（Gradient Clipping）：
   - 防止梯度爆炸，常用于 RNN / 长序列训练：

```python
import torch

model = torch.nn.Linear(1, 1)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

7. 随机性与可复现性：
   - 设置随机种子并尽量固定环境：

```python
import random
import numpy as np
import torch

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
```

8. 数据并行/分布式训练：
   - `nn.DataParallel`（简单，但性能/可伸缩性有限）
   - `torch.nn.parallel.DistributedDataParallel`（推荐用于多 GPU）

9. Mixed Precision（AMP）：
   - 使用 `torch.cuda.amp.autocast()` 与 `GradScaler()` 可加速训练并节省显存，尤其在 Transformer、CNN 上效果明显（示例见下节）。

10. 保存/加载 checkpoint：

```python
import torch

# checkpoint 示例需要的最小上下文
epoch = 0
device = torch.device('cpu')
model = torch.nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
scaler = torch.cuda.amp.GradScaler(enabled=False)

# 保存
torch.save({
	'epoch': epoch,
	'model_state_dict': model.state_dict(),
	'optimizer_state_dict': optimizer.state_dict(),
	'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
	'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
}, 'checkpoint.pth')

# 加载
ckpt = torch.load('checkpoint.pth', map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
optimizer.load_state_dict(ckpt['optimizer_state_dict'])
if ckpt.get('scheduler_state_dict') is not None and scheduler is not None:
	scheduler.load_state_dict(ckpt['scheduler_state_dict'])
if ckpt.get('scaler_state_dict') is not None and scaler is not None:
	scaler.load_state_dict(ckpt['scaler_state_dict'])
start_epoch = ckpt.get('epoch', 0) + 1
```

---

## 四、混合精度（AMP）训练示例

使用 AMP 能在不牺牲精度的情况下加速训练并节省显存。下面给出一个典型模式：

### 典型执行顺序

1. `autocast()` 让部分算子自动使用更低精度
2. `scaler.scale(loss).backward()` 对损失做缩放，减少梯度下溢风险
3. `scaler.step(optimizer)` 执行优化器更新
4. `scaler.update()` 动态调整缩放因子

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

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

epochs = 1
train_loader = [(torch.randn(2, 20), torch.randint(0, 3, (2,)))]

model = SimpleModel(20, 3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler()

for epoch in range(epochs):
	model.train()
	for x, y in train_loader:
		x, y = x.to(device), y.to(device)
		optimizer.zero_grad()
		with autocast():
			logits = model(x)
			loss = criterion(logits, y)
		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()

	# 验证时不需要 scaler，但仍用 torch.no_grad() 和 model.eval()

```

注意：AMP 只在 CUDA 可用时有意义。

> **视觉提示：**
> 如果你的机器没有 GPU，这一段可以先理解流程；如果有 GPU，建议你实际跑一遍，感受显存和速度差异。

---

## 五、多 GPU 训练（简要）

1. DataParallel：
   - 最简单，写法：`model = nn.DataParallel(model)`，会把 batch 划分到多张 GPU
   - 缺点：单主机、单进程，可能成为瓶颈

2. DistributedDataParallel（DDP，推荐）：
   - 需要初始化进程组（`torch.distributed.init_process_group`），每个进程绑定一个 GPU
   - 使用 `DistributedSampler` 来在不同进程间划分数据
   - 更高效、扩展性好

这里只给出概念，DDP 的完整示例需要较多 boilerplate，推荐参考官方教程

> **排版小提示：**
> 如果你后面想把这一节扩成正式教程，建议单独拆成：
> `单卡训练`、`DataParallel`、`DistributedDataParallel` 三个小节，这样会更清晰。

---

## 六、诊断与性能分析

1. 使用小数据集与 tiny model 快速验证训练脚本是否正确
2. 使用 `torch.utils.tensorboard` 或 `wandb` 记录 loss/acc/learning rate
3. 用 `torch.profiler` 或 `nvprof` 分析瓶颈（数据加载、GPU 计算或内存传输）
4. 若 GPU 利用率低，检查 `num_workers`、`pin_memory`、batch_size 是否合适

---

## 七、进阶技巧（速查）

- 使用 `gradient_accumulation` 在显存受限时模拟更大的 batch
- 使用 `checkpointing`（torch.utils.checkpoint）减小中间激活占用
- 把数据从磁盘预处理到 LMDB / Memmap / HDF5 加速读取
- 使用 `WeightedRandomSampler` 处理类别不平衡
- 在训练中使用 `scheduler.step(metric)` 的早停策略

### 什么时候该优先考虑这些技巧？

| 场景 | 建议 |
|---|---|
| 显存不足 | AMP、梯度累积、checkpointing |
| 训练太慢 | 增大 batch、优化 DataLoader、AMP |
| 类别不平衡 | `WeightedRandomSampler` |
| 验证集波动大 | 早停、学习率调度 |
| I/O 成瓶颈 | `num_workers`、`pin_memory`、预处理 |

---

## 八、练习题（动手实践）

练习 1：补全训练循环

给出模型、DataLoader、criterion、optimizer，补全训练函数 `train_one_epoch`（参照上文）。

练习 2：实现模型 checkpoint 保存/加载

实现保存模型和 optimizer 状态的代码，并写出从 checkpoint 断点恢复训练的例子。

练习 3：实现混合精度训练

在上面简单训练脚本中加入 AMP（`autocast` + `GradScaler`），并比较训练速度与显存占用（如果有 GPU）。

练习 4：实现梯度累积

当 batch 太大导致 OOM 时，用梯度累积把实际等效 batch_size 扩大 4 倍。

---

## ✅ 参考答案（示例实现）

<details>
<summary>点击展开参考实现</summary>

### 练习 1 参考

见本文二节 `train_one_epoch` 函数。

### 练习 2 参考

见本文第 3 节“保存/加载 checkpoint”代码片段。

### 练习 3 参考

见本文第 4 节“混合精度（AMP）训练示例”。

### 练习 4 参考

示例：

```python
import torch

# 下面示例默认相关对象已经定义，这里补一组最小占位，方便单独阅读。
device = torch.device('cpu')
model = torch.nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()
loader = [(torch.randn(2, 1), torch.randn(2, 1))]

accum_steps = 4
optimizer.zero_grad()
for i, (x,y) in enumerate(loader):
	x,y = x.to(device), y.to(device)
	logits = model(x)
	loss = criterion(logits, y) / accum_steps
	loss.backward()
	if (i + 1) % accum_steps == 0:
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
		optimizer.step()
		optimizer.zero_grad()
```

</details>

---

## 九、小结（要牢记的 10 点）

1. 训练循环 = 前向 + 反向 + 更新
2. 在训练/验证间切换 `model.train()` / `model.eval()` 并使用 `torch.no_grad()`
3. 每次 backward 之前清零梯度；梯度默认累加
4. 使用 `DataLoader` 的 `num_workers` 与 `pin_memory` 提升 I/O 性能
5. 对大模型考虑 AMP、梯度累积、checkpointing
6. 使用合适的 scheduler 与早停策略提升训练质量
7. 保存完整 checkpoint（model + optimizer + scheduler + scaler + epoch）以便重启
8. 在多卡场景使用 DDP 并结合 `DistributedSampler`
9. 记录训练日志（TensorBoard / wandb）并分析训练曲线
10. 写小脚本快速验证训练逻辑，再放大规模训练

祝你训练顺利！如果需要，我可以把这些示例自动抽成 `examples/` 下的可运行脚本，并生成一个小 README 指南，告诉你如何在 CPU/GPU 上运行它们。

---

## ✨ 额外阅读建议

如果你想继续把训练循环学得更扎实，可以按这个顺序继续补：

1. `A.6_设置高效的数据加载器.md`：先把数据喂进模型的方式彻底搞懂
2. `A.5_实现多层神经网络.md`：再看模型是怎么搭起来的
3. `A.4_轻松实现自动微分.md`：理解梯度为什么能被算出来
4. 回到本节：把完整训练循环串起来理解



