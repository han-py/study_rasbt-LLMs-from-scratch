# 🚚 PyTorch 中如何设置高效的数据加载器（DataLoader）——超详细入门指南

> **一句话理解数据加载器：**
> 数据加载器就是把“原始数据”按你想要的方式，**自动切成一批一批（batch）**，并且**高效地送进模型** 的工具。
>
> 如果把训练神经网络比作做饭：
> - **Dataset** 像是你的食材仓库
> - **DataLoader** 像是负责把食材分批、洗好、切好、按顺序送到厨房的帮手
> - **batch** 就是一盘一盘端上来的菜
> - **shuffle** 就像把食材打乱再取，避免模型“记顺序”

---

## 📚 目录

- [一、为什么需要高效的数据加载器？](#一为什么需要高效的数据加载器)
- [二、Dataset 和 DataLoader 的区别](#二dataset-和-dataloader-的区别)
- [三、最简单的 Dataset：TensorDataset](#三最简单的-datasettensordataset)
- [四、DataLoader 的基本用法](#四dataloader-的基本用法)
- [五、batch_size 是什么？](#五batch_size-是什么)
- [六、shuffle=True 是什么？](#六shuffletrue-是什么)
- [七、drop_last=True 是什么？](#七drop_lasttrue-是什么)
- [八、num_workers 是什么？](#八num_workers-是什么)
- [九、pin_memory=True 是什么？](#九pin_memorytrue-是什么)
- [十、自定义 Dataset：最常见写法](#十自定义-dataset最常见写法)
- [十一、完整示例：自己写一个回归数据集](#十一完整示例自己写一个回归数据集)
- [十二、DataLoader 返回的是什么？](#十二dataloader-返回的是什么)
- [十三、批处理（batch）为什么重要？](#十三批处理batch为什么重要)
- [十四、DataLoader 在训练循环里的位置](#十四dataloader-在训练循环里的位置)
- [十五、collate_fn：自定义批处理方式](#十五collate_fn自定义批处理方式)
- [十六、一个变长文本的小例子](#十六一个变长文本的小例子)
- [十七、sampler 和 batch_sampler 的概念](#十七sampler-和-batch_sampler-的概念)
- [十八、性能优化技巧](#十八性能优化技巧)
- [十九、常见错误和小提醒](#十九常见错误和小提醒)
- [二十、练习题：巩固数据加载器](#二十练习题巩固数据加载器)
- [小结](#小结)

---

## 🌟 一、为什么需要高效的数据加载器？

如果我们把全部数据一次性塞给模型，会遇到很多问题：

1. **内存不够**：大数据集可能放不下
2. **训练不稳定**：一次看全部数据不利于梯度更新
3. **速度不理想**：数据读取、预处理、传输会拖慢训练
4. **不方便打乱顺序**：模型可能学到“数据排列顺序”而不是模式

因此，在 PyTorch 中通常不会直接把整个数据集一次性送入模型，而是使用：

- `Dataset`：负责“定义数据是什么”
- `DataLoader`：负责“怎么按批读取数据、怎么打乱、怎么并行加载”

---

## 🌟 二、Dataset 和 DataLoader 的区别

### 1. `Dataset`：数据的抽象
`Dataset` 负责告诉 PyTorch：
- 一共有多少条数据
- 每条数据如何取出
- 每条数据长什么样

你可以把它理解成“数据说明书”。

### 2. `DataLoader`：数据读取的执行者
`DataLoader` 负责：
- 按 batch 读取
- 是否打乱顺序
- 是否使用多个进程并行读取
- 是否丢弃最后一个不完整 batch
- 是否使用更快的内存拷贝方式

你可以把它理解成“数据配送中心”。

---

## 🌟 三、最简单的 Dataset：`TensorDataset`

如果你的数据已经是张量格式，可以直接用 `TensorDataset`。

```python
import torch
from torch.utils.data import TensorDataset

x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

dataset = TensorDataset(x, y)
print(len(dataset))
print(dataset[0])
```

### 这里发生了什么？

- `x` 是输入特征
- `y` 是标签
- `TensorDataset(x, y)` 会把它们按行一一对应起来

取 `dataset[0]` 时，会返回：

```python
(tensor([1.]), tensor([2.]))
```

---

## 🌟 四、DataLoader 的基本用法

有了 `Dataset` 以后，就可以交给 `DataLoader` 了。

```python
import torch
from torch.utils.data import TensorDataset, DataLoader

x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=2, shuffle=False)

for batch_x, batch_y in loader:
	print(batch_x)
	print(batch_y)
```

### 输出会是什么？

会按 batch_size=2 分成两批：

1. 第一批：`[[1.0], [2.0]]`
2. 第二批：`[[3.0], [4.0]]`

---

## 🌟 五、`batch_size` 是什么？

`batch_size` 表示一次送入模型多少条样本。

### 为什么不一次只送 1 条？
- 太慢
- 梯度波动大

### 为什么不一次送全部？
- 内存压力大
- 更新不灵活

所以通常选一个折中值，比如：
- 16
- 32
- 64
- 128

---

## 🌟 六、`shuffle=True` 是什么？

`shuffle=True` 表示在每个 epoch 开始前，把数据顺序随机打乱。

```python
loader = DataLoader(dataset, batch_size=2, shuffle=True)
```

### 为什么要打乱？

如果数据是按类别排好的，比如：
- 前 100 条全是猫
- 后 100 条全是狗

那模型在训练时就会先看到一大堆猫，再看到一大堆狗，这样训练会不稳定。

打乱后，模型每次看到的样本分布更均匀。

---

## 🌟 七、`drop_last=True` 是什么？

有时数据总数不能被 batch_size 整除。

比如：
- 总共有 10 条数据
- batch_size = 4

那么会得到：
- 4 条
- 4 条
- 2 条

最后这个 2 条的 batch 就是不完整的。

如果设置：

```python
drop_last=True
```

那么最后不完整的 batch 会被直接丢掉。

### 什么时候常用？
- 某些训练策略要求 batch 大小完全一致
- 多卡训练中有时希望 batch 尺寸固定

---

## 🌟 八、`num_workers` 是什么？

`num_workers` 表示有多少个子进程并行加载数据。

```python
loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)
```

### 它的作用

- 提高数据读取速度
- 让 CPU 在后台预处理数据
- 减少 GPU 等待数据的时间

### 但要注意

- 在 Windows 下，有时 `num_workers>0` 需要特别注意入口保护 `if __name__ == "__main__":`
- 在非常小的数据集上，开太多 workers 反而可能更慢

---

## 🌟 九、`pin_memory=True` 是什么？

如果你使用 GPU，`pin_memory=True` 可以帮助 CPU 内存更快地拷贝到 GPU。

```python
loader = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=True)
```

### 什么时候有帮助？
- 训练在 GPU 上进行时
- 数据要频繁从 CPU 传到 GPU 时

---

## 🌟 十、自定义 Dataset：最常见写法

很多时候数据不是已经整理好的张量，而是：
- 图片文件
- 文本文件
- CSV 表格
- 音频文件

这时就要自己写 `Dataset`。

### 典型写法

```python
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def __len__(self):
		return len(self.x)

	def __getitem__(self, idx):
		return self.x[idx], self.y[idx]
```

### 三个核心方法

#### 1. `__init__`
负责把数据存起来。

#### 2. `__len__`
告诉 PyTorch 这个数据集有多少条数据。

#### 3. `__getitem__`
告诉 PyTorch 如何按索引取出一条数据。

---

## 🌟 十一、完整示例：自己写一个回归数据集

下面我们构造一个简单任务：

> 输入 `x`，目标 `y = 2x + 1`

```python
import torch
from torch.utils.data import Dataset, DataLoader

class RegressionDataset(Dataset):
	def __init__(self):
		self.x = torch.arange(1, 11, dtype=torch.float32).unsqueeze(1)
		self.y = 2 * self.x + 1

	def __len__(self):
		return len(self.x)

	def __getitem__(self, idx):
		return self.x[idx], self.y[idx]

dataset = RegressionDataset()
loader = DataLoader(dataset, batch_size=3, shuffle=False)

for batch_x, batch_y in loader:
	print("x:", batch_x.squeeze().tolist(), "y:", batch_y.squeeze().tolist())
```

### 你会看到什么？

数据会被分成若干批，每批 3 条左右：
- 第一批：1, 2, 3
- 第二批：4, 5, 6
- 第三批：7, 8, 9
- 第四批：10

---

## 🌟 十二、DataLoader 返回的是什么？

DataLoader 每次迭代时，返回的是一个 batch。

对于监督学习任务，通常返回：

```python
batch_x, batch_y
```

如果你的 `Dataset` 返回的是：

```python
return feature, label, extra_info
```

那么 DataLoader 也会按 batch 把这几个部分一起打包回来。

---

## 🌟 十三、批处理（batch）为什么重要？

### 1. 更高效
一次处理一批数据，比一条一条快很多。

### 2. 梯度更稳定
小批量梯度比单样本梯度更平滑。

### 3. 更适合 GPU
GPU 喜欢并行计算，batch 可以更好地利用硬件。

---

## 🌟 十四、DataLoader 在训练循环里的位置

一个标准训练流程通常是：

```python
for epoch in range(num_epochs):
	for batch_x, batch_y in loader:
		optimizer.zero_grad()
		logits = model(batch_x)
		loss = criterion(logits, batch_y)
		loss.backward()
		optimizer.step()
```

### 这里 DataLoader 负责什么？

- 提供 `batch_x`
- 提供 `batch_y`
- 每轮都可重新打乱数据

---

## 🌟 十五、`collate_fn`：自定义批处理方式

有些数据条目长度不一样，比如文本：
- 第一句 5 个词
- 第二句 12 个词
- 第三句 8 个词

这时不能直接简单堆叠成一个规则张量，需要先补齐长度。

### `collate_fn` 的作用

`collate_fn` 允许你自定义“如何把一堆样本拼成一个 batch”。

### 典型用途
- 文本 padding
- 变长序列处理
- 图结构数据批处理
- 多输入多输出样本组合

### 示例：对变长文本做 padding

```python
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

def collate_fn(batch):
	sequences = [item[0] for item in batch]
	labels = torch.tensor([item[1] for item in batch])
	padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
	return padded_sequences, labels
```

---

## 🌟 十六、一个变长文本的小例子

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class TextDataset(Dataset):
	def __init__(self):
		self.samples = [
			(torch.tensor([1, 2, 3]), 0),
			(torch.tensor([4, 5]), 1),
			(torch.tensor([6, 7, 8, 9]), 0)
		]

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		return self.samples[idx]

def collate_fn(batch):
	seqs = [item[0] for item in batch]
	labels = torch.tensor([item[1] for item in batch])
	padded = pad_sequence(seqs, batch_first=True, padding_value=0)
	return padded, labels

dataset = TextDataset()
loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

for x, y in loader:
	print(x)
	print(y)
```

### 这个例子说明了什么？

- 文本长度不同
- 不能直接堆成规则矩阵
- 需要 `collate_fn` 在 batch 级别统一处理

---

## 🌟 十七、`sampler` 和 `batch_sampler` 的概念

除了 `shuffle=True` 之外，PyTorch 还允许你更细致地控制取样方式。

### `sampler`
决定“取哪些样本、按什么顺序取”。

### `batch_sampler`
决定“每一批由哪些样本组成”。

### 常见 sampler
- `SequentialSampler`
- `RandomSampler`
- `WeightedRandomSampler`

这些在类别不平衡、特殊训练策略里很有用。

---

## 🌟 十八、性能优化技巧

### 1. 预先把数据整理成张量
如果数据集很小，直接用 `TensorDataset` 很方便。

### 2. 合理设置 `batch_size`
过小浪费 GPU，过大容易爆内存。

### 3. 合理设置 `num_workers`
根据系统和数据规模尝试调整。

### 4. GPU 场景下使用 `pin_memory=True`
有助于更快的数据传输。

### 5. 用 `collate_fn` 做批级别预处理
比如 padding、截断、过滤非法样本。

---

## 🌟 十九、常见错误和小提醒

### 1. `__getitem__` 返回类型不一致
会导致 DataLoader 拼 batch 时出错。

### 2. 变长数据没有处理好
直接堆叠会报错。

### 3. `num_workers` 设置不当
在 Windows 下尤其要注意入口保护。

### 4. 忘记把数据转成合适 dtype
比如分类标签一般要是整数类型。

### 5. `shuffle` 和 `sampler` 混用
很多场景下二者不能随意同时指定。

---

## 🎯 二十、练习题：巩固数据加载器

### 练习 1：构造一个 TensorDataset
请用 `TensorDataset` 保存下面两个张量，并用 `DataLoader` 按 batch_size=2 读取。

```python
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])
```

**问题：** 读取时会输出几批？每批包含什么？

---

### 练习 2：自定义 Dataset
写一个 `Dataset`，让它返回：

```python
(x[idx], 2 * x[idx] + 1)
```

**问题：** 它需要实现哪三个方法？

---

### 练习 3：理解 shuffle
把 `shuffle=False` 和 `shuffle=True` 分别设置到同一个 DataLoader 上，观察输出顺序变化。

**问题：** 为什么训练时通常要打乱数据？

---

### 练习 4：实现 collate_fn
给定几条长度不同的序列，用 `pad_sequence` 写出一个 `collate_fn`。

**问题：** 为什么文本任务经常需要 `collate_fn`？

---

## ✅ 参考答案

<details>
<summary>点击展开答案</summary>

### 练习 1 参考答案

```python
import torch
from torch.utils.data import TensorDataset, DataLoader

x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=2, shuffle=False)

for batch_x, batch_y in loader:
	print(batch_x, batch_y)
```

会输出 2 批：
- 第一批：`(1,2)`、`(2,4)`
- 第二批：`(3,6)`、`(4,8)`

### 练习 2 参考答案

需要实现：
- `__init__`
- `__len__`
- `__getitem__`

### 练习 3 参考答案

`shuffle=True` 会在每个 epoch 打乱样本顺序，避免模型记住固定顺序。

### 练习 4 参考答案

```python
import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
	seqs = [item[0] for item in batch]
	labels = torch.tensor([item[1] for item in batch])
	padded = pad_sequence(seqs, batch_first=True, padding_value=0)
	return padded, labels
```

</details>

---

## 🌈 小结

这一节最重要的是理解：

- `Dataset` 描述数据本身
- `DataLoader` 负责高效按批读取数据
- `batch_size`、`shuffle`、`drop_last`、`num_workers`、`pin_memory` 都会影响训练效率和结果
- 对于变长数据，可以使用 `collate_fn`
- 在真实训练循环中，DataLoader 是连接数据和模型的桥梁

如果你能熟练使用 `Dataset` 和 `DataLoader`，后面无论是图像、文本还是表格数据，都能更顺畅地进入模型训练流程。

