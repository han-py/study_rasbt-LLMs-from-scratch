# 🧱 PyTorch 核心基石：深入理解张量 (Tensor) 完全指南

> **💡 小白通俗解析：什么是张量 (Tensor)？**
> 想象一下你在管理一个巨型仓库：
> - **标量（0维，Scalar）** 就像是一个单一的包裹（比如一个数字 `7`）。
> - **向量（1维，Vector）** 就像是一排包裹（比如一列数字 `[1, 2, 3]`）。
> - **矩阵（2维，Matrix）** 就像是一个有很多层、很多列的货架。
> - **多维张量（3维+，Tensor）** 就像是一整个仓库，甚至多个仓库的集合！
>
> **一句话总结**：张量就是带有维度的数字集合，它是 PyTorch 中传递数据、计算参数的**通用集装箱**。张量极其类似 NumPy 数组，但它的杀手锏在于：**可以放到 GPU 上实现极其夸张的加速运算**，并支持自动求导（这对于训练大模型来说是必须的）。

---

## 📚 目录

- [一、从零开始创建张量](#一从零开始创建张量)
- [二、查户口：张量的三大核心属性](#二查户口张量的三大核心属性)
- [三、张量变形记：形状变换与维度控制](#三张量变形记形状变换与维度控制)
- [四、张量的数学魔法](#四张量的数学魔法)
- [五、课后实战测试（检验学习成果）](#五课后实战测试检验学习成果)

---

## 🌟 一、从零开始创建张量

在使用张量之前，首先要导入 PyTorch。

```python
import torch
import numpy as np
```

### 1. 手动创建不同维度的张量

```python
# 📦 0维张量（标量 - Scalar）
scalar = torch.tensor(7)
print("标量:", scalar, " | 维度:", scalar.ndim)

# 📦 1维张量（向量 - Vector）
vector = torch.tensor([7, 7])
print("向量:", vector, " | 维度:", vector.ndim)

# 📦 2维张量（矩阵 - Matrix）
matrix = torch.tensor([[1, 2], 
                       [3, 4]])
print("矩阵:\n", matrix, "\n维度:", matrix.ndim)
```

### 2. 使用内置函数快速生成张量 (超级常用！)

很多时候我们不需要手动输入数字，而是需要 PyTorch 帮我们快速生成特定形状的张量（比如全是0，全是随机数等）。

```python
# 🎲 生成 3x4 的随机浮点数张量（范围在 0 到 1 之间）
# 就像给神经网络初始化随机的权重
rand_tensor = torch.rand(size=(3, 4))

# 🍩 生成 2x2 的全零张量
zero_tensor = torch.zeros(size=(2, 2))

# 🧊 生成包含一段连续数字的张量（类似 python 的 range）
range_tensor = torch.arange(start=0, end=10, step=1) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# 🌉 从 NumPy 数组转换，无缝对接！
np_array = np.array([1, 2, 3])
tensor_from_np = torch.from_numpy(np_array)
```

---

## 🌟 二、查户口：张量的三大核心属性

当代码报错时，90%的情况下是张量的这三个属性不对！一定要熟记：
1. **Shape（形状）**：张量的各个维度大小是什么？（比如 3行4列）
2. **Dtype（数据类型）**：张量里装的是什么数据？（如 32位浮点数 还是 64位整数）
3. **Device（设备）**：张量存放在哪里？（是在 CPU 上还是在 GPU 上？）

```python
some_tensor = torch.rand(3, 4)

print(f"形状 (Shape): {some_tensor.shape}")      # 输出: torch.Size([3, 4])
print(f"数据类型 (Data type): {some_tensor.dtype}") # 输出: torch.float32
print(f"所处设备 (Device): {some_tensor.device}")   # 流行的设备名称: 'cpu' 或 'cuda' (Nvidia GPU)
```

---

## 🌟 三、张量变形记：形状变换与维度控制

在大模型中，数据从一层传递到另一层时，经常需要**改变形状**以匹配下一个层的输入要求。

### 1. 基本的 Reshape / View
```python
x = torch.arange(1, 13) # [1, 2, ..., 12] ，形状是 [12]

# 我们可以把它重新塑形成 3行 4列 的矩阵
reshaped_x = x.reshape(3, 4)
# 也可以塑形成 4行 3列
viewed_x = x.view(4, 3)

print(reshaped_x)
```
> 💡 **进阶提示**：`view` 要求内存地址连续，而 `reshape` 更像万金油。平时“无脑”用 `reshape` 即可。

### 2. 添加或移除维度 (Squeeze / Unsqueeze)
这在深度学习中非常常见。假设你需要送入模型的照片要求是 `[批次大小, 色彩通道, 高, 宽]`，即使只有一张照片，你也必须给它硬塞一个“批次”维度进去。

```python
tensor = torch.tensor([1, 2, 3]) # 当前形状是 [3]
print(tensor.shape)

# 🎈 unsqueeze(0)：在索引0的位置“强行挤进去”一个新的维度
tensor_unsqueeze = tensor.unsqueeze(dim=0) 
print(tensor_unsqueeze.shape) # 变成 [1, 3]

# 🎈 squeeze()：把所有维度大小为 1 的“空洞”维度全部挤出去
tensor_squeeze = tensor_unsqueeze.squeeze()
print(tensor_squeeze.shape) # 又变回 [3]
```

---

## 🌟 四、张量的数学魔法

### 1. 元素级运算 (Element-wise)
逐个元素相加/相乘，要求形状必须匹配（或者满足广播机制）。

```python
tensor1 = torch.tensor([1, 2, 3])
tensor2 = torch.tensor([10, 20, 30])

print(tensor1 + tensor2) # [11, 22, 33]
print(tensor1 * tensor2) # [10, 40, 90]
```

### 2. 高阶奥义：矩阵乘法 (Matrix Multiplication) ⚔️
大模型底层（例如自注意力机制中的 Q·K^T 甚至前馈网络）绝大多数的核心计算都是矩阵乘法。
矩阵乘法的硬性规则：**内部维度必须匹配！**
- `(3, 2)` 的矩阵 乘以 `(2, 4)` 的矩阵 = `(3, 4)` 的矩阵。（中间那个 2 必须一样）。

```python
mat1 = torch.rand(3, 2)
mat2 = torch.rand(2, 4)

# 矩阵乘法的三种写法（结果完全一样），推荐最后一种 "@" 符号：
result1 = torch.matmul(mat1, mat2)
result2 = torch.mm(mat1, mat2)  # mm 是 matmul 的简写，但只能算 2 维矩阵
result3 = mat1 @ mat2           # Python 专属语法糖，极度优雅 ✨

print(result3.shape) # 会输出 torch.Size([3, 4])
```

---

## 🎯 五、🧠 课后实战测试（检验学习成果！）

是时候动动手了！请尝试在旁边新建一块 Jupyter Notebook 或者 Python 脚本，完成以下任务。如果有困难，可以点击下方的隐藏答案。

### 📝 练习题提纲

**题目 1：创建与属性**
- 创建一个形状为 `(7, 7)` 的全随机数张量 `tensor_A`。
- 打印出这个张量的**形状**与**数据类型**。

**题目 2：强大的矩阵乘法**
- 创建一个形状为 `(1, 7)` 的全随机数张量 `tensor_B`。
- 尝试将 `tensor_B` 与上面的 `tensor_A` 执行矩阵相乘，将结果命名为 `tensor_C`。
- 预测一下：`tensor_C` 的形状会是什么？

**题目 3：形状魔法**
- 操作刚才得到的 `tensor_C`。现在要求你不改变其数据排列，仅仅把它强制转换为 `(7, 1)` 的形状（这相当于对一维向量求转置的效果）。

**题目 4：极速切换 GPU（附加挑战）**
- 检查你的电脑/运行环境是否支持 CUDA（GPU加速），如果支持，将 `tensor_C` 传送到 GPU 上运行。（如果不支持，打印 "只支持 CPU"）。

---

<details>
<summary>✅ <b>点击这里展开查看参考答案！</b></summary>
<br>

```python
import torch

# ============= 题目 1 =============
tensor_A = torch.rand(7, 7)
print("题1 shape:", tensor_A.shape)
print("题1 dtype:", tensor_A.dtype)

# ============= 题目 2 =============
tensor_B = torch.rand(1, 7)
# (1, 7) 乘以 (7, 7) -> 内部维度 7 抵消，结果是 (1, 7)
tensor_C = tensor_B @ tensor_A
print("题2 tensor_C 形状:", tensor_C.shape) 

# ============= 题目 3 =============
# 将 (1, 7) 变成 (7, 1)
tensor_C_reshaped = tensor_C.reshape(7, 1)

# 另外一种快速办法是直接调用转置： 
# tensor_C_transposed = tensor_C.T 

print("题3 变身后的形状:", tensor_C_reshaped.shape)

# ============= 题目 4 =============
if torch.cuda.is_available():
    tensor_C_gpu = tensor_C.to("cuda")
    print("题4: 成功转移阵地到 GPU 啦！现在的设备是:", tensor_C_gpu.device)
else:
    print("题4: 你当前的设备没有装配可用的 Nvidia GPU 环境，所以只支持 CPU 哦~")
```
</details>

> 🎉 **恭喜你！**
> 掌握了这些张量的基本操作，你已经跨越了 PyTorch 学习之路的极其重要的一道门槛！在大语言模型学习中，无论是整理输入模型的文本 Token 数据，还是处理注意力机制的 Q、K、V 矩阵，本质上永远都是在折腾这些“张量”！

