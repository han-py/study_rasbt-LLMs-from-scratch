# =====================================================================
# 🐰 【小白导读：代码清单 7-1 下载数据集】
# 我们要去教 AI 听懂指令，首先得有“教材”。
# 这个代码块的作用是去网上下载一份名为 instruction-data.json 的文件。
# 它里面包含了成千上万条“提问+要求+答案”的练习题，下载好后存入电脑内存中。
# =====================================================================
# 代码清单 7-1 下载数据集
import json
import os
import urllib

from openpyxl.formula import tokenizer
from statsmodels.tsa.arima import params
from torch.onnx.symbolic_opset9 import tensor


def download_and_load_file(file_path, url):
    """
    🐰 【小白通俗解析此函数】
    作用：跑腿下载员
    工作流程：代码会先看看你电脑里有没有我们要的题库文件 (file_path)。
    如果没有，就顺着网线 (url) 跑过去下载拿回来，塞进一个 JSON（文本字典）里，最后把整个题目数据端交给你。
    """
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            test_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(test_data)
    with open (file_path, "r") as file:
        data = json.load(file)
    return  data

file_path = "instruction-data.json"
url = (
    "http://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)

data = download_and_load_file(file_path, url)
# print("Number of entries:", len(data))
#
# print("Example entry:\n", data[50])
# print("Another example entry:\n", data[999])


# 代码清单 7-2 实现提示词格式函数、
def format_input(entry):
    """
    🐰 【小白通俗解析此函数】
    作用：扮演一个“剧本组装机”。
    AI 是缺乏默认前置语境的，你要让它做题，得给它立个人设。
    这个函数把下载来的原始题目字典 (entry) 拼凑成一段格式极为工整的固定多段式长话，
    里面明确用 ### Instruction: 等标签标注好哪里是问题、哪里是补充材料，方便 AI 阅读时不看岔劈。
    """
    # 这段话的作用是给 AI 强行加戏，赋予它一个角色设定：
    # “下面是一个描述任务的指令。请写出一个恰当的回复来完成这个请求。”
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    # 如果用户不仅给了指令，还给了一段要处理的内容 (比如“把这段话翻译成英文”，下面紧接着你要翻译的话)
    # 那就把这段话也拼接到输入里，否则就空着。
    input_text = (
        f"\n\n### Input:\n{entry['input']}" if entry["input"] else  ""
    )
    # 最后把设定、指令和具体内容像积木一样拼成一段完整的话返回
    return instruction_text + input_text
"""
带格式的输入就像下面这样：
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Identify the correct spelling of the following word.

### Input:
Occassion

### Response:
The correct spelling is 'Occasion.'
"""

model_input = format_input(data[999])
desired_response = f"\n\n### Response:\n{data[999]['output']}"
# print(model_input + desired_response)


# 代码清单 7-3 划分数据集
# 假设我们下载了 1000 道题，这里就是计算我们要按什么比例把它分给不同的阶段
train_portion = int(len(data) * 0.85) # 计算出 85% 的题量作为训练集。比如 850 题，让 AI 看着标准答案反复练习
test_portion = int(len(data) * 0.1) # 计算出 10% 的题量作为测试集。保留绝对神秘感，等彻底学完再拿来摸底考试
val_portion = len(data) - train_portion - test_portion # 剩下 5% 的题量作为验证集。在平时练习的间隙抽考一下，看看是不是学偏了

# 利用 Python 的切片（[:xx]）把一份大数据一切为三
train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]

# print("Training set length:", len(train_data))
# print("Validation set length:", len(val_data))
# print("Test set length:", len(test_data))


# 代码清单 7-4 实现一个指令数据集类
import torch
from torch.utils.data import Dataset

class InstructionDataset(Dataset):
    """
    🐰 【小白通俗解析此类】
    作用：大模型的“题库管家”。
    PyTorch 必须要通过特制的管家才能发卷子。这个管家的主要任务是：
    在开始阶段就提前把文本题库彻底“数字化”(tokenize)，让所有汉字和字母变成数字 ID，随时等着被抽调。
    """
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = [] # 这是一个空箱子，用来存放翻译好的“数字密码表”
        for entry in data:
            # 1. 先把指令和输入拼出完整的题面
            instruction_plus_input = format_input(entry) # 预词元化文本
            # 2. 再把标准答案拼接上去，这里我们加上 ### Response: 作为分割线，告诉AI后续是答案
            response_text = f"\n\n### Response:\n{entry['output']}"
            # 3. 把题面和答案拼成极其长的一段自然语言
            full_text = instruction_plus_input + response_text
            # 4. 最后，把这段文字喂给分词器(tokenizer)，将所有汉字/英文转化成数字编号列表，装进箱子里！
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        # 只要 DataLoader 想抽某一道考题，就会触发这个方法，把对应序号的数字密码题提取出去
        return self.encoded_texts[ index]

    def __len__(self):
        # 告诉别人这个资料库里面一共存了多少道大题
        return len(self.data)

import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
# print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

def custom_collate_draft_1(
        batch,
        pad_token_id=50256,
        device="cpu"
):
    """
    🐰 【小白通俗解析此草稿函数1】
    作用：探路草稿，理解“截长补短”的第一步。
    模型想要方方正正的矩阵，如果有几个长短不一的句子混在一起，它就报错。
    这里展示了如何找最长序列，然后用 50256(特殊空格) 把短序列全填补到一样长。
    """
    batch_max_length = max(len(item)+1 for item in batch) # 找到批次中最长的序列
    inputs_lst = []

    for item in batch: # 填充并准备输入
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1]) # 删除之前添加的额外填充词元
        inputs_lst.append(inputs)

    inputs_tensor = torch.stack(inputs_lst).to(device) # 输入列表变成一个张量并转移到目标设备
    return inputs_tensor

inputs_1 = [0, 1, 2, 3, 4]
inputs_2 = [5, 6]
inputs_3 = [7, 8, 9]
batch = (
    inputs_1,
    inputs_2,
    inputs_3
)
# print(custom_collate_draft_1(batch))

def custom_collate_draft_2(
        batch,
        pad_token_id=50256,
        device="cpu"
):
    """
    🐰 【小白通俗解析此草稿函数2】
    作用：探路草稿，理解“题目和答案怎么错位”。
    训练模型就像接龙，输入是“天 空 是”，目标答案就是“空 是 蓝”。把整体向左推移一个字表，从而形成 targets。
    """
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1]) # 截断输入的最后一个词元
        targets = torch.tensor(padded[1:]) # 向左易懂一个位置得到目标
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

inputs, targets = custom_collate_draft_2(batch)
# print(inputs)
# print(targets)


# 代码清单 7-5 实现一个自定义的批聚合函数
def custom_collate_fn(
        batch,
        pad_token_id=50256, # 50256 这个数字在 GPT-2 字典里代表特殊的“填充符/空白符”
        ignore_index=-100,  # 这个值很重要！PyTorch 算分时一旦看到 -100，就不加分也不扣分。我们不想让 AI 浪费脑力去记忆怎么生成填充符。
        allowed_max_length=None,
        device="cpu"
):
    """
    🐰 【小白通俗解析此函数】
    作用：完美的究极“发卷打包员”！(Collate的含义就是归纳整理)
    这是训练任务最关键的处理卡口！它不仅做到了长短句自动补齐、一键打乱错开得到目标数组，
    还非常聪明地把答案区的无用空格 (50256) 全替成了“免死金牌” (-100)。
    有了它，长短各异的人类语言才能平安变成模型能吃下去的方正矩阵。
    """
    # 找到发下来的这一批考卷里，最长的那一种有多长？
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = ( # 如果某个卷子太短了，就在它屁股后面疯狂填充 50256（补空白），直到逼齐最长的那个卷子
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1]) # 切掉最后一个词，因为输入的作用是用前文猜后文，最后一词没必要当输入
        targets = torch.tensor(padded[1:]) # 切掉第一个词，整体向左挪一位，变成答案（即告诉AI，看到前面的字，你应该输出这个字）

        # 屏蔽填充词元，也就是把用来补白的 50256 通通换成免责金牌 -100
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            # 万一卷子真的长得离谱，强行截断，以免把显卡内存撑爆
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append( inputs)
        targets_lst.append( targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

inputs, targets = custom_collate_fn(batch)
# print(inputs)
# print(targets)

logits_1 = torch.tensor(
    [
        [-1.0, 1.0], # 第一个词元的预测
        [-0.5, 1.5] # 第二个词元的预测
    ]
)
targets_1 = torch.tensor([0, 1]) # 要生成的正确词元索引
loss_1 = torch.nn.functional.cross_entropy(logits_1, targets_1)
# print(loss_1)

logits_2 = torch.tensor(
    [
        [-1.0, 1.0],
        [-0.5, 1.5],
        [-0.5, 1.5] # 新的第三个词元的预测
    ]
)
targets_2 = torch.tensor([0, 1, 1])
loss_2 = torch.nn.functional.cross_entropy(logits_2, targets_2)
# print(loss_2)

targets_3 = torch.tensor([0, 1, -100])
loss_3 = torch.nn.functional.cross_entropy(logits_2, targets_3)
# print(loss_3)
# print("loss_1 == loss_3:", loss_1 == loss_3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 取消注释这两行就可以在 Apple Silicon 芯片上使用GPU
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
print("Device:",  device)

from functools import partial

customized_collate_fn = partial(
    custom_collate_fn,
    device=device,
    allowed_max_length=1024
)


# 代码清单 7-6 初始化数据加载器
from torch.utils.data import DataLoader

num_workers = 0 # 如果你的操作系统支持 Python 进程的并行，那么可以加大这个数值
batch_size = 8

torch.manual_seed(123)

train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers,
)

val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
)

test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
)

# print("Train loader:")
# for inputs, targets in train_loader:
#     print(inputs.shape, targets.shape)


# 代码清单 7-7 加载预训练模型
from gpt_download import download_and_load_gpt2

import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """
    🐰 【小白通俗解析此类】
    这就是轰动AI界的 Transformer 的核心灵魂 —— 多头注意力机制（Self-Attention）！
    作用：让一句话里的每一个词，都去跟句子里的其他词“眉目传情”。
    比如“苹果公司”和“吃苹果”，普通的词典不知道苹果区别。但通过计算这里面的 Query、Key、Value，
    模型能算出各自词的注意力得分，从而明白前一个是公司，后一个是水果。
    为什么叫“多头”？相当于请了几个不同的员工：男员工关注句子主谓宾，女员工关注感情色彩，分工合作更全面。
    """
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec

class LayerNorm(nn.Module):
    """
    🐰 【小白通俗解析此类】
    作用：层归一化（Layer Normalization），也就是大模型的“情绪稳定器”。
    如果不加这层，数据在几百层神经过山车里来回乘法，结果可能几万亿或者接近零。
    这个开关能把过激的数值强制拉回到平稳标准区间，保证模型不会学到走火入魔。
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        var = x.var(dim = -1, keepdim = True, unbiased = False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    """
    🐰 【小白通俗解析此类】
    作用：GELU 激活函数，大模型的“大脑突触开关”。
    单纯的矩阵相乘（线性代数）是没法理解人类复杂逻辑的。
    GELU 给原本直来直去的数学公式加入了一道平滑优美的“非线性弯道曲线”。有了它，AI才真正有了开窍悟性。
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):
    """
    🐰 【小白通俗解析此类】
    作用：前馈神经网络（FFN），被称为模型参数里的“核心知识存储器”。
    刚刚在注意力层，词和词互相“交流”完感情。现在，带着交流感情结果的词来到前馈层进行“内化和发散”。
    在这里先把维度急剧放大 4 倍去发散思考，然后再重新收缩回来。模型学到的各种理科文科知识大多存在这层里。
    """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], cfg["emb_dim"] * 4),
            GELU(),
            nn.Linear(cfg["emb_dim"] * 4, cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    """
    🐰 【小白通俗解析此类】
    作用：构建百层模型大楼的基础单位 —— “乐高主板”。
    把上面提到的各种高科技组件组装成一个标准流水线：
    词语进来 -> 进调节器(Norm) -> 进行词语互动(多头注意力) -> 捷径加回原参数(防止学忘)
    -> 再进调节器(Norm) -> 进知识库扩容(前馈网络) -> 再次加回原参数 -> 出去。
    一层跑完，GPT 会把这块电路板复制重叠 N 层！
    """
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            context_length = cfg["context_length"],
            num_heads = cfg["n_heads"],
            dropout = cfg["drop_rate"],
            qkv_bias = cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # 在注意力块中添加快捷连接
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = shortcut + x  # 将原始输入添加回来

        # 在前馈层中添加快捷链接
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = shortcut + x
        return x

import numpy as np

class GPTModel(nn.Module):
    """
    🐰 【小白通俗解析此类】
    作用：将所有积木组装拼接，落成宏伟的 GPT 大模型！
    一句话怎么走完全程？
    1. 你给它几个数字密码（Token IDs）。
    2. 它去字典(tok_emb)套现，并查位置表(pos_emb)加上前后顺序。
    3. 然后穿越几十层刚才写的乐高积木块（trf_blocks）。
    4. 最后一层层归一化后过个线性输出头（out_head），计算出下一个词各个选项的概率。
    """
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_dex):
        batch_size, seq_len = in_dex.shape
        tok_embeds = self.tok_emb(in_dex)

        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_dex.device)  # device 的设置允许我们在 CPU 或 GPU 上训练模型，具体取决于输入数据所在的设备
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

def assign(left, right):
    """
    🐰 【小白通俗解析此函数】
    作用：简单的搬运工。由于不同作者的矩阵形状可能相反，它先校验形状，没问题就把右边的参数原封不动赋给左边。
    """
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch.Left: {left.shape}, "
                         "Right: {right.shape}"
        )
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt, params): # 将模型的位置信息和词元嵌入权重设置为 params 中指定的值
    """
    🐰 【小白通俗解析此函数】
    作用：传说中的“记忆灵魂移植”。
    此时我们辛辛苦苦手敲出了大模型的骨骼(gpt)，但脑子是空的，是个白痴。
    我们要用这个函数，把人家 OpenAI 花费上千张 A100 大显卡、熬了几千个小时练出来的 GPT-2 的脑部参数(params)，
    一行行拆解、对准接口，精确地塞进我们自己搭的身体里。站在巨人的肩膀上，微调才有效！
    """
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for b in range(len(params["blocks"])): # 遍历模型中的每一个 Transformer 块
        q_w, k_w, v_w = np.split( # np.split 函数用于将注意力和偏置权重平均分为3个部分，分别用于查询组件、键组件和值组件
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1
        )
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T
        )
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T
        )
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T
        )

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1
        )
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b
        )
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b
        )
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b
        )

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T
        )
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"]
        )
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T
        )

        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"]
        )
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T
        )
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"]
        )

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"]
        )
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"]
        )
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"]
        )
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"]
        )

        gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
        gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
        gpt.out_head.weight = assign(gpt.out_head.weight,params["wte"]) # OpenAI 的原始 GPT-2 模型在其输出层中复用了词元嵌入权重，以减少参数总数，这一概念被称为“权重绑定”

BASE_CONFIG = {
    "vocab_size": 50257, # 词汇表大小
    "context_length": 1024, # 上下文长度
    "drop_rate": 0.0, # dropout 率
    "qkv_bias": True, # 查询-键-值偏置
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

CHOOSE_MODEL = "gpt2-medium (355M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
# print(model_size)

settings, params = download_and_load_gpt2(
    model_size=model_size,
    models_dir="gpt2"
)

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()

torch.manual_seed(123)
input_text = format_input(val_data[0])
print(input_text)

# 代码清单 5-4 修改后更具多样性的文本生成函数
def generate(model, idx, max_new_tokens, context_size,temperature=0.0, top_k=None, eos_id=None):
    """
    🐰 【小白通俗解析此函数】
    作用：究极完全体的“AI 打字机 / 文本生成引擎”。
    这就是现实中我们在使用 ChatGPT 时背后真实运行的代码罗盘：
    里面支持了基于温度 (temperature) 算法的情感随机性控制——温度低时它保守理智说正确废话，温度高时它微醺充满创作力。
    内部也支持取概率前K名 (top_k) 的保险锁机制，防止它生成一些无边无际的废话。
    每一次循环跳动，都是它苦思冥想“吐出”下一个新字的结晶。
    """
    for _ in range(max_new_tokens): # 这个for循环与之前一样，获取logits,并且只关注最后一个时间步
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        if top_k is not None: # 使用Top-k采样筛选logits
            top_logits , _ = torch.topk(logits, k=top_k)
            min_val = top_logits[:,-1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf")).to(logits.device),
                logits
            )
        if temperature > 0.0: # 使用温度缩放
            logits = logits / temperature
            probas = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probas, num_samples=1)
        else: # 当禁用温度缩放时，像以前一样执行贪心解码，选取下一个词元
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if idx_next == eos_id: # 如果遇到序列结束词元，则提前停止生成
            break
        idx = torch.cat((idx, idx_next), dim=-1)
    return idx

# 代码清单 5-1 用于文本到词元ID转换的工具函数
def text_to_token_ids(text, tokenizer):
    """
    🐰 【小白通俗解析此函数】
    作用：文字变暗号（编码器）。AI 无法阅读人类的文字外衣，必须先把文字查表转成类似 [12, 54, 888] 的张量。
    """
    encoded = tokenizer.encode(text, allowed_special = {'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)   # 使用 .unsqueeze(0) 添加 batch 维度
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    """
    🐰 【小白通俗解析此函数】
    作用：暗号变人类常言（解码器）。AI 苦算出来的全是一通密文数字，必须通过这个解码器翻译成咱人听得懂的汉字或英文输出。
    """
    flat = token_ids.squeeze(0)    # 移除batch维度
    return tokenizer.decode(flat.tolist())

token_ids = generate(
    model=model,
    idx=text_to_token_ids(input_text, tokenizer),
    max_new_tokens=35,
    context_size=BASE_CONFIG["context_length"],
    eos_id=50256,
)
generated_text = token_ids_to_text(token_ids, tokenizer)

response_text = generated_text[len(input_text):].strip()
print(response_text)

def calc_loss_batch(input_batch, target_batch, model, device):
    """
    🐰 【小白通俗解析此函数】
    作用：单批次“改卷计分器”。
    原理：拿走学生(model)刚给出的预测答案(logits)，将它死死对准目标正确答案(target_batch)逐个词元比对。
    算出一个叫作交叉熵损失 (Cross Entropy Loss) 的数值。这个数值越接近 0 越好，说明学生全蒙对了。
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss

# 代码清单 5-2 用于计算训练集和验证集损失的函数
def calc_loss_loader(data_loader, model, device, num_batches = None):
    """
    🐰 【小白通俗解析此函数】
    作用：算所有卷子的“综合平均分”。
    把管家(data_loader)里存放的成百上千张卷子拿出来让 AI 全做一遍，把刚才所有单批次的计分器得分全部加起来求平均！
    这是评估 AI 当前总实力的核心宏观指标！
    """
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)  # 如果没有指定遍历多少个批次（num_batches），那么就遍历所有批次
    else:
        num_batches = min(num_batches, len(data_loader))  # 如果 num_batches 超过数据加载器中的批次数，那么就需要减少批次数，以匹配数据加载器中的总批次数
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            total_loss += loss.item()  # 每个批次的损失的总和
        else:
            break
    return total_loss / num_batches  # 对所有批次的损失求平均值

# 代码清单 5-3 预训练大模型的主函数
def train_model_simple(model, train_loader,val_loader,
                       optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    """
    🐰 【小白通俗解析此主函数】
    作用：超级硬核的 AI “魔鬼训练营流水线”！！(指令微调全靠它运作)。
    这几十行代码掌控了 AI “吃资料 -> 犯错 -> 纠错长记性” 的生死循环：
    它要不断把考卷送进 AI 大脑，算差距(Loss)，接着施展反向传播(loss.backward)这招降鬼十八掌，
    看是哪里的参数出错，最后利用教练优化器(optimizer)挥舞大锤纠正那个参数的角度，日进不休，直到变强。
    """
    # 这些列表好比老师手里的“积分册”，用来一直跟踪损失和所见的词元
    train_losses, val_losses ,track_tokens_seen = [], [], []  # 初始化列表以跟踪损失和所见的词元
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):  # 开始主训练循环 (epoch 意味着要把所有复习卷从头到尾盘几遍，1遍即1个epoch)
        model.train() # 告诉模型：“进入上课吃苦状态”，开启所有的学习和反思机制
        for input_batch, target_batch in train_loader: # 从发卷机 DataLoader 里抓一批考卷过来
            optimizer.zero_grad()  # 🌟【一擦黑板】清空上一次解题的缓存，保证每次新推导不受干扰

            # 🌟【二做题，算分差】让模型做题得出答案，然后立刻跟标准答案比对，算出所谓的“Loss”（错误率/损失值）
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )

            # 🌟【三找原因】所谓反向传播，就是让这个错误率沿着神经网络一层层往回找，发现是哪个神经元里的哪颗参数坏了事
            loss.backward()  # 计算损失梯度

            # 🌟【四纠错打补丁】用所谓的“优化器(optimizer)”，把那个坏参数旋钮偷偷转一点点方向！真正完成了知识进脑子！
            optimizer.step()  # 使用损失梯度更新模型权重

            tokens_seen += input_batch.numel() # 记录一下看了几个字元了
            global_step += 1

            if global_step % eval_freq == 0:  # 考了多少步就休息一下，进行阶段性验收小测验（避免学偏）
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}):"
                      f"Train loss {train_loss:.3f},"
                      f"Val loss {val_loss:.3f}"
                )

        generate_and_print_sample(  # 每一册做完，挑一题让它现场答出一段话，打印在你的黑框窗口里，让你直观感受效果！
            model, tokenizer, device, start_context
        )
    return train_losses, val_losses, track_tokens_seen

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    🐰 【小白通俗解析此函数】
    作用：阶段性封闭测验。
    在训练时每隔一段时间，它会让模型暂停学习(model.eval())，并锁死作弊更新功能(torch.no_grad())。
    专门用训练集和另一批没见过的验证集去考测它一番。
    如果训练集得分很高，验证集得分很烂，就是妥妥的书呆子(过拟合)，需要工程师介入干涉了。
    """
    model.eval()  # 告诉模型：“进入考试验收状态”，此时会关闭 Dropout 之类的脑补丢弃功能，这样每次考出来的分才公平稳定。
    with torch.no_grad():  # “禁止作弊找原因”，也就是禁止算梯度和参数更新，因为测验是为了测试实力不是为了调参数。还能省大量的显存。
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss

def generate_text_simple(model, idx,  # idx 是当前文本的索引数组，其形状为(batch, n_tokens)
                         max_new_tokens, context_size):
    """
    🐰 【小白通俗解析此函数】
    作用：简化版打字机(接龙引擎)。主要是不带温度等花哨机制，每次都老实巴交地挑得分 100% 最高的字。
    用于在训练的间隙偷懒省事，快速考校一眼 AI。
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]  # 将当前文本截断至支持的长度。如果大语言模型仅支持 5 个词元，但此时文本长度为 10，则只有最后 5 个词元会被用作输入文本
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]  # 只关注最后一个输出的内容，因此形状会从 (batch, n_token, vocab_size) 变为 (batch, vocab_size)
        probas = torch.softmax(logits, dim=-1)  # probas 的形状为 (batch, vocab_size)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # idx_next 的形状为 (batch, 1)
        idx = torch.cat((idx, idx_next), dim=1)  # 将计算出的下一个字符的索引添加到索引数组中，此时 idx 的形状会变为 (batch, n_tokens + 1)

    return  idx

def generate_and_print_sample(model, tokenizer, device, start_context):
    """
    🐰 【小白通俗解析此函数】
    作用：“期中才艺小汇报”。
    训练过程干巴巴的全是数字滚动太无聊了，所以每一大轮测验结束后，
    这段代码会让模型就地写个几十个字的小作文并且丢到屏幕上。
    你可以当场肉眼看看——这家伙从只会阿巴阿巴，是不是变成了一个有条理的智能助理。
    """
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # 紧凑的打印格式
    model.train()

model.to( device)
torch.manual_seed(123)

with torch.no_grad():
    train_loss = calc_loss_loader(
        train_loader, model, device, num_batches=5
    )
    val_loss = calc_loss_loader(
        val_loader, model, device, num_batches=5
    )

print("Training loss:", train_loss)
print("Validation loss:", val_loss)


# 代码清单 7-8 对预训练的大语言模型进行指令微调
import time

start_time = time.time()
torch.manual_seed(123)
# 这一波是我们请来的“魔鬼教练/优化器”，AdamW 是目前练大模型最好用的调参工具之一。
# lr = 0.00005 （学习率）代表它纠错的力度非常轻微，因为我们是在已有知识基础上微调，动作太猛会把以前的常识毁掉（灾难性遗忘）
optimizer = torch.optim.AdamW(
    model.parameters(), lr=0.00005, weight_decay=0.1
)
num_epochs = 2 # 所有的题重复做2遍！

# 正式启动！漫长的等待，此时显卡会呼啸起来...
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,\
    start_context=format_input(val_data[0]), tokenizer=tokenizer
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
# plot_losses 会把训练损失和验证损失画在同一张图里，方便比较。
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    """
    🐰 【小白通俗解析此函数】
    作用：“生成成绩动态走势折线图”。
    有了 matplotlib 画图库，我们不再看滚动的黑白数字代码。
    它会画一条随着不断复习而向下滑落甚至平稳到底的线——Loss 下降曲线！一条完美的下坠曲线是所有算法工程师最大的多巴胺高潮！
    """
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label = "Training loss")
    ax1.plot(
        epochs_seen, val_losses, linestyle="-.",label = "Validation loss"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc = "upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny() # 创建共享同一个y轴的第二个x轴
    ax2.plot(tokens_seen, train_losses, alpha=0)  # 对齐刻度线的隐藏图表
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

torch.manual_seed(123)

for entry in test_data[:3]: # 遍历前3个测试样本
    input_text = format_input(entry)
    token_ids = generate( # 使用7.5节中引入的生成函数
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256,
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)

    response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
    )
    print(input_text)
    print(f"\nCorrect response:\n>> {entry['output']}\n")
    print(f"\nModel response:\n>> {response_text.strip()}\n")
    print("-------------------------------------")


# 代码清单 7-9 生成测试集上的回复
from tqdm import tqdm

for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
    input_text = format_input(entry)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256,
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)

    response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
    )
    test_data[i]["model_response"] = response_text

with open("instruction-data-with-response.json", "w") as file:
    json.dump(test_data, file, indent=4) # 为格式美观而指定缩进


print(test_data[0])


import re

file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL) }-sft.pth" # 去除文件名中的空白字符和括号
torch.save(model.state_dict(), file_name)
print(f"Model saved as {file_name}")


# 命令行输入 ollama run llama3


# 验证 Ollama 会话是否正常运行
import psutil

def check_if_running(process_name):
    """
    🐰 【小白通俗解析此函数】
    作用：“进程监工小机器人”。
    因为最后我们要召唤本地安装的 Llama/Ollama 过来当阅卷老师，
    所以这个函数专门跑到底层任务管理器里，搜一遍这程序有没有按时打卡上班。没上班直接给你红牌警告报错。
    """
    running = False
    for proc in psutil.process_iter(["name"]):
        if process_name in proc.info["name"]:
            running = True
            break
    return running

ollama_running = check_if_running("ollama")

if not ollama_running:
    raise RuntimeError(
        "Ollama not running. Launch ollama before proceeding."
    )
print("Ollama running:", check_if_running("ollama"))


# 代码清单 7-10 与本地部署的 Ollama 模型交互
import  urllib.request

def query_model(
        prompt,
        model="llama3",
        url="http://localhost:11434/api/v1/chat",
):
    """
    🐰 【小白通俗解析此函数】
    作用：“和本机的外部 AI (Ollama) 的 HTTP 对讲机”。
    我们自己的 GPT 小弟考完试后，自己当然不能给自己打分。
    所以在这个函数里，我们会包装一个正式的打分申请表（POST Request），按规定地址顺网线递交给那个正在值班的 Llama 打分老师。
    然后端着盘子听取阅卷老师的批语并保存下来。
    """
    # 这里我们在代码里假装成一个外部的调用者，把我们要打分的资料和试卷装进 JSON 箱子里，发给本机的 Llama 阅卷老师。
    data = { # 创建字典格式的数据
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "options":{ # 设置种子得到确定性的返回结果 (这里设温度为0，意思是要求最严格、最铁面白无私的裁判标准)
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048
        }
    }

    payload = json.dumps(data).encode("utf-8") # 将字典变成 JSON 格式的字符串，并编码为 UTF-8 字节
    # 创建一个请求对象，将方法设置为 POST ，并加入必要的请求头
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST"
    )

    request.add_header("Content-Type", "application/json")

    response_data = ""
    with urllib.request.urlopen(request) as response: # 发送请求并捕获模型回复
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]

    return response_data

model = "llama3"
result = query_model("What do Llamas eat?", model=model)
print(result)

for entry in test_data[:3]:
    prompt = (
        f"Given the input `{format_input(entry)}` "
        f"and correct output `{entry['output']}`, "
        f"score the model response `{entry['model_response']}`"
        f" on a scale from 0 to 100, where 100 is the best score."
    )
    print("\nDataset response:")
    print(">>", entry['output'])
    print("\nModel response:")
    print(">>", entry['model_response'])
    print("\nScore:")
    print(">>", query_model(prompt))
    print("\n-------------------------------------")


# =====================================================================
# 🐰 【小白导读：代码清单 7-11 评估指令微调后的大语言模型】
# 这一步汇总所有的判卷结果。
# 依次将咱们微调模型写的答案、标准答案丢给大裁判（Llama3），让它打分（0-100分）。
# 最终算出咱们这个经过岗前培训的 AI，平均水平能拿多少分。恭喜完结！
# =====================================================================
# 代码清单 7-11 评估指令微调后的大语言模型
def generate_model_scores(json_data, json_key, model="llama3"):
    """
    🐰 【小白通俗解析此函数】
    作用：批阅成绩全自动化大巡视！
    它像是一个流水线监工，一条条把小弟做的题(model_response)和标准答案(output)抽出来。
    然后塞进去问裁判“满分100，这题你给几分？”
    收到裁判大人的只言片语分数后，全部积存在数组里，并在最后给大家播报均摊分数结束战斗。
    """
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry[json_key]}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only." # 修改提示词，以便仅返回分数
        )
        score = query_model(prompt, model=model)
        try:
            scores.append(int(score))
        except ValueError:
            print(f"Could not convert score: {score}")
            continue

    return scores

scores = generate_model_scores(test_data, "model_response")
print(f"Number of scores: {len( scores)} of {len(test_data)}")
print(f"Average score: {sum(scores) / len(scores):.2f}\n")