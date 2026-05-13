import os

from jinja2.optimizer import optimize

# =========================================================
# 第一部分：搭建一个最小可用的 GPT 模型
# 这一大段主要做两件事：
# 1) 先把 Transformer 的核心组件一块块实现出来；
# 2) 再把它们组合成一个可以预测下一个词元的 GPTModel。
# 如果你是第一次看，可以先把它理解成：
# “输入一句话 -> 模型给出下一个词最可能是什么” 的完整流水线。
# =========================================================

# 某些 Windows 环境下，PyTorch / Intel MKL 可能会因为多线程库重复加载而报错。
# 这里设置环境变量，相当于告诉底层运行时：允许重复加载这个库，避免程序一启动就崩溃。
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # 解决多线程运行时出现的错误

import torch
import torch.nn as nn
import tiktoken

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # 每个注意力头分到的维度 = 总输出维度 / 头数

        # Q、K、V 三个线性层分别把输入映射成查询、键和值。
        # 它们的输入维度都是 d_in，输出维度都是 d_out。
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # 把多个头拼接后的结果再投影回 d_out，方便和残差连接对接。
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        # 预先创建一个上三角矩阵作为因果掩码：右上方位置为 1，表示“未来信息不能看”。
        # 这个 buffer 不参与训练，但会随着模型一起保存和迁移设备。
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        # x 的形状通常是 (batch_size, num_tokens, d_in)
        b, num_tokens, d_in = x.shape

        # 把输入分别映射成 K/Q/V。
        # 形状从 (b, num_tokens, d_in) 变成 (b, num_tokens, d_out)
        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 下面这一步是在“逻辑上”把最后的 d_out 拆成多个头。
        # 例如 d_out=768, num_heads=12，那么每个头就是 64 维。
        # 变形后得到 (b, num_tokens, num_heads, head_dim)
        # 这样每个 token 都会有多个头各自的一份 Q/K/V 表示。
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 转置后变成 (b, num_heads, num_tokens, head_dim)
        # 这样每个注意力头都可以独立计算“自己这一路”的注意力分数。
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 计算注意力分数：Q 和 K 做矩阵乘法。
        # 结果形状是 (b, num_heads, num_tokens, num_tokens)
        # 最后一个维度表示：当前 token 对序列中每个 token 的关注程度。
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # 取出和当前序列长度对应的掩码，并转成布尔类型。
        # mask 中为 True 的位置表示要被遮住（未来位置）。
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # 把未来位置的分数改成负无穷，softmax 后它们的概率就会接近 0。
        # 这就是“因果注意力”：只能看当前 token 以及它前面的 token，不能偷看未来。
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 缩放点积注意力：除以 sqrt(head_dim) 可以避免分数过大导致 softmax 过于极端。
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        # dropout 用在注意力权重上，训练时随机丢弃一部分连接，减少过拟合。
        attn_weights = self.dropout(attn_weights)

        # 注意力权重乘上 V，得到每个 head 的上下文表示。
        # 形状先是 (b, num_heads, num_tokens, head_dim)，再转成 (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 把多个 head 的结果拼回一个大向量。
        # 例如 12 个 head × 每个 64 维 = 768 维。
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec

class LayerNorm(nn.Module):
    # LayerNorm 的作用：对每个 token 的特征维做标准化，让训练更稳定。
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        # scale 和 shift 是可学习参数：标准化后再做线性变换，保留模型表达能力。
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        # 对最后一维做均值和方差统计，也就是每个 token 自己内部的特征统计。
        mean = x.mean(dim = -1, keepdim = True)
        var = x.var(dim = -1, keepdim = True, unbiased = False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    # GELU 是一种常见的激活函数，GPT / Transformer 里经常使用。
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):
    # 前馈网络：每个 token 独立通过一个两层 MLP 做非线性变换。
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
    # 一个标准 Transformer Block = 多头注意力 + 前馈网络 + 两次残差连接 + 两次归一化。
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
        # 第一段：注意力子层
        # shortcut 保存原始输入，方便后面和子层输出相加，这就是“残差连接”。
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = shortcut + x  # 将原始输入添加回来

        # 第二段：前馈子层
        # 再做一次残差连接，让梯度更容易传播，深层网络更容易训练。
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = shortcut + x
        return x

class GPTModel(nn.Module):
    # 一个简化版 GPT：词元嵌入 + 位置嵌入 + 多层 Transformer + 预测词表的输出层。
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
        # in_dex 的形状通常是 (batch_size, seq_len)，里面是 token id。
        batch_size, seq_len = in_dex.shape
        # 把 token id 变成向量表示。
        tok_embeds = self.tok_emb(in_dex)

        # 位置编码：给每个位置一个独特的向量，让模型知道“第几个词”。
        # 这里用 arange(seq_len) 生成 0,1,2,...,seq_len-1。
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_dex.device)  # device 的设置允许我们在 CPU 或 GPU 上训练模型，具体取决于输入数据所在的设备
        )
        # token embedding + position embedding，得到最终输入表示。
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        # 输出层给出每个位置对词表中每个 token 的打分（logits）。
        logits = self.out_head(x)
        return logits

GPT_CONFIG_124M = {
    # 这是一个近似 GPT-2 small / 124M 的配置。
    # vocab_size：词表大小；context_length：最大上下文长度；emb_dim：隐藏维度；
    # n_heads：注意力头数；n_layers：Transformer 层数；drop_rate：dropout 概率；qkv_bias：QKV 线性层是否带偏置。
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

model = GPTModel(GPT_CONFIG_124M)

def generate_text_simple(model, idx,  # idx 是当前文本的索引数组，其形状为(batch, n_tokens)
                         max_new_tokens, context_size):
    # 这个函数是“贪心生成”：每一步都选概率最大的下一个 token。
    # 它很稳定，但通常不如采样方法多样。
    for _ in range(max_new_tokens):
        # 只保留最近 context_size 个 token，防止输入长度超过模型可处理范围。
        idx_cond = idx[:, -context_size:]  # 将当前文本截断至支持的长度。如果大语言模型仅支持 5 个词元，但此时文本长度为 10，则只有最后 5 个词元会被用作输入文本
        with torch.no_grad():
            logits = model(idx_cond)

        # 只取最后一个位置的输出，因为我们要预测“下一个 token”。
        logits = logits[:, -1, :]  # 只关注最后一个输出的内容，因此形状会从 (batch, n_token, vocab_size) 变为 (batch, vocab_size)
        probas = torch.softmax(logits, dim=-1)  # probas 的形状为 (batch, vocab_size)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # idx_next 的形状为 (batch, 1)
        # 把预测到的 token 追加到序列末尾，形成更长的上下文。
        idx = torch.cat((idx, idx_next), dim=1)  # 将计算出的下一个字符的索引添加到索引数组中，此时 idx 的形状会变为 (batch, n_tokens + 1)

    return  idx


# 代码清单 5-1 用于文本到词元ID转换的工具函数
# 这两个函数负责在“字符串”和“token id 张量”之间互相转换。
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special = {'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)   # 使用 .unsqueeze(0) 添加 batch 维度
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)    # 移除batch维度
    return tokenizer.decode(flat.tolist())

# 用一个短句做推理测试，看看模型是否能顺利从 prompt 往后生成。
start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model = model,
    idx = text_to_token_ids(start_context, tokenizer),
    max_new_tokens = 10,
    context_size = GPT_CONFIG_124M["context_length"],
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


# 下面这一小段是在演示“监督学习”里输入和目标应该怎样对齐。
# inputs 是模型看到的上下文，targets 是对应的“下一步正确答案”。
inputs = torch.tensor([[16833, 3626, 6100],  # ["every effort moves",
                       [40, 1107, 588]])     # "I really like"]
targets = torch.tensor([[3626, 6100, 345],   # ["effort moves you",
                       [1107, 588, 11311]])  # "really like chocolate"]

with torch.no_grad():  # 屏蔽模型参数的梯度跟踪，因为我们还没开始训练
    logits = model(inputs)
# 先用 softmax 看看每个词元的输出概率。
# 这里的 logits 可以看成是模型对每个候选词元的原始打分；
# softmax 之后才会变成“概率”，也就是每个词元被选中的可能性。
probas = torch.softmax(logits, dim=-1)  # 词汇表中每个词元的概率
print(probas.shape)


# 这里先取出每个位置预测到的 token id，用于和目标标签做对比。
# argmax 会在每一行里选出分数最高的位置，也就是模型“最想猜”的词元。
token_ids = torch.argmax(probas, dim=-1, keepdim=True)
print("Token IDs:\n", token_ids)

print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
print(f"Outputs batch 1:"
      f"{token_ids_to_text(token_ids[0].flatten(), tokenizer)}")


# 下面计算的是“正确答案 token”的概率。
# 例如第 1 个样本里，第 0、1、2 个位置分别对应正确标签的概率。
text_idx = 0
target_probas_1 = probas[text_idx, [0, 1, 2],  targets[text_idx]]
print("Text 1:", target_probas_1)

text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2],  targets[text_idx]]
print("Text 2:", target_probas_2)

# 把两个样本中正确答案的概率拼起来，再取对数。
# 这么做的原因是：语言模型训练里常用“对数概率”来衡量整体好坏，
# 这样很多很小的概率值乘在一起时，就不会迅速下溢到 0。
log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print(log_probas)

# 平均对数概率：可以把它理解成“整体上模型给正确答案多大的信心”。
avg_log_probas = torch.mean(log_probas)
print(avg_log_probas)

# 因为训练时通常希望“越小越好”的损失函数，
# 所以把平均对数概率取负号，就得到一个更符合优化习惯的值。
neg_avg_log_probas = avg_log_probas * -1
print(neg_avg_log_probas)

print("Logits shape:", logits.shape)
print("Targets shape:", targets.shape)

# 交叉熵损失要求把 (batch, seq_len, vocab_size) 拉平成二维，
# 再把 target 拉平成一维，这样才能逐位置计算分类损失。
# 这里相当于把“每个位置预测下一个词”的问题，转成一堆独立的分类问题。
logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()
print("Flattened logits:", logits_flat.shape)
print("Flattened targets:", targets_flat.shape)

loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print(loss)


# =========================================================
# 第三部分：把一整段文本变成训练数据
# 语言模型训练不是直接拿整本书喂进去，而是要切成很多小窗口：
# - 输入是前面的 token
# - 目标是右移一位后的 token
# 这样模型就能学会“根据上文预测下一个词”。
# =========================================================

# 加载数据集
# 这里读取的是《The Verdict》文本，它会被当作一个小型语言建模语料。
file_path = "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

# 检查数据集中的字符数和词元数
# 这里先统计字符数和 token 数，方便了解数据集规模。
# 一般来说，字符数只说明原始文本长度，token 数更贴近模型真实处理的数据量。
total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("Characters:", total_characters)
print("Tokens:", total_tokens)


# 按 9:1 切分训练集和验证集。
# 训练集用于更新参数，验证集用于评估泛化能力。
# 训练损失下降不代表模型真的学会了，验证集能帮助我们看出是否过拟合。
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

from torch.utils.data import Dataset

class GPTDatasetV1(Dataset):
    # 把一整段文本切成很多“长度固定、彼此重叠”的小样本。
    # 每个样本的输入是前面的 token，标签是右移 1 位后的 token。
    # 这就是语言模型最常见的自监督学习方式。
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # 对全部文本进行分词
        token_ids = tokenizer.encode(txt)
        # 使用滑动窗口将文本划分为长度为max_length的重叠序列
        # 例如：
        # input = [t0, t1, t2, ...]
        # target = [t1, t2, t3, ...]
        # 这样模型就学会“根据前文预测下一个词”。
        # stride 越小，样本重叠越多；stride 越大，样本之间重复越少。
        for i in range(0, len(token_ids) - max_length, stride):
            input_ids = token_ids[i: i + max_length]
            target_ids = token_ids[i + 1: i + 1 + max_length]
            self.input_ids.append(torch.tensor(input_ids))
            self.target_ids.append(torch.tensor(target_ids))

    # 返回数据集的总行数
    def __len__(self):
        return len(self.input_ids)

    # 返回数据集的指定行
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_V1(
        txt, batch_size = 4, max_length = 256, stride = 128, shuffle = True, drop_last = True, num_workers = 0
):
    # 初始化分词器
    tokenizer = tiktoken.get_encoding("gpt2")
    # 创建数据集
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    # DataLoader 负责把很多单样本整理成 batch，并在训练时自动迭代。
    # 它还可以帮我们做打乱、批处理、丢弃最后一个不完整 batch 等操作。
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        drop_last = drop_last,    # 如果drop_last为True且批次大小小于指定的batch_size，则会删除最后一批，以防止在训练期间出现损失剧增
        num_workers = num_workers,    # 用于预处理的CPU进程数
    )
    return dataloader

# 固定随机种子，保证每次运行时的随机行为尽量一致。
# 这样做有助于调试，也方便比较不同版本代码的效果。
torch.manual_seed(123)

train_loader = create_dataloader_V1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0,
)
val_loader = create_dataloader_V1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0,
)

# 遍历数据加载器，确保它们被正确创建
# 这里的 shape 检查很重要：如果输入和标签尺寸对不上，后面的 loss 计算就会报错。
print("Train loader:")
for x,y in train_loader:
    print(x.shape, y.shape)

print("\nValidation loader:")
for x,y in val_loader:
    print(x.shape, y.shape)


# =========================================================
# 第四部分：把数据送进模型，并计算训练 / 验证损失
# 这一部分的关键是理解“训练”和“评估”的区别：
# - 训练时要更新参数
# - 验证时只看效果，不更新参数
# 同时这里也会统一处理 device，保证数据和模型在同一个设备上。
# =========================================================

# 实现一个工具函数，用于计算通过训练集加载器和验证集加载器返回的给定批次的交叉熵损失
# 这一层把 batch 数据搬到指定设备上，然后交给模型计算 loss。
# 这样可以把“单个 batch 的评估逻辑”复用到训练和验证两种场景。
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss

# 代码清单 5-2 用于计算训练集和验证集损失的函数
# 这个函数支持只抽样前 num_batches 个 batch，避免每次评估都太慢。
def calc_loss_loader(data_loader, model, device, num_batches = None):
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


# 自动选择 GPU 或 CPU。
# 如果机器有 CUDA，就走 GPU；否则退回 CPU，代码不用改。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)  # 如果你有一台支持CUDA的GPU机器，那么大语言模型将自动在GPU上训练且不需要修改代码
with torch.no_grad():  # 因为还没有开始训练，所以不使用梯度追踪，这样会更高效
    train_loss = calc_loss_loader(train_loader, model, device)  # 通过“设备”设置，可以确保所有的数据和大语言模型在同一个设备上
    val_loss = calc_loss_loader(val_loader, model, device)
print("Training loss:", train_loss)
print("Validation loss:", val_loss)


# =========================================================
# 第五部分：正式训练、定期评估、并生成样本文本
# 这一段才是完整的训练循环：
# - 前向传播
# - 反向传播
# - 参数更新
# - 周期性评估
# - 训练后生成一段文本看看效果
# 你可以把它理解为“模型边学边考试，还会顺手写一小段作文”。
# =========================================================

# 代码清单 5-3 预训练大模型的主函数
# 训练主循环：每个 batch 做一次前向传播、反向传播和参数更新。
def train_model_simple(model, train_loader,val_loader,
                       optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses ,track_tokens_seen = [], [], []  # 初始化列表以跟踪损失和所见的词元
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):  # 开始主训练循环
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # 重置上一个批次迭代中的损失梯度
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            loss.backward()  # 计算损失梯度
            optimizer.step()  # 使用损失梯度更新模型权重
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:  # 可选的评估步骤
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

        generate_and_print_sample(  # 每轮之后打印一个文本样本
            model, tokenizer, device, start_context
        )
    return train_losses, val_losses, track_tokens_seen

# 在评估模式下关闭 dropout，让验证结果更稳定。
# 评估时不需要随机性，所以要切到 eval()，评估完再切回 train()。
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()  # 在评估阶段禁用 dropout, 以产出稳定且可复现的结果
    with torch.no_grad():  # 评估阶段也会禁用梯度跟踪，因为这是不需要的，而且这样可以减少计算开销
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss

# 每轮训练结束后，用一个固定 prompt 生成样本文本，方便直观看到模型有没有学到东西。
# 这一步相当于训练过程中的“肉眼检查”，能帮助判断模型是否真的在变好。
def generate_and_print_sample(model, tokenizer, device, start_context):
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


# 下面这一段是真正开始训练的入口。
# 先创建新模型、放到 device 上，再定义 AdamW 优化器，最后调用训练函数。
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),  # .parameters() 方法返回模型的所有可训练参数
    lr=0.0004, weight_decay=0.1
)
num_epochs = 10
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)


# 创建一张简单的图表，将训练集和验证集的损失并列显示
# 这能直观看出模型是欠拟合、正常收敛，还是开始过拟合。
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
# plot_losses 会把训练损失和验证损失画在同一张图里，方便比较。
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
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

# 横轴用 epoch 表示训练进度，tokens_seen 表示模型“看过多少词元”。
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)


# 先将模型移到CPU上，因为相对较小的模型的推断不需要GPU。此外，在训练后，需要将模型置于评估模式，以关闭诸如dropout之类的随即组件
# 这一步的意义是：训练阶段和推理阶段的运行方式不同，推理时应该关闭随机失活。
model.to("cpu")
model.eval()

# 再次初始化 tokenizer，确保后面生成文本时使用的是一致的 GPT-2 编码器。
# 重新取一次 tokenizer 可以避免前后上下文混用导致的编码不一致。
tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate_text_simple(
    model = model,
    idx = text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens = 25,
    context_size = GPT_CONFIG_124M["context_length"]
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


# =========================================================
# 第六部分：温度采样、Top-k 采样，让生成更有变化
# 如果一直只选概率最大的词，输出会很死板。
# 这部分是在教模型“不要每次都那么保守”，
# 通过温度和 top-k 控制随机性，让生成结果更自然、更丰富。
# =========================================================

### 温度缩放
# 这一节在演示：当模型不是“必须选最大概率词”时，怎么让生成更有随机性。
# temperature 越低，分布越尖锐；temperature 越高，分布越平坦。
vocab = {
    "closer": 0,
    "every": 1,
    "effort": 2,
    "forward": 3,
    "incher": 4,
    "moves": 5,
    "pizza": 6,
    "toward": 7,
    "you": 8,
}
inverse_vocab = {v: k for k, v in vocab.items()}
next_token_logits = torch.tensor(
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)

probas = torch.softmax(next_token_logits, dim=-0)
next_token_id = torch.argmax(probas).item()
print(inverse_vocab[next_token_id])

# 这里用 multinomial 从概率分布里随机抽样，而不是总选最大值。
# 这样可以得到更自然、更多样的结果，但同时也更“不可预测”。
torch.manual_seed(123)
next_token_id = torch.multinomial(probas, num_samples=1).item()
print(inverse_vocab[next_token_id])
def print_sampled_tokens(probas):
    torch.manual_seed(123)
    sample = [torch.multinomial(probas, num_samples=1).item()
              for i in range(1_000)] # 重复1000次
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")

# print_sampled_tokens(probas)


# softmax_with_temperature 的核心思想很简单：
# 先把 logits 除以温度，再做 softmax。
# 温度 < 1 会更保守，温度 > 1 会更发散。
def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)

# temperatures = [1, 0.1, 5]
# scaled_probas = [softmax_with_temperature(next_token_logits, t)
#                 for t in temperatures]
# x = torch.arange(len(vocab))
# bar_width= 0.15
# fig, ax = plt.subplots(figsize=(5, 3))
# for i,T in enumerate(temperatures):
#     rects = ax.bar(x + i * bar_width, scaled_probas[i],
#                    bar_width, label = f"temperature = {T}")
# ax.set_ylabel("Probability")
# ax.set_xticks(x)
# ax.set_xticklabels(vocab.keys(), rotation = 90)
# ax.legend()
# plt.tight_layout ()
# plt.show()


### Top-k 采样
# Top-k 会先找出分数最高的 k 个候选词，只允许在这几个里选择，
# 其余词元直接被压成负无穷，相当于“完全禁选”。
top_k = 3
top_logits, top_pos = torch.topk(next_token_logits, k=top_k)
# print("Top-k tokens:", top_logits)
# print("Top-k positions:", top_pos)

new_logits = torch.where(
    condition=next_token_logits <top_logits[-1], # 识别出比前三个logits值中最低的logits值还低的logits值
    input=torch.tensor(float("-inf")),  # 将这些更低的logits值替换为负无穷大
    other=next_token_logits # 保留其他词元的原始logits值
)
# print(new_logits)

topk_probas = torch.softmax(new_logits, dim=-0)
# print(topk_probas)


# 代码清单 5-4 修改后更具多样性的文本生成函数
# 相比贪心解码，这个版本支持温度采样和 top-k 截断，因此生成结果更丰富。
# 适合想要“更像人写的”输出，而不是每次都固定同一个答案。
def generate(model, idx, max_new_tokens, context_size,temperature=0.0, top_k=None, eos_id=None):
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

# 这里再做一次生成测试，确认新版本 generate 能同时支持 top_k 和 temperature。
torch.manual_seed(123)
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=25,
    temperature=1.4
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


# =========================================================
# 第七部分：保存 / 恢复模型参数的思路
# 这部分是实战里很常见的操作：
# - 训练到一半保存下来
# - 之后可以继续训练
# - 也可以直接加载做推理
# 这样不用每次都从头训练。
# =========================================================

# torch.save(model.state_dict(), "model.pth")
#
# model = GPTModel(GPT_CONFIG_124M)
# model.load_state_dict(torch.load("model.pth", map_location=device))
# model.eval()


# torch.save({
#     "model_state_dict": model.state_dict(),
#     "optimizer_state_dict": optimizer.state_dict(),
#     },
#     "model_and_optimizer.pth"
# )
#
# checkpoint = torch.load("model_and_optimizer.pth", map_location=device)
# model = GPTModel(GPT_CONFIG_124M)
# model.load_state_dict(checkpoint["model_state_dict"])
# optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
# optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
# model.train()


# pip install tensorflow>=2.15.0 tqdm>=4.66

# =========================================================
# 第八部分：下载并加载官方 GPT-2 权重
# 这一段是在展示“把我们手写的模型，变成一个和 GPT-2 对齐的模型”。
# 重点不是重新训练，而是：
# 1) 下载官方参数
# 2) 按照层的结构逐一映射到我们的模型里
# 3) 让自己实现的代码直接拥有预训练能力
# =========================================================

import urllib.request
url = (
    "https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch05/"
    "01_main-chapter-code/gpt_download.py"
)
filename = url.split('/')[-1]
urllib.request.urlretrieve(url, filename)


# 下面开始加载 OpenAI 官方 GPT-2 预训练权重。
# 这样做的目的是把我们自己实现的 GPTModel，变成一个“能直接对齐 GPT-2 参数结构”的模型。
# 简单说，就是把“结构相同但参数随机”的模型，替换成“结构相同且参数已经训练好”的模型。
from gpt_download import download_and_load_gpt2
settings, params = download_and_load_gpt2(
    model_size = "124M", models_dir="gpt2"
)
print("settings:", settings)
print("Parameter dictionary keys:", params.keys())

print(params["wte"])
print("Token embedding weight tensor dimensions:", params["wte"].shape)


# 不同 GPT-2 尺寸对应不同的隐藏维度、层数和注意力头数。
# 这里只是定义映射，后面通过 model_name 选择其中一种配置。
model_configs = {
    "gpt2-small (124M)":{"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)":{"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)":{"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)":{"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

model_name = "gpt2-small(124M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])

# GPT-2 原始上下文长度是 1024，这里把最大上下文扩展到 1024。
NEW_CONFIG.update({"context_length": 1024})

# GPT-2 的 QKV 线性层通常带偏置，因此这里打开 qkv_bias。
NEW_CONFIG.update({"qkv_bias": True})

gpt = GPTModel(NEW_CONFIG)
gpt.eval()


# assign 的作用：检查左右张量形状是否一致，然后把右侧数据包装成可训练参数。
# 这样可以比较安全地把预训练权重塞到我们自己写的模型里。
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch.Left: {left.shape}, "
                         "Right: {right.shape}"
        )
    return torch.nn.Parameter(torch.tensor(right))

# 代码清单 5-5 将OpenAI的权重加载到GPT模型代码中
import numpy as np

# 把下载下来的 GPT-2 权重，逐层拷贝到我们自己的 GPT 模型里。
# 这一步最关键的地方是：各层参数名、形状、顺序都要严格对应。
# 如果有一个张量维度不匹配，就说明参数没有对齐成功。
def load_weights_into_gpt(gpt, params): # 将模型的位置信息和词元嵌入权重设置为 params 中指定的值
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

# 加载权重后，模型就不再是随机初始化，而是变成一个真正可用的 GPT-2 版本。
# 这一步通常是文本生成质量明显提升的关键。
load_weights_into_gpt(gpt, params)
gpt.to(device)


# =========================================================
# 第九部分：用加载好权重的 GPT-2 再做一次生成
# 这是最后的验证：
# 如果权重加载正确，模型输出通常会比随机初始化时自然很多。
# 这一步就像是“拿现成毕业作品验收一下”。
# =========================================================

# 使用加载好权重的 GPT-2 模型做一次完整生成，验证权重导入是否正常。
# 如果这里输出的句子比较通顺，就说明参数映射基本成功。
torch.manual_seed(123)
token_ids = generate(
    model=gpt,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to( device),
    max_new_tokens=25,
    context_size=NEW_CONFIG["context_length"],
    top_k=50,
    temperature=1.5
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
