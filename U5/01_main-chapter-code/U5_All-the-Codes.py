import torch
import torch.nn as nn
import tiktoken

class MultiHeadAttention(nn.Module):
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
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):
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

class GPTModel(nn.Module):
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

GPT_CONFIG_124M = {
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
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]  # 将当前文本截断至支持的长度。如果大语言模型仅支持 5 个词元，但此时文本长度为 10，则只有最后 5 个词元会被用作输入文本
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]  # 只关注最后一个输出的内容，因此形状会从 (batch, n_token, vocab_size) 变为 (batch, vocab_size)
        probas = torch.softmax(logits, dim=-1)  # probas 的形状为 (batch, vocab_size)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # idx_next 的形状为 (batch, 1)
        idx = torch.cat((idx, idx_next), dim=1)  # 将计算出的下一个字符的索引添加到索引数组中，此时 idx 的形状会变为 (batch, n_tokens + 1)

    return  idx


# 代码清单 5-1 用于文本到词元ID转换的工具函数
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special = {'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)   # 使用 .unsqueeze(0) 添加 batch 维度
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)    # 移除batch维度
    return tokenizer.decode(flat.tolist())

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model = model,
    idx = text_to_token_ids(start_context, tokenizer),
    max_new_tokens = 10,
    context_size = GPT_CONFIG_124M["context_length"],
)
# print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


inputs = torch.tensor([[16833, 3626, 6100],  # ["every effort moves",
                       [40, 1107, 588]])     # "I really like"]
targets = torch.tensor([[3626, 6100, 345],   # ["effort moves you",
                       [1107, 588, 11311]])  # "really like chocolate"]

with torch.no_grad():  # 屏蔽模型参数的梯度跟踪，因为我们还没开始训练
    logits = model(inputs)
probas = torch.softmax(logits, dim=-1)  # 词汇表中每个词元的概率
# print(probas.shape)


token_ids = torch.argmax(probas, dim=-1, keepdim=True)
# print("Token IDs:\n", token_ids)

# print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
# print(f"Outputs batch 1:"
#       f"{token_ids_to_text(token_ids[0].flatten(), tokenizer)}")


text_idx = 0
target_probas_1 = probas[text_idx, [0, 1, 2],  targets[text_idx]]
# print("Text 1:", target_probas_1)

text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2],  targets[text_idx]]
# print("Text 2:", target_probas_2)

log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
# print(log_probas)

avg_log_probas = torch.mean(log_probas)
# print(avg_log_probas)

neg_avg_log_probas = avg_log_probas * -1
# print(neg_avg_log_probas)

# print("Logits shape:", logits.shape)
# print("Targets shape:", targets.shape)

logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()
# print("Flattened logits:", logits_flat.shape)
# print("Flattened targets:", targets_flat.shape)

loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
# print(loss)


# 加载数据集
file_path = "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

# 检查数据集中的字符数和词元数
total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
# print("Characters:", total_characters)
# print("Tokens:", total_tokens)


train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

from torch.utils.data import Dataset

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # 对全部文本进行分词
        token_ids = tokenizer.encode(txt)
        # 使用滑动窗口将文本划分为长度为max_length的重叠序列
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
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        drop_last = drop_last,    # 如果drop_last为True且批次大小小于指定的batch_size，则会删除最后一批，以防止在训练期间出现损失剧增
        num_workers = num_workers,    # 用于预处理的CPU进程数
    )
    return dataloader

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

# # 遍历数据加载器，确保它们被正确创建
# print("Train loader:")
# for x,y in train_loader:
#     print(x.shape, y.shape)
#
# print("\nValidation loader:")
# for x,y in val_loader:
#     print(x.shape, y.shape)


# 实现一个工具函数，用于计算通过训练集加载器和验证集加载器返回的给定批次的交叉熵损失
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss

# 代码清单 5-2 用于计算训练集和验证集损失的函数
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)  # 如果你有一台支持CUDA的GPU机器，那么大语言模型将自动在GPU上训练且不需要修改代码
with torch.no_grad():  # 因为还没有开始训练，所以不使用梯度追踪，这样会更高效
    train_loss = calc_loss_loader(train_loader, model, device)  # 通过“设备”设置，可以确保所有的数据和大语言模型在同一个设备上
    val_loss = calc_loss_loader(val_loader, model, device)
print("Training loss:", train_loss)
print("Validation loss:", val_loss)