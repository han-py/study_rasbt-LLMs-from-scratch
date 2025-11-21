import torch
from torch import nn

# GPT模型配置参数
# vocab_size: 词汇表大小，GPT-2的词汇表大小为50257
# context_length: 上下文长度，表示模型一次能处理的最大token数量
# emb_dim: 嵌入维度，每个token被映射到的向量维度
# n_heads: 多头注意力机制中的注意力头数量
# n_layers: Transformer块的数量
# drop_rate: Dropout概率，用于防止过拟合
# qkv_bias: 是否在查询、键、值的线性变换中使用偏置项
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # 词汇表大小
    "context_length": 1024,  # 上下文长度
    "emb_dim": 768,          # 嵌入维度
    "n_heads": 12,           # 注意力头的数量
    "n_layers": 12,          # 层数
    "drop_rate": 0.1,        # dropout率
    "qkv_bias": False,       # 查询-键-值偏置
}

# 代码清单 4-1 一个包含占位符的GPT模型架构类
# 这是一个简化版的GPT模型，用于演示模型的基本结构
# 实际的GPT模型会使用真实的Transformer块和层归一化
class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        """
        初始化简化版GPT模型
        
        参数:
        cfg: 配置字典，包含模型超参数
        """
        super().__init__()
        # 词嵌入层：将词汇索引映射为稠密向量表示
        # 输入维度：词汇表大小，输出维度：嵌入维度
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        
        # 位置嵌入层：为序列中每个位置学习一个位置向量
        # 输入维度：上下文长度，输出维度：嵌入维度
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        
        # Dropout层：在嵌入层之后应用dropout防止过拟合
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # 使用Sequential容器包装多个Transformer块
        # 这里使用的是占位符DummyTransformerBlock
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg)
              for _ in range(cfg["n_layers"])]
        )

        # 最终的层归一化：使用占位符DummyLayerNorm
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])

        # 输出层：将嵌入维度映射回词汇表大小，用于预测下一个token
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_dex):
        """
        前向传播函数
        
        参数:
        in_dex: 输入的token索引张量，形状为(batch_size, seq_len)
        
        返回:
        logits: 未归一化的预测分数，形状为(batch_size, seq_len, vocab_size)
        """
        batch_size,seq_len = in_dex.shape
        # 获取词嵌入表示
        tok_embeds =self.tok_emb(in_dex)
        # 获取位置嵌入表示
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device = in_dex.device)
        )
        # 将词嵌入和位置嵌入相加得到最终的嵌入表示
        x = tok_embeds + pos_embeds
        # 应用dropout
        x = self.drop_emb(x)
        # 通过Transformer块处理
        x = self.trf_blocks(x)
        # 最终层归一化
        x = self.final_norm(x)
        # 通过输出层得到logits
        logits = self.out_head(x)
        return logits

# 一个简单的占位符类，稍后将被真正的TransformerBlock替换
# 占位符类不执行任何实际计算，只是直接返回输入
class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        """
        初始化占位符Transformer块
        
        参数:
        cfg: 配置字典
        """
        super().__init__()

    # 这个代码块不执行任何操作，只返回其输入
    def forward(self, x):
        """
        前向传播函数（占位符）
        
        参数:
        x: 输入张量
        
        返回:
        直接返回输入张量
        """
        return x

# 一个简单的占位符类，稍后将被真正的层归一化替换
class DummyLayerNorm(nn.Module):
    # 这里的参数只是为了模仿层归一化的接口
    def __init__(self, normalized_shape, eps=1e-5,):
        """
        初始化占位符层归一化
        
        参数:
        normalized_shape: 归一化维度
        eps: 防止除零的小常数
        """
        super().__init__()

    def forward(self, x):
        """
        前向传播函数（占位符）
        
        参数:
        x: 输入张量
        
        返回:
        直接返回输入张量
        """
        return x

# 多头注意力机制实现类
# 这是Transformer架构的核心组件，用于计算token之间的相关性
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        """
        初始化多头注意力层
        
        参数:
        d_in: 输入维度
        d_out: 输出维度
        context_length: 上下文长度
        dropout: dropout比率
        num_heads: 注意力头的数量
        qkv_bias: 是否在QKV线性变换中使用偏置
        """
        super().__init__()
        # 确保输出维度可以被头数整除
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        # 计算每个注意力头的维度
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

        # 定义查询、键、值的线性变换矩阵
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # 输出投影矩阵，用于合并所有注意力头的结果
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        # 注册因果遮罩作为缓冲区，用于防止未来信息泄露
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        """
        前向传播函数
        
        参数:
        x: 输入张量，形状为(batch_size, num_tokens, d_in)
        
        返回:
        context_vec: 注意力机制的输出，形状为(batch_size, num_tokens, d_out)
        """
        b, num_tokens, d_in = x.shape

        # 分别计算查询、键、值的线性变换
        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 通过增加num_heads维度隐式拆分矩阵
        # 将最后一个维度展开: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 转置: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 计算缩放点积注意力（自注意力）并应用因果遮罩
        attn_scores = queries @ keys.transpose(2, 3)  # 每个头的点积运算

        # 原始遮罩截断到token数量并转换为布尔值
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # 使用遮罩填充注意力分数，将被遮蔽位置的值设为负无穷
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 应用softmax获取注意力权重，并进行dropout
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 计算上下文向量: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 合并所有头，其中 self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        # 可选的输出投影
        context_vec = self.out_proj(context_vec)  

        return context_vec

# 代码清单 4-2 层归一化类
# 层归一化用于稳定神经网络训练过程，加速收敛
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        """
        初始化层归一化
        
        参数:
        emb_dim: 嵌入维度
        """
        super().__init__()
        # 防止除零错误的小常数
        self.eps = 1e-5
        # 可学习的缩放参数
        self.scale = nn.Parameter(torch.ones(emb_dim))
        # 可学习的偏移参数
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        """
        前向传播函数
        
        参数:
        x: 输入张量
        
        返回:
        归一化后的张量
        """
        # 计算沿最后一个维度的均值和方差
        mean = x.mean(dim = -1, keepdim = True)
        var = x.var(dim = -1, keepdim = True, unbiased = False)
        # 执行归一化
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        # 应用可学习的缩放和平移
        return self.scale * norm_x + self.shift

# 代码清单 4-3 GELU激活函数的实现
# GELU (Gaussian Error Linear Unit) 是一种平滑的ReLU变体
class GELU(nn.Module):
    def __init__(self):
        """
        初始化GELU激活函数
        """
        super().__init__()

    def forward(self, x):
        """
        GELU激活函数的近似实现
        
        公式: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        
        参数:
        x: 输入张量
        
        返回:
        经过GELU激活函数处理的张量
        """
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
        ))

# 代码清单 4-4 前馈神经网络模块
# Transformer中的前馈网络，用于对每个位置的表示进行非线性变换
class FeedForward(nn.Module):
    def __init__(self, cfg):
        """
        初始化前馈网络
        
        参数:
        cfg: 配置字典，必须包含emb_dim键
        """
        super().__init__()
        # 两层全连接网络，中间层维度扩大4倍
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], cfg["emb_dim"] * 4),  # 扩展维度
            GELU(),                                         # GELU激活函数
            nn.Linear(cfg["emb_dim"] * 4, cfg["emb_dim"]),  # 投影回原始维度
        )

    def forward(self, x):
        """
        前向传播函数
        
        参数:
        x: 输入张量
        
        返回:
        经过前馈网络处理的张量
        """
        return self.layers(x)

# 代码清单 4-5 用于演示快捷连接的神经网络
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        """
        初始化示例深度神经网络
        
        参数:
        layer_sizes: 每层的大小列表
        use_shortcut: 是否使用快捷连接
        """
        super().__init__()
        self.use_shortcut = use_shortcut
        # 五个层的实现
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(layer_sizes[0], layer_sizes[1]),
                GELU(),
            ),
            nn.Sequential(
                nn.Linear(layer_sizes[1], layer_sizes[2]),
                GELU(),
            ),
            nn.Sequential(
                nn.Linear(layer_sizes[2], layer_sizes[3]),
                GELU(),
            ),
            nn.Sequential(
                nn.Linear(layer_sizes[3], layer_sizes[4]),
                GELU(),
            ),
            nn.Sequential(
                nn.Linear(layer_sizes[4], layer_sizes[5]),
                GELU(),
            )
        ])

    def forward(self, x):
        """
        前向传播函数
        
        参数:
        x: 输入张量
        
        返回:
        经过网络处理的张量
        """
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x

# 实现一个用于在模型的反向传播过程中计算梯度的函数
def print_gradients(model, x):
    """
    打印模型各层权重的梯度均值
    
    参数:
    model: 要分析的模型
    x: 输入数据
    """
    output = model(x)  # 前向传播
    target = torch.tensor([[0.]])

    loss = nn.MSELoss ()
    loss = loss(output, target)  # 基于目标和输出之间的差距来计算损失

    loss.backward()  # 反向传播来计算梯度

    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")

# 代码清单 4-6 GPT 的 Transformer 块组件
# Transformer块是GPT模型的基本构建单元，包含多头注意力和前馈网络
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        """
        初始化Transformer块
        
        参数:
        cfg: 配置字典，包含模型超参数
        """
        super().__init__()
        # 多头自注意力机制
        self.att = MultiHeadAttention(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            context_length = cfg["context_length"],
            num_heads = cfg["n_heads"],
            dropout = cfg["drop_rate"],
            qkv_bias = cfg["qkv_bias"],
        )
        # 前馈神经网络
        self.ff = FeedForward(cfg)
        # 第一个层归一化，在注意力之前
        self.norm1 = LayerNorm(cfg["emb_dim"])
        # 第二个层归一化，在前馈网络之前
        self.norm2 = LayerNorm(cfg["emb_dim"])
        # 用于快捷连接的dropout
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        """
        前向传播函数
        
        参数:
        x: 输入张量，形状为(batch_size, num_tokens, emb_dim)
        
        返回:
        经过Transformer块处理的张量
        """
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

# 代码清单 4-7 GPT模型架构的实现
# 完整的GPT模型实现
class GPTModel(nn.Module):
    def __init__(self, cfg):
        """
        初始化GPT模型
        
        参数:
        cfg: 配置字典，包含模型超参数
        """
        super().__init__()
        # 词嵌入层
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # 位置嵌入层
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        # 嵌入层dropout
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # 堆叠多个Transformer块
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        # 最终层归一化
        self.final_norm = LayerNorm(cfg["emb_dim"])
        # 输出层（词预测层）
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_dex):
        """
        前向传播函数
        
        参数:
        in_dex: 输入token索引，形状为(batch_size, seq_len)
        
        返回:
        logits: 预测分数，形状为(batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = in_dex.shape
        # 获取词嵌入
        tok_embeds = self.tok_emb(in_dex)

        # 获取位置嵌入
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_dex.device)  # device 的设置允许我们在 CPU 或 GPU 上训练模型，具体取决于输入数据所在的设备
        )
        # 组合词嵌入和位置嵌入
        x = tok_embeds + pos_embeds
        # 应用嵌入层dropout
        x = self.drop_emb(x)
        # 通过所有Transformer块
        x = self.trf_blocks(x)
        # 最终层归一化
        x = self.final_norm(x)
        # 通过输出层得到logits
        logits = self.out_head(x)
        return logits

# 代码清单 4-8 GPT 模型中用于生成文本的函数
def generate_text_simple(model, idx,  # idx 是当前文本的索引数组，其形状为(batch, n_tokens)
                         max_new_tokens, context_size):
    """
    使用GPT模型生成文本
    
    参数:
    model: 训练好的GPT模型
    idx: 当前文本的索引数组，形状为(batch, n_tokens)
    max_new_tokens: 最大新生成token数量
    context_size: 上下文大小（模型最大支持的序列长度）
    
    返回:
    生成的文本索引数组
    """
    # 循环生成指定数量的新token
    for _ in range(max_new_tokens):
        # 将当前文本截断至支持的长度
        idx_cond = idx[:, -context_size:]  
        # 禁用梯度计算以提高效率
        with torch.no_grad():
            logits = model(idx_cond)

        # 只关注最后一个位置的输出
        logits = logits[:, -1, :]  
        # 应用softmax获取概率分布
        probas = torch.softmax(logits, dim=-1)  
        # 选择概率最高的token作为下一个token
        idx_next = torch.argmax(probas, dim=-1, keepdim=True) 
        # 将新token添加到序列中
        idx = torch.cat((idx, idx_next), dim=1)  

    return  idx