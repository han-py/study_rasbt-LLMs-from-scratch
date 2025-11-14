'''
将单头注意力扩展到多头注意力
'''

import torch
from torch import nn


inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89], # Your - 第1个词的3维嵌入向量
        [0.55, 0.87, 0.66], # journey - 第2个词的3维嵌入向量
        [0.57, 0.85, 0.64], # starts - 第3个词的3维嵌入向量
        [0.22, 0.58, 0.33], # with - 第4个词的3维嵌入向量
        [0.77, 0.25, 0.10], # one - 第5个词的3维嵌入向量
        [0.05, 0.80, 0.55]  # step - 第6个词的3维嵌入向量
    ]
)
batch = torch.stack((inputs, inputs), dim=0)
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)

        attn_scores.masked_fill(
            self.mask.bool()[:num_tokens, :num_tokens],  # 只使用前num_tokens行和列的掩码
            -torch.inf
        )

        attn_weights = self.dropout(torch.softmax(
            attn_scores / (keys.shape[-1] ** 0.5), dim=-1  # 缩放因子为维度的平方根
        ))

        context_vec = attn_weights @ values
        return context_vec


'''
叠加多个单头注意力层
'''
# 代码清单 3—4 一个实现多头注意力的封装类
class MultiHeadAttentionWrapper(nn.Module):
    '''
    多头注意力包装类
    
    该类通过组合多个独立的因果注意力头来实现多头注意力机制。
    每个头学习不同的注意力模式，最后将所有头的输出连接起来。
    '''
    
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        '''
        初始化多头注意力层
        
        参数:
        d_in: 输入维度
        d_out: 每个头的输出维度（总输出维度为num_heads * d_out）
        context_length: 上下文长度（序列最大长度）
        dropout: dropout比率
        num_heads: 注意力头的数量
        qkv_bias: 是否在查询、键、值的线性变换中使用偏置
        '''
        super().__init__()
        self.heads = nn.ModuleList(
            [
                CausalAttention(
                    d_in, d_out, context_length, dropout, qkv_bias
                )
                for _ in range(num_heads)
            ]
        )

    def forward(self, x):
        '''
        前向传播函数
        
        参数:
        x: 输入张量，形状为(batch_size, num_tokens, d_in)
        
        返回:
        out: 多头注意力输出，形状为(batch_size, num_tokens, num_heads * d_out)
        '''
        return torch.cat([head(x) for head in self.heads], dim=-1)

torch.manual_seed(123)
context_length = batch.shape[1] # 这是词元的数量
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, dropout=0.0, num_heads=2)
context_vec = mha(batch)
# print("context_vec.shape:", context_vec.shape)
# print("context_vec:", context_vec)


'''
通过权重划分实现多头注意力
'''
# 代码清单 3—5 一个高效的多头注意力类
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert(d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # 减少投影维度以匹配所需的输出维度
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out) # 使用一个线性层来组合头的输出
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill(mask_bool, -torch.inf)

        attn_weights = self.dropout(torch.softmax(
            attn_scores / (self.head_dim ** 0.5), dim=-1
        ))

        context_vec =(attn_weights @ values).transpose(1, 2)

        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # 添加一个可选的线性投影
        return context_vec