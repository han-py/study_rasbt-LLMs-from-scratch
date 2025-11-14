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
    '''
    高效的多头注意力实现类
    
    与MultiHeadAttentionWrapper不同，这个实现不创建多个独立的注意力头，
    而是在单个矩阵运算中并行处理所有头，这样更加高效。
    '''
    
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        '''
        初始化高效的多头注意力层
        
        参数:
        d_in: 输入维度
        d_out: 输出维度（必须能被num_heads整除）
        context_length: 上下文长度（序列最大长度）
        dropout: dropout比率
        num_heads: 注意力头的数量
        qkv_bias: 是否在查询、键、值的线性变换中使用偏置
        '''
        super().__init__()
        # 确保输出维度能被头数整除
        assert(d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        # 计算每个头的维度
        self.head_dim = d_out // num_heads # 减少投影维度以匹配所需的输出维度
        # 定义查询、键、值的线性变换层（所有头共享这些权重矩阵）
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # 输出投影层，用于组合所有头的输出
        self.out_proj = nn.Linear(d_out, d_out) # 使用一个线性层来组合头的输出
        # 定义dropout层
        self.dropout = nn.Dropout(dropout)
        # 创建并注册上三角掩码缓冲区
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        '''
        前向传播函数
        
        参数:
        x: 输入张量，形状为(batch_size, num_tokens, d_in)
        
        返回:
        context_vec: 多头注意力输出，形状为(batch_size, num_tokens, d_out)
        '''
        b, num_tokens, d_in = x.shape
        
        # 计算键、查询、值向量
        keys = self.W_key(x)      # 形状: (batch_size, num_tokens, d_out)
        queries = self.W_query(x) # 形状: (batch_size, num_tokens, d_out)
        values = self.W_value(x)  # 形状: (batch_size, num_tokens, d_out)
        
        # 将键、查询、值向量重塑为多头形式
        # view操作将最后一个维度(d_out)分解为(num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)      # 形状: (batch_size, num_tokens, num_heads, head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim) # 形状: (batch_size, num_tokens, num_heads, head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)   # 形状: (batch_size, num_tokens, num_heads, head_dim)

        # 转置操作，将num_heads维度移到前面，便于并行计算
        keys = keys.transpose(1, 2)      # 形状: (batch_size, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2) # 形状: (batch_size, num_heads, num_tokens, head_dim)
        values = values.transpose(1, 2)   # 形状: (batch_size, num_heads, num_tokens, head_dim)

        # 计算注意力分数
        # queries @ keys.transpose(2, 3)执行批量矩阵乘法
        # 结果形状: (batch_size, num_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(2, 3)
        
        # 获取掩码并应用到注意力分数上
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # 应用因果掩码
        attn_scores.masked_fill(mask_bool, -torch.inf)

        # 对注意力分数进行缩放并应用softmax得到注意力权重
        # 注意这里使用的是head_dim而不是d_out作为缩放因子
        attn_weights = self.dropout(torch.softmax(
            attn_scores / (self.head_dim ** 0.5), dim=-1
        ))

        # 使用注意力权重对值向量进行加权求和
        # 结果形状: (batch_size, num_heads, num_tokens, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 重新整理张量形状
        # contiguous()确保内存连续，view()需要连续的内存
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        
        # 应用输出投影层
        context_vec = self.out_proj(context_vec) # 添加一个可选的线性投影
        return context_vec

'''
总结:
多头注意力机制工作原理如下：

1. 并行处理：不是只计算一组查询、键和值，而是并行运行多个注意力头。
   每个头都有自己的权重矩阵，可以学习不同的注意力模式。

2. 独立计算：每个注意力头独立计算自己的注意力权重和输出。

3. 特征分割：通常每个头处理输入特征的一个子空间，这样不同的头可以关注不同类型的语义信息。

4. 结果合并：将所有头的输出连接在一起或通过线性投影层进行整合，形成最终的多头注意力输出。

两种实现方式的区别：
- MultiHeadAttentionWrapper：创建多个独立的注意力头，分别计算后再连接结果，直观但效率较低
- MultiHeadAttention：在单个矩阵运算中并行处理所有头，效率更高，是实际应用中的标准实现

优势：
- 允许模型同时关注来自不同位置的信息
- 增强了模型关注不同类型关系的能力
- 提高了模型的表达能力和泛化性能
'''