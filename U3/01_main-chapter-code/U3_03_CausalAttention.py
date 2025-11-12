'''
利用因果注意力隐藏未来词汇
'''
import torch
import torch.nn as nn


class SelfAttention_V2(nn.Module):
    # 构造函数，接收输入维度d_in、输出维度d_out和是否使用偏置项qkv_bias作为参数
    def __init__(self, d_in, d_out, qkv_bias=False):
        # 调用父类nn.Module的构造函数
        super().__init__()
        # 使用PyTorch内置的Linear层定义查询、键、值的线性变换
        # Linear层会自动创建可学习的权重和偏置参数
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)  # 查询权重矩阵
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)    # 键权重矩阵
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)  # 值权重矩阵

    # 前向传播函数，接收输入张量x
    def forward(self, x):
        # 通过Linear层计算键向量，形状为[len_seq, d_out]
        keys = self.W_key(x)
        # 通过Linear层计算查询向量，形状为[len_seq, d_out]
        queries = self.W_query(x)
        # 通过Linear层计算值向量，形状为[len_seq, d_out]
        values = self.W_value(x)
        # 计算注意力分数：查询向量与键向量的转置相乘
        # 结果形状为[len_seq, len_seq]，表示序列中每个元素与其他所有元素的相关性得分
        attn_scores = queries @ keys.T
        # 对注意力分数进行缩放并应用softmax函数得到注意力权重
        # 缩放因子是维度的平方根，这有助于在高维空间中稳定梯度
        attn_weight = torch.softmax(
            attn_scores / (keys.shape[-1] ** 0.5), dim=-1  # 在最后一个维度上进行softmax归一化
        )
        # 使用注意力权重对值向量进行加权求和得到上下文向量
        # 输出形状为[len_seq, d_out]
        context_vec = attn_weight @ values
        # 返回计算得到的上下文向量
        return context_vec

# 定义输入数据：6个单词的3维嵌入向量
# 每一行代表一个单词的词嵌入，例如第一行[0.43, 0.15, 0.89]是单词"Your"的嵌入向量
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

d_in = inputs.shape[1]  # 输入嵌入维度 d_in=3，表示每个词由3维向量表示
d_out = 2  # 输出维度 d_out=2，经过线性变换后的向量维度，通常是为了降维或提取更有意义的特征

# 设置随机种子以确保结果可重现
torch.manual_seed(789)
# 创建SelfAttention_V2实例，传入输入维度和输出维度
sa_v2 = SelfAttention_V2(d_in, d_out)


'''
因果注意力的掩码实现
'''
# 计算查询和键向量
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)

# 计算注意力分数矩阵
attn_scores = queries @ keys.T

# 应用softmax得到注意力权重（这里是没有因果掩码的标准注意力权重）
attn_weights = torch.softmax(
    attn_scores / (keys.shape[-1] ** 0.5), dim=-1
)
# print(attn_weights)

# 获取上下文长度（即序列长度）
context_length = attn_scores.shape[0]

# 创建下三角矩阵作为简单掩码（方法1）
# torch.tril会保留矩阵的下三角部分（包括对角线），其余位置置0
mask_simple = torch.tril(torch.ones(context_length, context_length))
# print(mask_simple)

# 将注意力权重与简单掩码相乘，屏蔽未来位置的信息
mask_simple *= attn_weights
# print(mask_simple)

# 对每行进行归一化，使每行的权重和为1
row_sums = mask_simple.sum(dim=-1, keepdim=True)
mask_simple_norm = mask_simple / row_sums
# print(mask_simple_norm)

# 创建上三角矩阵作为掩码（方法2，推荐方法）
# torch.triu会保留矩阵的上三角部分（diagonal=1表示从主对角线上方开始）
# 这种方式更适合用于因果注意力，因为可以将未来位置设置为负无穷
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)

# 使用掩码填充注意力分数，将未来位置的分数设为负无穷
# 这样在经过softmax后，这些位置的权重就会接近0
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
# print(masked)

# 对掩码后的注意力分数应用softmax，得到最终的因果注意力权重
# 现在每个位置只能关注到它之前和当前位置的信息
attn_weights = torch.softmax(
    masked / (keys.shape[-1] ** 0.5), dim=-1
)
# print(attn_weights)


'''
利用dropout掩码额外的注意力权重
'''
torch.manual_seed(123)
dropout = nn.Dropout(p=0.5) # 选择使用50%的dropout率
# example = torch.ones(6, 6) # 在这里创建一个全1矩阵
# print(dropout(example))

torch.manual_seed(123)
# print(dropout(attn_weights))


'''
实现一个简化的因果注意力类
'''
batch = torch.stack((inputs, inputs), dim=0)
# print(batch.shape) # 两个输入，每个输入有6个词元，每个词元的嵌入维度为3

# 代码清单 3-3 一个简化的因果注意力类
class CausalSelfAttention(nn.Module):
    '''
    因果自注意力机制类
    
    该类实现了因果注意力机制，确保在处理序列时只能访问当前位置及之前位置的信息，
    通过上三角掩码防止未来信息泄露。
    '''
    
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        '''
        初始化因果自注意力层
        
        参数:
        d_in: 输入维度
        d_out: 输出维度
        context_length: 上下文长度（序列长度）
        dropout: dropout比率
        qkv_bias: 是否在查询、键、值的线性变换中使用偏置
        '''
        super().__init__()
        # 定义查询、键、值的线性变换层
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # 定义dropout层
        self.dropout = nn.Dropout(dropout)

        # 创建并注册上三角掩码缓冲区，防止未来信息泄露
        # torch.triu创建上三角矩阵，diagonal=1表示从主对角线上方开始
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
        context_vec: 上下文向量，形状为(batch_size, num_tokens, d_out)
        '''
        # 获取输入张量的形状信息
        b, num_tokens, d_in = x.shape
        
        # 计算键、查询、值向量
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 计算注意力分数: 查询向量与键向量的转置相乘
        # 使用transpose(1, 2)对键向量进行转置操作
        attn_scores = queries @ keys.transpose(1, 2)
        
        # 应用因果掩码，将未来位置的注意力分数设为负无穷
        # 这样在softmax后这些位置的权重会接近0
        attn_scores.masked_fill(
            self.mask.bool()[:num_tokens, :num_tokens],  # 只使用前num_tokens行和列的掩码
            -torch.inf
        )
        
        # 对注意力分数进行缩放并应用softmax和dropout得到注意力权重
        attn_weights = self.dropout(torch.softmax(
            attn_scores / (keys.shape[-1] ** 0.5), dim=-1  # 缩放因子为维度的平方根
        ))

        # 使用注意力权重对值向量进行加权求和得到上下文向量
        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(123)
context_length = batch.shape[1]
# 创建因果自注意力实例
ca = CausalSelfAttention(d_in, d_out, context_length, dropout=0.0)
# 对批处理数据应用因果注意力
context_vecs = ca(batch)
print("context_vecs.shape:", context_vecs.shape)