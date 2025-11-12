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
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    # 前向传播函数，接收输入张量x
    def forward(self, x):
        # 通过Linear层计算键向量
        keys = self.W_key(x)
        # 通过Linear层计算查询向量
        queries = self.W_query(x)
        # 通过Linear层计算值向量
        values = self.W_value(x)
        # 计算注意力分数：查询向量与键向量的转置相乘
        attn_scores = queries @ keys.T
        # 对注意力分数进行缩放并应用softmax函数得到注意力权重
        attn_weight = torch.softmax(
            attn_scores / (keys.shape[-1] ** 0.5), dim=-1
        )
        # 使用注意力权重对值向量进行加权求和得到上下文向量
        context_vec = attn_weight @ values
        # 返回计算得到的上下文向量
        return context_vec

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


queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(
    attn_scores / (keys.shape[-1] ** 0.5), dim=-1
)
# print(attn_weights)