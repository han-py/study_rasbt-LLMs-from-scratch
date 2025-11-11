'''
带可训练权重的自注意力机制
'''

import torch

# 创建输入张量，表示一个包含6个词的序列，每个词由3维向量表示
# 这是一个6x3的矩阵，每一行代表一个词的嵌入向量(embedding vector)
# 例如：第一行[0.43, 0.15, 0.89]是单词"Your"的嵌入向量表示
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

x_2 = inputs[1]  # 获取第二个输入元素(索引为1)，即"journey"对应的向量 [0.55, 0.87, 0.66]
d_in = inputs.shape[1]  # 输入嵌入维度 d_in=3，表示每个词由3维向量表示
d_out = 2  # 输出维度 d_out=2，经过线性变换后的向量维度，通常是为了降维或提取更有意义的特征

# 初始化权重矩阵，设置随机种子确保结果可重现
# 在实际应用中，这些权重会在训练过程中通过反向传播进行学习和更新
# 使用 torch.manual_seed(123) 确保每次运行代码时生成的随机数相同，便于调试和复现结果
# 使用 torch.nn.Parameter 将张量标记为模型参数，使其可以被优化器更新
# requires_grad=False 表示在当前示例中不进行梯度计算，实际训练时应设为 True
torch.manual_seed(123)
# 查询权重矩阵(W_query): 用于计算查询向量(Q)，形状为(d_in, d_out)即(3, 2)
# 查询向量用于衡量当前词对序列中其他词的关注程度
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)  
# 键权重矩阵(W_key): 用于计算键向量(K)，形状为(d_in, d_out)即(3, 2)  
# 键向量用于表示词的特征，供其他词进行匹配和比较
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)    
# 值权重矩阵(W_value): 用于计算值向量(V)，形状为(d_in, d_out)即(3, 2)
# 值向量包含实际的信息内容，在注意力机制中会被加权求和
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)  

# 对单个输入向量 x_2 进行线性变换，得到对应的查询、键、值向量
# 使用矩阵乘法(@)将输入向量与相应的权重矩阵相乘
query_2 = x_2 @ W_query  # 查询向量，形状为(2,)，用于衡量当前词对其他词的关注度
keys_2 = x_2 @ W_key    # 键向量，形状为(2,)，用于表示词的特征供其他词匹配
values_2 = x_2 @ W_value # 值向量，形状为(2,)，用于加权求和得到最终的上下文表示

# 对整个输入序列进行线性变换，得到所有位置的键向量和值向量
# inputs形状为(6, 3)，W_key和W_value形状为(3, 2)
# 通过矩阵乘法得到的结果keys和values形状均为(6, 2)
# keys[i] 表示第i个词的键向量，values[i] 表示第i个词的值向量
keys = inputs @ W_key    # 所有输入词的键向量，形状为(6, 2)
values = inputs @ W_value # 所有输入词的值向量，形状为(6, 2)

# 打印键向量和值向量的形状，验证计算结果的维度是否正确
print("key.shape:", keys.shape)     # 应该是 torch.Size([6, 2])，表示6个词，每个词2维
print("value.shape:", values.shape) # 应该是 torch.Size([6, 2])，表示6个词，每个词2维