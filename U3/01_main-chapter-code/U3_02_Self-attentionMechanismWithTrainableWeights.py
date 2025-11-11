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
# print("key.shape:", keys.shape)     # 应该是 torch.Size([6, 2])，表示6个词，每个词2维
# print("value.shape:", values.shape) # 应该是 torch.Size([6, 2])，表示6个词，每个词2维

# 注意：Python从0开始进行检索
# 获取索引为1的键向量(keys_2)，即"journey"对应的键向量
keys_2 = keys[1]
# 计算查询向量query_2和键向量keys_2的点积，得到注意力分数
# 点积越大表示两个向量越相似，即"journey"对自身的关注度越高
attn_score_22 = query_2.dot(keys_2)
# print(attn_score_22)

# 给定query的全部注意力分数
# 通过矩阵乘法计算query_2与所有键向量(keys.T)的点积，得到注意力分数向量
# keys.T是keys的转置，形状从(6, 2)变为(2, 6)
# query_2 @ keys.T的结果形状为(6,)，表示"journey"对序列中每个词的注意力分数
attn_scores_2 = query_2 @ keys.T
# print(attn_scores_2)

# 获取键向量的维度，用于缩放注意力分数，防止softmax函数进入饱和区域
d_k = keys.shape[-1]
# 对注意力分数进行缩放(除以sqrt(d_k))并应用softmax函数，得到注意力权重
# 缩放因子sqrt(d_k)有助于在维度较大时保持梯度的稳定性
# softmax函数将注意力分数转换为概率分布，所有权重之和为1
# dim=-1表示在最后一个维度上进行softmax操作
attn_weights_2 = torch.softmax(attn_scores_2 / d_k ** 0.5, dim=-1)
# print(attn_weights_2)

# 使用注意力权重对值向量进行加权求和，得到最终的上下文向量
# attn_weights_2形状为(6,)，values形状为(6, 2)
# 结果context_vec_2形状为(2,)，表示考虑了上下文信息的"journey"词向量表示
context_vec_2 = attn_weights_2 @ values
print(context_vec_2)

'''
总结:
本代码演示了带可训练权重的自注意力机制的完整计算过程：

1. 输入表示：使用一个6词序列，每个词由3维向量表示

2. 权重矩阵初始化：
   - W_query (查询权重): 用于计算查询向量(Q)
   - W_key (键权重): 用于计算键向量(K)
   - W_value (值权重): 用于计算值向量(V)
   
3. 线性变换：
   - 对输入序列分别应用三个权重矩阵，得到对应的Q、K、V向量

4. 注意力计算流程：
   - 计算查询向量与所有键向量的点积得到注意力分数
   - 对注意力分数进行缩放处理(除以sqrt(d_k))
   - 应用softmax函数将分数转换为概率分布(注意力权重)
   - 使用注意力权重对值向量进行加权求和得到最终的上下文向量

这种机制允许模型在处理序列数据时，动态地关注输入序列的不同部分，
从而更好地捕捉词与词之间的依赖关系。
'''