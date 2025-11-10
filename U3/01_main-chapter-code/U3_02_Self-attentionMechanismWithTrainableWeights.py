'''
带可训练权重的自注意力机制
'''

import torch

inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89], # Your
        [0.55, 0.87, 0.66], # journey
        [0.57, 0.85, 0.64], # starts
        [0.22, 0.58, 0.33], # with
        [0.77, 0.25, 0.10], # one
        [0.05, 0.80, 0.55] # step
    ]
)

x_2 =  inputs[1] # 第二个输入元素
d_in = inputs.shape[1] # 输入嵌入维度d_in=3
d_out = 2 # 输出维度d_out=2

# 初始化权重矩阵
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False) # 设置requires_grad=False以减少输出中的其他项，但如果要在模型训练中使用这些权重矩阵，就需要设置requires_grad=True，以便再训练中更新这些矩阵
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

# 计算查询向量、键向量和值向量
query_2 = x_2 @ W_query
keys = x_2 @ W_key
values = x_2 @ W_value
print("Query vector:", query_2)