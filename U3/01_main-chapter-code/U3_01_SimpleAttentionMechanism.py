'''
简单注意力机制
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

# 第二个输入词元作为查询向量
query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
# print(attn_scores_2)

'''
进行归一化处理
'''
# attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
# print("Attention weights:", attn_weights_2_tmp)
# print("Sum:", attn_weights_2_tmp.sum())

def softmax_naive(x): # 简单的softmax实现，可能会遇到数值稳定性问题，比如溢出和下溢。实践中建议使用softmax的Pytorch实现
    return torch.exp(x) / torch.exp(x).sum(dim=0)

# attn_weights_2_naive = softmax_naive(attn_scores_2)
# print("Attention weights:", attn_weights_2_naive)
# print("Sum:", attn_weights_2_naive.sum())

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
# print("Attention weights:", attn_weights_2)
# print("Sum:", attn_weights_2.sum())

# 第二个输入词元作为查询向量
query = inputs[1]
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i
# print("Context vector:", context_vec_2)

# 计算所有输入词元的注意力权重
# attn_scores = torch.empty(6, 6)
# for i, x_i in enumerate(inputs):
#     for j, x_j in enumerate(inputs):
#         attn_scores[i, j] = torch.dot(x_i, x_j)
# print(attn_scores)
# 使用矩阵乘法更快
attn_scores = inputs @ inputs.T
# print(attn_scores)

# 对每一行进行归一化
attn_weights = torch.softmax(attn_scores, dim=-1)
# print(attn_weights)

# row_2_sum = sum([0.1390, 0.2369, 0.2326, 0.1242, 0.1108, 0.1565])
# print("Row 2 sum:", row_2_sum)
# print("All row sums:", attn_weights.sum(dim=-1))

# 用注意力权重通过矩阵乘法计算出所有上下文向量
all_context_vecs = attn_weights @ inputs
print("All context vectors:", all_context_vecs)
print("Previous 2nd context vector:", context_vec_2)