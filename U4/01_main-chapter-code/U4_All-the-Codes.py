import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import torch
import torch.nn as nn
import tiktoken


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
class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # 使用占位符替换 TransformerBlock
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg)
              for _ in range(cfg["n_layers"])]
        )

        # 使用占位符替换层归一化
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])

        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_dex):
        batch_size,seq_len = in_dex.shape
        tok_embeds =self.tok_emb(in_dex)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device = in_dex.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

# 一个简单的占位符类，稍后将被真正的TransformerBlock替换
class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    # 这个代码块不执行任何操作，只返回其输入
    def forward(self, x):
        return x

# 一个简单的占位符类，稍后将被真正的层归一化替换
class DummyLayerNorm(nn.Module):
    # 这里的参数只是为了模仿层归一化的接口
    def __init__(self, normalized_shape, eps=1e-5,):
        super().__init__()

    def forward(self, x):
        return x


# tokenizer = tiktoken.get_encoding("gpt2")
# batch = []
# txt1 = "Every effort moves you"
# txt2 = "Every day holds a"
#
# batch.append(torch.tensor(tokenizer.encode(txt1)))
# batch.append(torch.tensor(tokenizer.encode(txt2)))
# batch = torch.stack(batch, dim = 0)
# # print(batch)


# torch.manual_seed(123)
# model = DummyGPTModel(GPT_CONFIG_124M)
# logits = model(batch)
# # print("Output shape:", logits.shape)
# # print(logits)


# torch.manual_seed(123)
# batch_example = torch.randn(2, 5)  # 创建两个训练样本，每个样本包含5个维度
# layer = nn.Sequential(nn.Linear(5,6), nn.ReLU())
# out = layer(batch_example)
# # print(out)
# mean = out.mean(dim=-1, keepdim = True)
# var = out.var(dim=-1, keepdim = True)
# # print("Mean:\n", mean)
# # print("Var:\n", var)
# out_norm =(out - mean) / torch.sqrt(var)
# mean = out_norm.mean(dim=-1, keepdim = True)
# var = out_norm.var(dim=-1, keepdim = True)
# # print("Normalized layer Outputs:\n", out_norm)
# # print("Mean:\n", mean)
# # print("Variance:\n", var)
# # 为了提高可读性，可以通过将 sci_mode 设置为 False 来关闭科学计数法，从而在打印张量值时避免使用科学记数法
# torch.set_printoptions(sci_mode=False)
# print("Mean:\n", mean)
# print("Variance:\n", var)


# 代码清单 4-2 层归一化类
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


# batch_example = torch.randn(2, 5)
# ln = LayerNorm(emb_dim=5)
# out_ln = ln(batch_example)
# mean = out_ln.mean(dim = -1, keepdim = True)
# var = out_ln.var(dim = -1, unbiased = False, keepdim = True)
# print("Mean:\n", mean)
# print("Variance:\n", var)


# 代码清单 4-3 GELU激活函数的实现
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
        ))


# # 为了直观地比较 GELU 函数与 RELU 函数，可以将它们并排绘制出来
# import matplotlib.pyplot as plt
# gelu , relu = GELU(), nn.ReLU()
# # 在 -3 和 3 之间创建 100 个样本数据点
# x = torch.linspace(-3, 3, 100)
# y_gelu, y_relu = gelu(x), relu(x)
# plt.figure(figsize = (8, 3))
# for i, (y, label) in enumerate(zip([ y_gelu, y_relu], ["GELU", "ReLU"]), 1):
#     plt.subplot(1, 2, i)
#     plt.plot(x, y)
#     plt.title(f"{label} activation function")
#     plt.xlabel("x")
#     plt.ylabel(f"{label}(x)")
#     plt.grid(True)
# plt.tight_layout()
# plt.show()


# 代码清单 4-4 前馈神经网络模块
class FeedForwardNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], cfg["emb_dim"] * 4),
            GELU(),
            nn.Linear(cfg["emb_dim"] * 4, cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


# ffn = FeedForwardNN(GPT_CONFIG_124M)
# x = torch.rand(2, 3, 768)  # 创建批次维度为 2 的样本输入
# out = ffn(x)
# print(out.shape)


# 代码清单 4-5 用于演示快捷连接的神经网络
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
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
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x


# layer_sizes = [3, 3, 3, 3, 3, 1]
# sample_input = torch.tensor([[1., 0., -1.]])
# torch.manual_seed(123)  # 指定随机种子用于初始化权重，以确保结果可复现
# model_without_shortcut = ExampleDeepNeuralNetwork(
#     layer_sizes, use_shortcut= False
# )


# 实现一个用于在模型的反向传播过程中计算梯度的函数
def print_gradients(model, x):
    output = model(x)  # 前向传播
    target = torch.tensor([[0.]])

    loss = nn.MSELoss ()
    loss = loss(output, target)  # 基于目标和输出之间的差距来计算损失

    loss.backward()  # 反向传播来计算梯度

    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")


# print_gradients(model_without_shortcut, sample_input)


# 实例化一个包含跳跃连接的模型，并观察它的比较结果
layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1., 0., -1.]])
torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)
print_gradients(model_with_shortcut, sample_input)