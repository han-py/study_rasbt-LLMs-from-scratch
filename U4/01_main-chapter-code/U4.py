import  torch
import  torch.nn as nn

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
class DummyGPTModel(nn.modules):
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
            torch.arrange(seq_len, device = in_dex.device)
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
        super.__init__()

    def forward(self, x):
        return x