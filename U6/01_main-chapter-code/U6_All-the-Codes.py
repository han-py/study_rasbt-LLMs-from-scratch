# 代码清单 6-1 下载和解压数据集
import urllib.request
import os
import zipfile
from pathlib import Path

from datasets.utils import extract
from flatbuffers import encode
from pandas.core.common import random_state
from patsy import origin

url = "http://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

def download_and_unzip_spam_data(
        url, zip_path, extracted_path, data_file_path
):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return

    with urllib.request.urlopen(url) as response: # 下载文件
        with open(zip_path, 'wb') as out_file:
            out_file.write(response.read())

    with zipfile.ZipFile(zip_path, 'r') as zip_ref: # 解压文件
        zip_ref.extractall(extracted_path)

    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path) # 添加 .tsv 文件扩展名
    print(f"File downloaded and saved as {data_file_path}")

download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)


import pandas as pd
df = pd.read_csv(
    data_file_path,
    sep="\t",
    header=None,
    names=["Label", "Text"]
)
# print(df)
# print(df.Label.value_counts())


# 代码清单 6-2 创建一个平衡的数据集
def create_balanced_dataset(df):
    num_spam =df[df["Label"] == "spam"].shape[0] # 统计“垃圾信息”的样本数量
    ham_subset = df[df["Label"] == "ham"].sample(
        num_spam,
        random_state=123
    ) # 随机采样“非垃圾消息”，使其数量与“垃圾消息”一致
    balanced_df = pd.concat([
        ham_subset, df[df["Label"] == "spam"]
    ]) # 将“垃圾消息”与采样后的“非垃圾消息”组合，构成平衡数据集
    return balanced_df

balanced_df = create_balanced_dataset(df)
# print(balanced_df["Label"].value_counts())

balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1}) # 将标签转换为数值形式，方便后续模型训练


# 代码清单 6-3 划分数据集
def random_split(df, train_frac, validation_frac):

    df = df.sample(
        frac = 1, random_state = 123
    ).reset_index(drop = True) # 打乱整个 Dataframe
    train_end = int(len(df) * train_frac) # 计算拆分索引
    validation_end = train_end + int(len(df) * validation_frac)

    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df

# train_df, validation_df, test_df = random_split(
#     balanced_df, 0.7, 0.1
# ) # 作为剩余部分，测试集比例被隐含设置为0.2
#
# train_df.to_csv("train.csv", index=None)
# validation_df.to_csv("validation.csv", index=None)
# test_df.to_csv("test.csv", index=None)


import  tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
# print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))


# 代码清单 6-4 构建一个 PyTorch Dataset 类
import torch
from torch.utils.data import Dataset

class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        # 文本分词
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # 如果序列长度超过 max_length，则进行截断
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        # 填充到最长序列的长度
        self.encoded_texts = [
            encoded_text + [pad_token_id] *
            (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length

train_dataset = SpamDataset(
    csv_file="train.csv",
    max_length=None,
    tokenizer=tokenizer
)
# print(train_dataset.max_length)
val_dataset = SpamDataset(
    csv_file="validation.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)
test_dataset = SpamDataset(
    csv_file="test.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)


# 代码清单 6-5 在PyTorch 中船舰数据加载器
from torch.utils.data import DataLoader

num_workers = 0 # 此设置确保了与大多数计算机的兼容性
batch_size = 8
torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last= True,
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last= False,
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last= False,
)

for input_batch, target_batch in train_loader:
    pass
# print("Input batch dimensions:", input_batch.shape)
# print("Target batch dimensions:", target_batch.shape)

# print(f"{len(train_loader)} training batches")
# print(f"{len(val_loader)} validation batches")
# print(f"{len(test_loader)} test batches")


CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"
BASE_CONFIG = {
    "vocab_size": 50257, # 词汇表大小
    "context_length": 1024, # 上下文长度
    "drop_rate": 0.0, # dropout率
    "qkv_bias": True # 查询-键-值偏置
}
model_configs = {
    "gpt2-small(124M)":{"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium(355M)":{"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large(774M)":{"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl(1558M)":{"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])