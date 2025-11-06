import urllib.request
import re

from numpy import integer

# url = ("https://raw.githubusercontent.com/rasbt/"
#        "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
#        "the-verdict.txt")
# file_path = "the-verdict.txt"
# urllib.request.urlretrieve(url, file_path)


# 代码清单2-1 通过Python读取短篇小说THE VERDICT作为文本样本
# with open ("the-verdict.txt", "r", encoding="utf-8") as f:
#     raw_text = f.read()
# print("Total number of character:", len(raw_text))
# print(raw_text[:99])

# preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
# 去除空白符
# preprocessed = [item.strip() for item in preprocessed if item.strip()]
# print("Total number of tokens:", len(preprocessed))
# print(preprocessed[:30])

# 确定词汇表的大小
# all_words = sorted(set(preprocessed))
# print("Total number of unique words:", len(all_words))


# 代码清单2-2 创建词汇表
# 添加<unk>标记来处理未知词
# all_tokens = sorted(list(set(preprocessed)))
# all_tokens.extend(["<|endoftext|>","<|unk|>"])  # 添加未知词标记
# vocab = {token:integer for integer, token in enumerate(all_tokens)}
# print("Total number of unique words:", len(vocab))
# for i, item in enumerate(list(vocab.items())[-5:]):
#     print(item)
# for i, item in enumerate(vocab.items()):
#     print(item)
#     if i >= 50:
#         break


# 代码清单2-3 实现简单的文本分词器
class SimpleTokenizerV1:
    """简单的文本分词器版本1

    该分词器将字符串转换为整数序列，以及将整数序列转换回字符串。
    它使用一个词汇表字典来进行字符串和整数之间的映射。
    """

    def __init__(self, vocab):
        """初始化分词器

        Args:
            vocab (dict): 词汇表，键是词汇(token)，值是对应的整数ID
        """
        # 字符串到整数的映射
        self.str_to_int = vocab
        # 整数到字符串的映射（用于解码）
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        """将文本编码为整数序列

        Args:
            text (str): 需要编码的原始文本

        Returns:
            list: 整数ID列表
        """
        # 使用正则表达式分割文本，保留分隔符
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        # 移除空白字符并清理预处理后的标记
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        # 将标记转换为对应的整数ID
        ids = [self.str_to_int[token] for token in preprocessed]
        return ids

    def decode(self, ids):
        """将整数序列解码为文本

        Args:
            ids (list): 整数ID列表

        Returns:
            str: 解码后的文本
        """
        # 将整数ID转换为对应的文字，并用空格连接
        text = " ".join([self.int_to_str[i] for i in ids])
        # 移除标点符号前的空格，使文本更自然
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

# # 测试分词器
# tokenizer = SimpleTokenizerV1(vocab)
# text = """"It's the last he painted, you know,"Mrs.Gisburn said with pardonable pride."""
# ids = tokenizer.encode(text)
# print(ids)
# print(tokenizer.decode(ids))
# # 如果有训练集之外的新样本，则会报错 KeyError: 'Hello'
# text = "Hello, do you like tea?"
# print(tokenizer.encode(text))


# 代码清单2-4 能够处理未知单词的文本分词器
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]

        # 用<|unk|>词元替换未知单词
        preprocessed = [
            item if item in self.str_to_int
            else "<|unk|>"
            for item in preprocessed
        ]

        ids = [self.str_to_int[token] for token in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

# # 测试分词器V2
# text1 = "Hello, do you like tea?"
# text2 = "In the sunlit terraces of the palace."
# text = "<|endoftext|> ".join([text1, text2])
# print( text)
# tokenizer = SimpleTokenizerV2(vocab)
# print(tokenizer.encode(text))
# print(tokenizer.decode(tokenizer.encode(text)))

from importlib.metadata import version
import tiktoken
# print("tiktoken version:", version("tiktoken"))

# 实例化tiktoken中的BPE分词器
# tokenizer = tiktoken.get_encoding("gpt2")

# text = (
#     "Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace."
# )
# integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# print(integers)
# strings = tokenizer.decode(integers)
# print(strings)

# with open ("the-verdict.txt", "r", encoding="utf-8") as f:
#     raw_text = f.read()
#
# enc_text = tokenizer.encode(raw_text)
# print("Total number of tokens:", len(enc_text))
#
# enc_sample = enc_text[50:]

# 上下文大小决定的输入中包含了多少个词元
context_size = 4
# x = enc_sample[: context_size]
# y = enc_sample[1 : context_size + 1]
# print(f"x: {x}")
# print(f"y: {y}")

# for i in range(1, context_size + 1):
#     context = enc_sample[:i]
#     desired = enc_sample[i]
#     print(f"{context} ----> {desired}")

# for i in range(1, context_size + 1):
#     context = enc_sample[:i]
#     desired = enc_sample[i]
#     print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))


# 代码清单2-5 一个用于批处理输入和目标的数据集
import torch
from torch.utils.data import Dataset

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # 对全部文本进行分词
        token_ids = tokenizer.encode(txt)
        # 使用滑动窗口将文本划分为长度为max_length的重叠序列
        for i in range(0, len(token_ids) - max_length, stride):
            input_ids = token_ids[i: i + max_length]
            target_ids = token_ids[i + 1: i + 1 + max_length]
            self.input_ids.append(torch.tensor(input_ids))
            self.target_ids.append(torch.tensor(target_ids))

    # 返回数据集的总行数
    def __len__(self):
        return len(self.input_ids)

    # 返回数据集的指定行
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


# 代码清单2-6 用于批量生成输入-目标对的数据加载器
def create_dataloader_V1(
        txt, batch_size = 4, max_length = 256, stride = 128, shuffle = True, drop_last = True, num_workers = 0
):
    # 初始化分词器
    tokenizer = tiktoken.get_encoding("gpt2")
    # 创建数据集
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        drop_last = drop_last,    # 如果drop_last为True且批次大小小于指定的batch_size，则会删除最后一批，以防止在训练期间出现损失剧增
        num_workers = num_workers,    # 用于预处理的CPU进程数
    )
    return dataloader

with open ("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader_V1(
    raw_text, batch_size = 1, max_length = 4, stride = 1, shuffle = False
)
# 将dataloader转换为Python迭代器,以通过Python内置的next()函数获取下一个条目
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)
second_batch = next(data_iter)
print(second_batch)