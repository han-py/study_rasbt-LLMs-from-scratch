import urllib.request
import re

from numpy import integer

# url = ("https://raw.githubusercontent.com/rasbt/"
#        "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
#        "the-verdict.txt")
# file_path = "the-verdict.txt"
# urllib.request.urlretrieve(url, file_path)


# 代码清单2-1 通过Python读取短篇小说THE VERDICT作为文本样本
with open ("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
# print("Total number of character:", len(raw_text))
# print(raw_text[:99])

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
# 去除空白符
preprocessed = [item.strip() for item in preprocessed if item.strip()]
# print("Total number of tokens:", len(preprocessed))
# print(preprocessed[:30])

# 确定词汇表的大小
all_words = sorted(set(preprocessed))
# print("Total number of unique words:", len(all_words))


# 代码清单2-2 创建词汇表
# 添加<unk>标记来处理未知词
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>","<|unk|>"])  # 添加未知词标记
vocab = {token:integer for integer, token in enumerate(all_tokens)}
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
tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace."
)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)
strings = tokenizer.decode(integers)
print(strings)