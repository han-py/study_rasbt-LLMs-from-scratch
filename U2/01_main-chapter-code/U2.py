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
vocab = {token:integer for integer, token in enumerate(all_words)}
# for i, item in enumerate(vocab.items()):
#     print(item)
#     if i >= 50:
#         break

