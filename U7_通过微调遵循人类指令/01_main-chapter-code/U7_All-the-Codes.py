# 代码清单 7-1 下载数据集
import json
import os
import urllib

from openpyxl.formula import tokenizer
from torch.onnx.symbolic_opset9 import tensor


def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            test_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(test_data)
    with open (file_path, "r") as file:
        data = json.load(file)
    return  data

file_path = "instruction-data.json"
url = (
    "http://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)

data = download_and_load_file(file_path, url)
# print("Number of entries:", len(data))
#
# print("Example entry:\n", data[50])
# print("Another example entry:\n", data[999])


# 代码清单 7-2 实现提示词格式函数、
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\m{entry['instruction']}"
    )

    input_text = (
        f"\n\n### Input:\n{entry['input']}" if entry["input"] else  ""
    )
    return instruction_text + input_text
"""
带格式的输入就像下面这样：
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Identify the correct spelling of the following word.

### Input:
Occassion

### Response:
The correct spelling is 'Occasion.'
"""

model_input = format_input(data[999])
desired_response = f"\n\n### Response:\n{data[999]['output']}"
# print(model_input + desired_response)


# 代码清单 7-3 划分数据集
train_portion = int(len(data) * 0.85) # 使用85%的数据作为训练集
test_portion = int(len(data) * 0.1) # 使用10%的数据作为测试集
val_portion = len(data) - train_portion - test_portion # 使用剩下的5%的数据作为验证集

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]

# print("Training set length:", len(train_data))
# print("Validation set length:", len(val_data))
# print("Test set length:", len(test_data))


# 代码清单 7-4 实现一个指令数据集类
import torch
from torch.utils.data import Dataset

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry) # 预词元化文本
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[ index]

    def __len__(self):
        return len(self.data)

import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
# print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

def custom_collate_draft_1(
        batch,
        pad_token_id=50256,
        device="cpu"
):
    batch_max_length = max(len(item)+1 for item in batch) # 找到批次中最长的序列
    inputs_lst = []

    for item in batch: # 填充并准备输入
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1]) # 删除之前添加的额外填充词元
        inputs_lst.append(inputs)

    inputs_tensor = torch.stack(inputs_lst).to(device) # 输入列表变成一个张量并转移到目标设备
    return inputs_tensor

inputs_1 = [0, 1, 2, 3, 4]
inputs_2 = [5, 6]
inputs_3 = [7, 8, 9]
batch = (
    inputs_1,
    inputs_2,
    inputs_3
)
# print(custom_collate_draft_1(batch))

def custom_collate_draft_2(
        batch,
        pad_token_id=50256,
        device="cpu"
):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1]) # 截断输入的最后一个词元
        targets = torch.tensor(padded[1:]) # 向左易懂一个位置得到目标
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

inputs, targets = custom_collate_draft_2(batch)
# print(inputs)
# print(targets)


# 代码清单 7-5 实现一个自定义的批聚合函数
def custom_collate_fn(
        batch,
        pad_token_id=50256,
        ignore_index=-100,
        allowed_max_length=None,
        device="cpu"
):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = ( # 将序列填充至 max_length
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1]) # 截断输入的最后一个词元
        targets = torch.tensor(padded[1:]) # 向左移动一个位置得到目标

        # 把目标序列中除第一个填充词元外的所有填充词元都替换为 ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            # 可选地截断至最大序列长度
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append( inputs)
        targets_lst.append( targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

inputs, targets = custom_collate_fn(batch)
# print(inputs)
# print(targets)

logits_1 = torch.tensor(
    [
        [-1.0, 1.0], # 第一个词元的预测
        [-0.5, 1.5] # 第二个词元的预测
    ]
)
targets_1 = torch.tensor([0, 1]) # 要生成的正确词元索引
loss_1 = torch.nn.functional.cross_entropy(logits_1, targets_1)
# print(loss_1)

logits_2 = torch.tensor(
    [
        [-1.0, 1.0],
        [-0.5, 1.5],
        [-0.5, 1.5] # 新的第三个词元的预测
    ]
)
targets_2 = torch.tensor([0, 1, 1])
loss_2 = torch.nn.functional.cross_entropy(logits_2, targets_2)
# print(loss_2)

targets_3 = torch.tensor([0, 1, -100])
loss_3 = torch.nn.functional.cross_entropy(logits_2, targets_3)
print(loss_3)
print("loss_1 == loss_3:", loss_1 == loss_3)