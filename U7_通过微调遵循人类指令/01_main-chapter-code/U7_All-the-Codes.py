# 代码清单 7-1 下载数据集
import json
import os
import urllib

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