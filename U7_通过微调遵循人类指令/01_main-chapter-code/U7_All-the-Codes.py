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
print("Number of entries:", len(data))

print("Example entry:\n", data[50])