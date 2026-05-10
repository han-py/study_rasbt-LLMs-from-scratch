# 代码清单 6-1 下载和解压数据集
import urllib.request
import os
import zipfile
from pathlib import Path

from datasets.utils import extract
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