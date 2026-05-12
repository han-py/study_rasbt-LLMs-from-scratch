# ============================================================================
# 第六章：使用 GPT 模型进行文本分类（垃圾邮件检测）
# ============================================================================
# 
# 📚 本章学习目标：
#   学习如何将预训练的 GPT-2 模型微调为文本分类器，用于自动识别垃圾短信
#
# 🎯 应用场景：
#   - 自动过滤垃圾邮件/短信
#   - 内容审核系统
#   - 情感分析
#   - 主题分类
#
# 📋 主要步骤（12个代码清单）：
#   1. 数据准备：下载、清洗、平衡、划分数据集
#   2. 文本分词：使用 GPT-2 tokenizer 将文本转换为数字序列
#   3. 加载预训练模型：下载并加载 GPT-2 的预训练权重
#   4. 改造模型：将输出层从文本生成改为二分类
#   5. 微调训练：只训练部分参数，适应新任务
#   6. 评估测试：计算准确率、损失等指标
#   7. 实际应用：对新文本进行分类预测
#
# 🔑 关键技术概念：
#   - 迁移学习（Transfer Learning）：利用预训练模型的知识
#   - 冻结参数（Parameter Freezing）：保护预训练权重不被破坏
#   - 微调（Fine-tuning）：针对特定任务调整模型
#   - 类别平衡（Class Balancing）：处理数据不平衡问题
#   - Dropout：防止过拟合的正则化技术
#   - 交叉熵损失（Cross-Entropy Loss）：分类任务常用的损失函数
#
# 📊 预期结果：
#   - 训练完成后，模型在测试集上的准确率应达到 90% 以上
#   - 能够准确区分垃圾邮件和正常邮件
#
# ⏱️ 预计运行时间：
#   - CPU: 约 10-20 分钟
#   - GPU: 约 2-5 分钟
#
# ============================================================================

# ============================================================================
# 第一部分：数据准备
# ============================================================================
# 目标：获取 SMS 垃圾邮件数据集，并进行预处理
# 数据集来源：UCI Machine Learning Repository
# 数据集特点：包含约 5500 条短信，分为"spam"（垃圾邮件）和"ham"（正常邮件）
# ============================================================================

# -------------------------- 代码清单 6-1：下载和解压数据集 --------------------------
# 导入必要的库
import urllib.request  # Python 标准库，用于从网络下载文件
import os              # 操作系统接口模块，提供文件和目录操作功能
import zipfile         # 用于处理 ZIP 格式的压缩文件
from pathlib import Path  # 面向对象的文件系统路径操作（比传统字符串路径更安全）

# ⚠️ 注意：以下导入在当前代码中未使用，可能是从模板复制时遗留的
# 这些可以安全删除，但保留也不影响运行
from datasets.utils import extract
from flatbuffers import encode
from jinja2 import optimizer
from pandas.core.common import random_state
from patsy import origin
from sipbuild.generator import outputs

# 📥 数据集配置
# UCI 机器学习仓库提供的 SMS Spam Collection 数据集
url = "http://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"          # 下载的 ZIP 文件名（保存在当前目录）
extracted_path = "sms_spam_collection"        # 解压后的文件夹名
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"  # 最终的数据文件完整路径

def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    """
    下载并解压垃圾邮件数据集
    
    这个函数执行三个步骤：
    1. 从 URL 下载 ZIP 文件
    2. 解压 ZIP 文件到指定目录
    3. 重命名文件，添加 .tsv 扩展名
    
    参数:
        url (str): 数据集的下载链接
        zip_path (str): ZIP 文件的保存路径
        extracted_path (str): 解压后的目录路径
        data_file_path (Path): 最终数据文件的完整路径
    
    返回:
        None（直接保存到文件系统）
    
    类比：就像在网上下载一个压缩包，解压后把里面的文件改个名字
    """
    # ✅ 检查数据文件是否已存在，避免重复下载
    # 这是一个好习惯：节省时间和带宽
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return  # 如果文件已存在，直接退出函数

    # 📥 步骤1：从 URL 下载 ZIP 文件
    # urlopen() 打开网络连接，返回一个类似文件的对象
    with urllib.request.urlopen(url) as response:
        # 'wb' 表示以二进制写入模式打开文件
        # 二进制模式用于非文本文件（如 ZIP、图片等）
        with open(zip_path, 'wb') as out_file:
            # response.read() 读取所有下载的内容
            # out_file.write() 将内容写入本地文件
            out_file.write(response.read())
    
    print(f"Downloaded: {zip_path}")

    # 📦 步骤2：解压 ZIP 文件到指定目录
    # ZipFile 是处理 ZIP 文件的类
    # 'r' 表示以读取模式打开
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # extractall() 解压所有文件到目标目录
        # 如果目录不存在，会自动创建
        zip_ref.extractall(extracted_path)
    
    print(f"Extracted to: {extracted_path}")

    # 📝 步骤3：重命名文件，添加 .tsv 扩展名
    # 原始文件名为 "SMSSpamCollection"（无扩展名）
    # TSV = Tab-Separated Values（制表符分隔值），类似 CSV 但用制表符分隔
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)  # 重命名文件
    
    print(f"File renamed and saved as: {data_file_path}")
    print(f"✅ Data preparation complete!")

# 🚀 执行下载和解压操作
# 第一次运行时会下载文件，之后会跳过
download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)


# ============================================================================
# 读取和探索数据集
# ============================================================================
import pandas as pd  # pandas 是强大的数据处理库

# 📖 读取 TSV 文件
# TSV 文件使用制表符（\t）作为列分隔符
df = pd.read_csv(
    data_file_path,
    sep="\t",           # 指定分隔符为制表符（Tab）
    header=None,        # 文件没有表头行（第一行就是数据）
    names=["Label", "Text"]  # 手动指定列名
    # Label: 标签列，值为 "spam" 或 "ham"
    # Text: 文本列，包含短信内容
)

# 🔍 探索数据集（可选，已注释）
# print(df.head())  # 打印前 5 行，查看数据结构
# print(df.shape)   # 打印数据形状：(行数, 列数)
# print(df.Label.value_counts())  # 统计各类别数量

# 📊 数据集统计信息：
# - 总样本数：约 5574 条短信
# - ham（正常邮件）：约 4827 条（86.6%）
# - spam（垃圾邮件）：约 747 条（13.4%）
# ⚠️ 问题：类别严重不平衡！正常邮件远多于垃圾邮件
# 这会导致模型倾向于预测"正常"，因为这样准确率也很高
# 解决方案：见下一个代码清单


# ============================================================================
# -------------------------- 代码清单 6-2：创建平衡数据集 --------------------------
# ============================================================================
# 问题：原始数据集中正常邮件（ham）远多于垃圾邮件（spam）
#       比例约为 6.5:1，这会导致模型偏向预测"正常"
#
# 解决方案：随机采样正常邮件，使其数量与垃圾邮件相等
#          这样两类样本各占 50%，模型不会偏向任何一方
#
# 类比：就像拔河比赛，两边人数要相等才公平
# ============================================================================

def create_balanced_dataset(df):
    """
    创建平衡数据集，使两类样本数量相等
    
    工作原理：
    1. 统计垃圾邮件的数量
    2. 从正常邮件中随机抽取相同数量的样本
    3. 合并两个子集，形成平衡数据集
    
    参数:
        df (DataFrame): 原始的 Pandas DataFrame，包含 "Label" 和 "Text" 列
    
    返回:
        balanced_df (DataFrame): 平衡后的 DataFrame，ham 和 spam 数量相等
    
    注意:
        - 使用 random_state=123 确保每次运行结果可复现
        - 这会丢弃部分正常邮件样本，但对于演示目的足够了
    """
    # 📊 步骤1：统计"垃圾邮件"的样本数量
    # df["Label"] == "spam" 返回布尔 Series（True/False）
    # df[df["Label"] == "spam"] 筛选出所有垃圾邮件
    # .shape[0] 获取行数（样本数量）
    num_spam = df[df["Label"] == "spam"].shape[0]
    print(f"Number of spam messages: {num_spam}")
    
    # 🎲 步骤2：从正常邮件中随机抽取与垃圾邮件相同数量的样本
    # .sample() 方法随机抽样
    # n=num_spam: 抽取的数量等于垃圾邮件数量
    # random_state=123: 设置随机种子，确保结果可复现
    #                   （相同的种子产生相同的随机结果）
    ham_subset = df[df["Label"] == "ham"].sample(
        n=num_spam,
        random_state=123
    )
    print(f"Sampled {num_spam} ham messages from {df[df['Label'] == 'ham'].shape[0]} total")
    
    # 🔗 步骤3：将抽样的正常邮件与所有垃圾邮件合并
    # pd.concat() 沿指定轴连接多个 DataFrame
    # 默认沿行轴（axis=0）连接，即上下拼接
    balanced_df = pd.concat([
        ham_subset,              # 采样的正常邮件
        df[df["Label"] == "spam"]  # 所有垃圾邮件
    ])
    
    print(f"Balanced dataset created: {len(balanced_df)} total samples")
    return balanced_df

# 🚀 创建平衡数据集
balanced_df = create_balanced_dataset(df)

# 🔍 验证平衡性（可选，已注释）
# print(balanced_df["Label"].value_counts())
# 现在应该看到：ham 和 spam 各约 747 条

# 🔢 将文本标签转换为数值标签
# 原因：机器学习模型只能处理数值，不能直接处理字符串
# 映射关系：
#   "ham"  → 0  （正常邮件）
#   "spam" → 1  （垃圾邮件）
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

# 🔍 验证转换结果（可选）
# print(balanced_df.head())  # 查看前几行，确认 Label 列已变为 0 和 1


# ============================================================================
# -------------------------- 代码清单 6-3：划分数据集 --------------------------
# ============================================================================
# 将数据分为三个互斥的子集：
#
# 📚 训练集（Training Set）- 70%
#   - 用途：模型从这个数据集学习
#   - 类比：学生的课本和练习题
#
# 🔍 验证集（Validation Set）- 10%
#   - 用途：训练过程中监控模型性能，调整超参数
#   - 类比：模拟考试，用来检验学习效果
#
# 🧪 测试集（Test Set）- 20%
#   - 用途：最终评估模型在未见数据上的表现
#   - 类比：期末考试，只能用一次
#
# ⚠️ 重要原则：
#   - 三个集合必须互不重叠
#   - 测试集在训练过程中完全不可见
#   - 先打乱数据再划分，避免分布偏差
# ============================================================================

def random_split(df, train_frac, validation_frac):
    """
    随机划分数据集为训练集、验证集和测试集
    
    工作流程：
    1. 打乱数据顺序（随机shuffle）
    2. 按比例计算划分点
    3. 切片获取三个子集
    
    参数:
        df (DataFrame): 输入的 DataFrame
        train_frac (float): 训练集比例（如 0.7 表示 70%）
        validation_frac (float): 验证集比例（如 0.1 表示 10%）
                                  测试集比例 = 1 - train_frac - validation_frac
    
    返回:
        tuple: (train_df, validation_df, test_df) 三个子集
    
    示例:
        >>> train, val, test = random_split(df, 0.7, 0.1)
        >>> # train: 70%, val: 10%, test: 20%
    """
    # 🎲 步骤1：打乱数据顺序
    # 为什么要打乱？
    # - 原始数据可能按某种顺序排列（如时间、类别）
    # - 不打乱会导致某个集合中某一类样本过多
    # - 打乱确保每个集合都有代表性的样本分布
    
    # df.sample(frac=1) 返回所有数据，但顺序随机
    # frac=1 表示返回 100% 的数据（只是顺序变了）
    # random_state=123 确保每次运行结果一致（可复现性）
    # reset_index(drop=True) 重置索引
    #   - drop=True 表示丢弃旧索引，创建新的从 0 开始的索引
    #   - 否则索引会保持原来的值（如 5, 23, 156...），不方便使用
    df_shuffled = df.sample(
        frac=1, 
        random_state=123
    ).reset_index(drop=True)
    
    # 📐 步骤2：计算划分点的索引位置
    # len(df) 是总样本数
    # int() 将浮点数转换为整数（索引必须是整数）
    train_end = int(len(df_shuffled) * train_frac)  # 训练集结束位置（不包含）
    validation_end = train_end + int(len(df_shuffled) * validation_frac)  # 验证集结束位置
    
    print(f"Total samples: {len(df_shuffled)}")
    print(f"Training set: {train_end} samples ({train_frac*100:.0f}%)")
    print(f"Validation set: {validation_end - train_end} samples ({validation_frac*100:.0f}%)")
    print(f"Test set: {len(df_shuffled) - validation_end} samples ({(1-train_frac-validation_frac)*100:.0f}%)")

    # ✂️ 步骤3：切片获取三个子集
    # Python 切片语法：df[start:end] 包含 start，不包含 end
    train_df = df_shuffled[:train_end]  # 从开头到 train_end（前 70%）
    validation_df = df_shuffled[train_end:validation_end]  # 中间 10%
    test_df = df_shuffled[validation_end:]  # 从 validation_end 到末尾（剩余 20%）

    return train_df, validation_df, test_df

# 💾 以下代码用于首次运行时生成 CSV 文件
# 由于文件已经存在，这部分被注释掉了
# 如果需要重新生成，取消注释即可

# train_df, validation_df, test_df = random_split(
#     balanced_df, 0.7, 0.1
# )  # 训练集70%，验证集10%，测试集自动为剩余的20%
#
# # 保存为 CSV 文件
# # index=None 表示不保存索引列（我们不需要它）
# train_df.to_csv("train.csv", index=None)
# validation_df.to_csv("validation.csv", index=None)
# test_df.to_csv("test.csv", index=None)
#
# print("✅ Dataset split and saved successfully!")


# ============================================================================
# 第二部分：文本分词（Tokenization）
# ============================================================================
# 目标：将文本转换为模型可以理解的数字序列
#
# 🤔 为什么需要分词？
#   - 计算机不理解文字，只理解数字
#   - 分词器将文本拆分为"tokens"（词元）
#   - 每个 token 对应一个唯一的 ID（整数）
#
# 📝 示例：
#   文本："Hello world"
#   ↓ 分词
#   Tokens: ["Hello", " world"]
#   ↓ 编码
#   Token IDs: [15496, 995]
#
# 🔧 使用的工具：tiktoken（OpenAI 开发的高效分词库）
# ============================================================================

import tiktoken  # OpenAI 开发的快速分词库

# 🎯 加载 GPT-2 的分词器
# GPT-2 使用的是 BPE（Byte-Pair Encoding）分词算法
# 这种算法可以处理任意文本，包括未知单词
tokenizer = tiktoken.get_encoding("gpt2")

# 🔍 测试分词效果（可选，已注释）
# text = "Hello world"
# tokens = tokenizer.encode(text)
# print(f"Text: {text}")
# print(f"Tokens: {tokens}")
# print(f"Decoded: {tokenizer.decode(tokens)}")

# 💡 提示：
# - tokenizer.encode() 将文本转换为 token ID 列表
# - tokenizer.decode() 将 token ID 列表转换回文本
# - GPT-2 的词汇表大小：50,257 个不同的 tokens


# ============================================================================
# 第三部分：构建 PyTorch Dataset 类
# ============================================================================
# 目标：创建自定义数据集类，用于高效加载和处理数据
#
# 🤔 为什么需要 Dataset 类？
#   - PyTorch 的标准接口，与 DataLoader 配合使用
#   - 自动处理批量加载、打乱、并行读取
#   - 封装数据预处理逻辑（分词、填充等）
#
# 📋 Dataset 类必须实现的方法：
#   1. __init__(): 初始化，加载数据
#   2. __getitem__(index): 获取第 index 个样本
#   3. __len__(): 返回数据集大小
#
# 🔧 本例的特殊处理：
#   - 自动分词：将文本转换为 token IDs
#   - 统一长度：截断或填充到相同长度
#   - 返回张量：PyTorch 模型需要的数据格式
# ============================================================================

import torch  # PyTorch 深度学习框架
from torch.utils.data import Dataset  # Dataset 基类

torch.manual_seed(123)  # 设置随机种子，确保结果可复现

class SpamDataset(Dataset):
    """
    短信分类数据集类
    
    这个类负责：
    1. 读取 CSV 文件中的短信数据
    2. 使用 tokenizer 将文本转换为 token ID 序列
    3. 对所有序列进行截断或填充，使其长度统一
    4. 提供 __getitem__ 方法供 DataLoader 调用
    
    类比：就像把不同长度的句子放进相同大小的盒子
         - 短的句子：后面补空格（padding）
         - 长的句子：剪掉多余部分（truncation）
    
    属性:
        data (DataFrame): 原始数据
        encoded_texts (list): 分词后的 token ID 列表
        max_length (int): 统一的序列长度
    """
    
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        """
        初始化数据集
        
        参数:
            csv_file (str): CSV 文件路径（包含 "Label" 和 "Text" 列）
            tokenizer: 分词器对象（用于将文本转换为 token IDs）
            max_length (int, optional): 最大序列长度
                - None: 自动使用数据集中最长样本的长度
                - 整数: 使用指定长度，超过则截断，不足则填充
            pad_token_id (int): 填充标记的 ID
                - GPT-2 默认使用 50256 作为 padding token
                - 这个 token 不会被模型当作有意义的输入
        
        工作流程:
            1. 读取 CSV 文件
            2. 对所有文本进行分词
            3. 确定最大长度
            4. 截断或填充所有序列到统一长度
        """
        # 📖 步骤1：读取 CSV 文件
        # pandas 会自动解析 CSV，返回 DataFrame
        self.data = pd.read_csv(csv_file)
        print(f"Loaded {len(self.data)} samples from {csv_file}")
        
        # 🔤 步骤2：对所有文本进行分词，转换为 token ID 列表
        # 列表推导式：对每一行文本应用 tokenizer.encode()
        # 例如："Hello world" → [15496, 995]
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]
        
        # 📏 步骤3：确定最大序列长度
        if max_length is None:
            # 如果未指定，则使用数据集中最长样本的长度
            # 这样可以保留所有信息，但可能导致序列很长
            self.max_length = self._longest_encoded_length()
            print(f"Auto-detected max length: {self.max_length}")
        else:
            # 使用指定的最大长度
            self.max_length = max_length
            print(f"Using specified max length: {self.max_length}")
            
            # ✂️ 截断：如果序列长度超过 max_length，只保留前面的部分
            # 例如：max_length=10，序列有 15 个 tokens → 保留前 10 个
            self.encoded_texts = [
                encoded_text[:self.max_length]  # 切片：只取前 max_length 个
                for encoded_text in self.encoded_texts
            ]

        # 🔲 步骤4：填充所有序列到相同长度
        # 对于较短的序列，在末尾添加 padding tokens（ID=50256）
        # 例如：max_length=10，序列有 6 个 tokens → 添加 4 个 padding
        # [15496, 995, ...] + [50256, 50256, 50256, 50256]
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]
        
        print(f"✅ Dataset initialized with max_length={self.max_length}")

    def __getitem__(self, index):
        """
        获取第 index 个样本
        
        这个方法由 DataLoader 自动调用，用于获取单个样本
        
        参数:
            index (int): 样本索引（0 到 len(self)-1）
        
        返回:
            tuple: (token_ids, label)
                - token_ids (Tensor): token ID 张量，形状 (max_length,)
                - label (Tensor): 标签张量，标量（0 或 1）
        
        示例:
            >>> dataset = SpamDataset(...)
            >>> tokens, label = dataset[0]  # 获取第一个样本
            >>> print(tokens.shape)  # torch.Size([max_length])
            >>> print(label)  # tensor(0) 或 tensor(1)
        """
        encoded = self.encoded_texts[index]  # 获取分词后的 token ID 列表
        label = self.data.iloc[index]["Label"]  # 获取对应的标签（0 或 1）
        
        # 🔄 转换为 PyTorch 张量
        # dtype=torch.long 是因为：
        # - 嵌入层（Embedding）需要整数类型的索引
        # - long 是 64 位整数，可以表示很大的词汇表
        return (
            torch.tensor(encoded, dtype=torch.long),   # token IDs
            torch.tensor(label, dtype=torch.long)      # label
        )

    def __len__(self):
        """
        返回数据集大小（样本数量）
        
        这个方法让 Python 的 len() 函数可以工作
        
        示例:
            >>> len(dataset)  # 返回样本总数
        """
        return len(self.data)

    def _longest_encoded_length(self):
        """
        计算所有样本中最长的 token 序列长度
        
        这个方法在 __init__ 中被调用，用于自动确定 max_length
        
        返回:
            int: 最长序列的 token 数量
        
        注意:
            - 遍历所有样本，找到最长的
            - 这可能需要一些时间，但只做一次
        """
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length

# 🚀 创建三个数据集实例

# 📚 训练集：自动确定 max_length（基于训练集的最长样本）
train_dataset = SpamDataset(
    csv_file="train.csv",
    max_length=None,  # 自动计算最大长度
    tokenizer=tokenizer
)
# print(f"Training set max length: {train_dataset.max_length}")

# 🔍 验证集和测试集：使用与训练集相同的 max_length
# 为什么？确保所有数据集的维度一致，模型可以处理
val_dataset = SpamDataset(
    csv_file="validation.csv",
    max_length=train_dataset.max_length,  # 使用训练集的最大长度
    tokenizer=tokenizer
)

test_dataset = SpamDataset(
    csv_file="test.csv",
    max_length=train_dataset.max_length,  # 使用训练集的最大长度
    tokenizer=tokenizer
)

print(f"✅ All datasets created with max_length={train_dataset.max_length}")


# ============================================================================
# 第四部分：创建 DataLoader
# ============================================================================
# 目标：创建数据加载器，用于批量加载数据
#
# 🤔 为什么需要 DataLoader？
#   - 自动批量处理：一次加载多个样本（batch）
#   - 自动打乱：每个 epoch 随机打乱训练数据
#   - 并行加载：使用多进程加速数据读取
#   - 内存效率：不需要一次性加载所有数据到内存
#
# 📦 Batch（批次）的概念：
#   - 一次处理多个样本，而不是一个一个
#   - 优点：GPU 并行计算更高效
#   - 缺点：占用更多内存
#   - 常见 batch_size: 8, 16, 32, 64, 128
#
# 🔧 DataLoader 参数说明：
#   - dataset: 数据集对象
#   - batch_size: 每个批次的样本数
#   - shuffle: 是否打乱数据（训练集需要，验证/测试集不需要）
#   - num_workers: 并行工作进程数（0 表示主进程加载）
#   - drop_last: 是否丢弃最后一个不完整的批次
# ============================================================================

from torch.utils.data import DataLoader  # 数据加载器

# ⚙️ 配置参数
num_workers = 0  # 数据加载的工作进程数
# num_workers=0: 在主进程中加载数据
#   - 优点：兼容性好，调试方便
#   - 缺点：速度较慢
# num_workers>0: 使用多个子进程并行加载
#   - 优点：速度快，特别是大数据集
#   - 缺点：Windows 上可能有问题，需要 if __name__ == '__main__'

batch_size = 8  # 每个批次包含 8 个样本
# batch_size 的选择：
# - 越大：训练越快，GPU 利用率越高，但占用内存越多
# - 越小：内存占用少，但训练慢，梯度估计不够准确
# - 经验法则：从较小的值开始（如 8, 16），根据显存调整

torch.manual_seed(123)  # 设置随机种子，确保 shuffle 结果可复现

# 📚 训练集数据加载器
train_loader = DataLoader(
    dataset=train_dataset,  # 指定数据集
    batch_size=batch_size,  # 每批 8 个样本
    shuffle=True,           # ✅ 每个 epoch 打乱数据顺序
                            #    避免模型记住样本顺序
    num_workers=num_workers,# 工作进程数
    drop_last=True          # ✅ 丢弃最后一个不完整的批次
                            #    例如：如果有 100 个样本，batch_size=8
                            #    会有 12 个完整批次（96 个样本）+ 1 个不完整批次（4 个样本）
                            #    drop_last=True 会丢弃那 4 个样本
)

# 🔍 验证集数据加载器
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=False,          # ❌ 验证集不需要打乱
    num_workers=num_workers,
    drop_last=False         # ❌ 验证时需要使用所有样本，不丢弃
)

# 🧪 测试集数据加载器
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,          # ❌ 测试集不需要打乱
    num_workers=num_workers,
    drop_last=False         # ❌ 测试时需要使用所有样本，不丢弃
)

# 🔍 测试数据加载器：遍历一个批次，检查数据形状
for input_batch, target_batch in train_loader:
    break  # 只取第一个批次就退出

# print("Input batch dimensions:", input_batch.shape)
# 期望输出: torch.Size([8, max_length])
# - 8: batch_size
# - max_length: 每个样本的 token 数量

# print("Target batch dimensions:", target_batch.shape)
# 期望输出: torch.Size([8])
# - 8: batch_size
# - 每个元素是 0 或 1（标签）

# print(f"{len(train_loader)} training batches")
# print(f"{len(val_loader)} validation batches")
# print(f"{len(test_loader)} test batches")


# ============================================================================
# 第五部分：加载预训练 GPT-2 模型
# ============================================================================
# 目标：下载并加载预训练的 GPT-2 模型
#
# 🤔 什么是预训练模型？
#   - 在大规模文本语料上训练过的模型
#   - 已经学会了语言的基本规律（语法、语义等）
#   - 我们可以在此基础上进行"微调"，适应特定任务
#
# 🎯 迁移学习的优势：
#   1. 节省时间：不需要从头训练
#   2. 节省数据：小数据集也能取得好效果
#   3. 性能更好：利用了大规模预训练的知识
#
# 📊 GPT-2 模型家族：
#   - gpt2-small (124M):   1.24 亿参数，最小最快
#   - gpt2-medium (355M):  3.55 亿参数
#   - gpt2-large (774M):   7.74 亿参数
#   - gpt2-xl (1558M):     15.58 亿参数，最大最慢
#
# 💡 本例使用 gpt2-small，因为它：
#   - 足够完成分类任务
#   - 训练速度快
#   - 内存占用少
# ============================================================================

# 🎯 选择模型版本
CHOOSE_MODEL = "gpt2-small (124M)"  # 使用最小的 GPT-2 模型
# 其他选项：
# CHOOSE_MODEL = "gpt2-medium (355M)"
# CHOOSE_MODEL = "gpt2-large (774M)"
# CHOOSE_MODEL = "gpt2-xl (1558M)"

INPUT_PROMPT = "Every effort moves"  # 测试用的输入文本（当前未使用）

# ⚙️ GPT 模型的基础配置参数
BASE_CONFIG = {
    "vocab_size": 50257,     # 词汇表大小：GPT-2 有 50,257 个不同的 tokens
    "context_length": 1024,  # 上下文长度：模型最多能处理 1024 个 tokens
    "drop_rate": 0.0,        # Dropout 率：0.0 表示不使用 dropout
                             # dropout 是一种正则化技术，防止过拟合
    "qkv_bias": True         # Query-Key-Value 偏置：是否在注意力机制中使用偏置项
}

# 📊 不同规模 GPT 模型的详细配置
model_configs = {
    "gpt2-small (124M)": {
        "emb_dim": 768,    # 嵌入维度：每个 token 的向量表示有 768 维
        "n_layers": 12,    # Transformer 层数：模型有 12 个 Transformer 块
        "n_heads": 12      # 注意力头数：多头注意力有 12 个头
    },
    # 参数量 ≈ emb_dim² × n_layers × 常数
    # 124M ≈ 768² × 12 × 常数
    
    "gpt2-medium (355M)": {
        "emb_dim": 1024,
        "n_layers": 24,
        "n_heads": 16
    },
    
    "gpt2-large (774M)": {
        "emb_dim": 1280,
        "n_layers": 36,
        "n_heads": 20
    },
    
    "gpt2-xl (1558M)": {
        "emb_dim": 1600,
        "n_layers": 48,
        "n_heads": 25
    },
}

# 🔗 将选定模型的配置合并到 BASE_CONFIG 中
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
# 现在 BASE_CONFIG 包含了完整的模型配置
# print("Complete model config:", BASE_CONFIG)


# -------------------------- 代码清单 6-6：加载预训练 GPT 模型 --------------------------
from gpt_download import download_and_load_gpt2  # 导入下载和加载函数

import torch.nn as nn  # PyTorch 神经网络模块

# 📝 以下定义了 GPT 模型的所有组件类
# 这些类的详细解释请参考 U4 和 U5 章节
# 这里我们直接使用它们来构建模型

# ============================================================================
# 🔄 加载预训练权重到 GPT 模型
# ============================================================================

def assign(left, right):
    """
    将 NumPy 数组赋值给 PyTorch 参数
    
    作用：
    - 检查形状是否匹配
    - 将 NumPy 数组转换为 PyTorch Parameter
    
    参数:
        left: PyTorch Parameter（目标）
        right: NumPy array（源）
    
    返回:
        torch.nn.Parameter: 转换后的参数
    
    注意:
        - OpenAI 的权重是 NumPy 格式
        - 我们需要转换为 PyTorch 格式
        - 有些权重需要转置（.T），因为存储方式不同
    """
    if left.shape != right.shape:
        raise ValueError(
            f"Shape mismatch. Left: {left.shape}, Right: {right.shape}"
        )
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    """
    将预训练权重加载到 GPT 模型中
    
    这个函数非常关键，它将 OpenAI 训练的权重
    逐层复制到我们的模型结构中
    
    参数:
        gpt: GPTModel 实例（我们的模型）
        params: 字典，包含所有预训练权重
    
    工作流程:
        1. 加载嵌入层权重（token + position）
        2. 遍历每个 Transformer 块，加载其权重
        3. 加载最终层归一化和输出头权重
    
    类比：就像给一个空房子装修
         - 结构已经建好（模型架构）
         - 现在填入家具和装饰（权重）
    """
    # 📍 步骤1：加载位置嵌入和词元嵌入权重
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])
    
    # 🔄 步骤2：遍历每个 Transformer 块，加载权重
    for b in range(len(params["blocks"])):
        # --- 注意力机制的权重 ---
        
        # QKV 权重：从一个大矩阵拆分为三个小矩阵
        # OpenAI 将 Q、K、V 的权重存储在一起
        # 我们需要用 np.split 将它们分开
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 
            3,  # 分成 3 份
            axis=-1  # 沿最后一个维度分割
        )
        
        # 赋值给 Query、Key、Value 的权重
        # .T 是因为存储方式是转置的
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T
        )
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T
        )
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T
        )
        
        # QKV 偏置：同样拆分
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 
            3, 
            axis=-1
        )
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b
        )
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b
        )
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b
        )
        
        # 输出投影层的权重和偏置
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T
        )
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"]
        )
        
        # --- 前馈网络的权重 ---
        
        # 第一层 Linear（扩展层）
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T
        )
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"]
        )
        
        # 第二层 Linear（压缩层）
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T
        )
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"]
        )
        
        # --- 层归一化的参数 ---
        
        # 第一个 LayerNorm（在注意力之前）
        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"]  # "g" = gamma (scale)
        )
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"]  # "b" = beta (shift)
        )
        
        # 第二个 LayerNorm（在前馈之前）
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"]
        )
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"]
        )
    
    # 🏁 步骤3：加载最终的层归一化和输出头
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    
    # ⚠️ 重要：权重绑定（Weight Tying）
    # OpenAI 的 GPT-2 在输出层复用了词元嵌入权重
    # 这样可以减少参数数量，提高训练效率
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])
    
    print("✅ Pre-trained weights loaded successfully!")


# ============================================================================
# 🚀 加载预训练模型
# ============================================================================

# 📥 步骤1：从模型名称中提取大小信息
# CHOOSE_MODEL = "gpt2-small (124M)"
# split(" ") → ["gpt2-small", "(124M)"]
# [-1] → "(124M)"
# lstrip("(") → "124M)"
# rstrip(")") → "124M"
model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
print(f"Loading model: {model_size}")

# 📥 步骤2：下载并加载 GPT-2 模型的预训练权重
# 这个函数会：
# 1. 检查本地是否已有模型文件
# 2. 如果没有，从网上下载
# 3. 加载权重到内存（字典格式）
settings, params = download_and_load_gpt2(
    model_size=model_size, 
    models_dir="gpt2"  # 保存目录
)
# settings: 包含模型配置信息（如层数、头数等）
# params: 包含所有层的权重参数（嵌套字典）
#   - 'wte': word token embeddings
#   - 'wpe': word position embeddings
#   - 'blocks': list of transformer blocks
#   - 'g': final layer norm gamma
#   - 'b': final layer norm beta

print(f"✅ Model weights downloaded and loaded!")
print(f"   Number of parameters: {sum(p.numel() for p in params.values()) / 1e6:.1f}M")

# 🏗️ 步骤3：创建 GPT 模型实例（使用随机初始化的权重）
model = GPTModel(BASE_CONFIG)
print(f"✅ Model architecture created!")

# 🔄 步骤4：将预训练权重加载到模型中
# 这会将 OpenAI 训练的权重复制到我们的模型结构中
load_weights_into_gpt(model, params)
print(f"✅ Pre-trained weights transferred to model!")

# 📝 步骤5：将模型设置为评估模式
# eval() 会禁用 dropout 和 batch normalization 的训练行为
model.eval()
print(f"✅ Model set to evaluation mode")


# ============================================================================
# 🧪 测试预训练模型的文本生成能力（可选）
# ============================================================================
# 注意：这部分与分类任务无关，仅用于验证模型加载正确
# 如果您只关心分类任务，可以跳过这部分

def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    简单的文本生成函数（贪婪解码）
    
    工作原理：
    1. 输入当前文本的 token IDs
    2. 模型预测下一个最可能的 token
    3. 将新 token 添加到文本末尾
    4. 重复上述过程 max_new_tokens 次
    
    类比：就像接龙游戏，每次根据前面的内容猜下一个词
    
    参数:
        model: GPT 模型
        idx: 当前文本的 token IDs，形状 (batch, n_tokens)
        max_new_tokens: 要生成的新 token 数量
        context_size: 模型的最大上下文长度（1024）
    
    返回:
        idx: 更新后的 token IDs，形状 (batch, n_tokens + max_new_tokens)
    
    注意:
        - 这是最简单的生成方法（贪婪解码）
        - 每次都选择概率最高的 token
        - 可能导致重复和缺乏多样性
        - 更好的方法：top-k sampling, nucleus sampling 等
    """
    for _ in range(max_new_tokens):
        # ✂️ 步骤1：截断输入至模型支持的最大长度
        # 如果文本超过 1024 个 token，只保留最后 1024 个
        # idx[:, -context_size:] 表示取最后 context_size 个 token
        idx_cond = idx[:, -context_size:]
        
        # 🔮 步骤2：前向传播：获取模型的预测结果
        with torch.no_grad():  # 不计算梯度，节省内存
            logits = model(idx_cond)
        # logits Shape: (batch, seq_len, vocab_size)
        # 每个位置都有一个词汇表大小的向量
        
        # 🎯 步骤3：只关注最后一个位置的输出
        # 因为我们想预测下一个 token
        # logits[:, -1, :] Shape: (batch, vocab_size)
        logits = logits[:, -1, :]
        
        # 📊 步骤4：将 logits 转换为概率分布
        # softmax 确保所有概率之和为 1
        # probas Shape: (batch, vocab_size)
        probas = torch.softmax(logits, dim=-1)
        
        # 🏆 步骤5：选择概率最高的 token 作为下一个词
        # argmax 返回最大值的索引
        # idx_next Shape: (batch, 1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        
        # ➕ 步骤6：将新 token 拼接到原文本后面
        # torch.cat 沿指定维度连接张量
        # idx Shape: (batch, n_tokens) → (batch, n_tokens + 1)
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx


def text_to_token_ids(text, tokenizer):
    """
    将文本转换为 token ID 张量
    
    参数:
        text: 输入文本字符串
        tokenizer: 分词器
    
    返回:
        encoded_tensor: token ID 张量，形状 (1, seq_len)
    
    示例:
        >>> text_to_token_ids("Hello world", tokenizer)
        tensor([[15496, 995]])
    """
    # 分词：文本 → token IDs
    # allowed_special={'<|endoftext|>'} 允许特殊 token
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    
    # 转换为张量并添加 batch 维度
    # .unsqueeze(0): [token_ids] → [[token_ids]]
    # 例如：[15496, 995] → [[15496, 995]]
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """
    将 token ID 张量转换回文本
    
    参数:
        token_ids: token ID 张量，形状 (batch, seq_len)
        tokenizer: 分词器
    
    返回:
        text: 解码后的文本字符串
    
    示例:
        >>> token_ids_to_text(tensor([[15496, 995]]), tokenizer)
        'Hello world'
    """
    # 移除 batch 维度
    # .squeeze(0): [[token_ids]] → [token_ids]
    flat = token_ids.squeeze(0)
    
    # 解码：token IDs → 文本
    # .tolist() 将张量转换为 Python 列表
    return tokenizer.decode(flat.tolist())


# 🧪 测试 1：让模型续写句子
print("\n🧪 Testing text generation...")
text_1 = "Every effort moves you"
print(f"Input: {text_1}")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_1, tokenizer),
    max_new_tokens=15,  # 生成 15 个新 token
    context_size=BASE_CONFIG['context_length'],  # 1024
)

generated_text = token_ids_to_text(token_ids, tokenizer)
print(f"Generated: {generated_text}")

# 🧪 测试 2：让模型判断是否为垃圾邮件（零样本测试）
text_2 = (
    "Is the following text 'spam'? Answer with 'yes' or 'no':"
    " 'You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award.'"
)
print(f"\nInput: {text_2[:50]}...")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_2, tokenizer),
    max_new_tokens=23,
    context_size=BASE_CONFIG['context_length'],
)

generated_text = token_ids_to_text(token_ids, tokenizer)
print(f"Generated: {generated_text}")

# print(model)  # 打印模型结构（已注释，输出会很长）


# ============================================================================
# 🛠️ 第六部分：将 GPT 改造为分类器
# ============================================================================
# 代码清单 6-7：添加分类层
#
# 🎯 目标：将文本生成模型改造为二分类模型
#
# 📋 策略：迁移学习（Transfer Learning）
#   1. 冻结大部分预训练权重（不更新）
#   2. 只微调最后一层和最后的 Transformer 块
#   3. 替换输出层为分类层
#
# 💡 为什么这样做？
#   - 预训练模型已经学到了丰富的语言知识
#   - 我们不想破坏这些知识
#   - 只需要调整少量参数适应新任务
#   - 节省时间和计算资源
#
# 🔧 具体步骤：
#   1. 冻结所有参数
#   2. 替换输出层
#   3. 解冻最后 一个 Transformer 块
#   4. 解冻最后的 LayerNorm
# ============================================================================

print("\n🛠️ Modifying model for classification task...")

# ❄️ 步骤1：冻结所有参数的梯度（不更新这些参数）
for param in model.parameters():
    param.requires_grad = False
# requires_grad=False 意味着：
# - 反向传播时不会计算这些参数的梯度
# - 优化器不会更新这些参数
# - 但它们仍然参与前向传播

print(f"✅ Froze all parameters")

# 🎲 设置随机种子，确保新层的初始化可复现
torch.manual_seed(123)

# 🎯 定义分类任务的类别数
num_classes = 2  # 二分类：spam (1) 或 ham (0)

# 🔄 步骤2：替换输出层
# 原来：输出词汇表大小的 logits（用于生成文本）
#       Shape: (batch, seq_len, 50257)
# 现在：输出 2 个 logits（用于分类）
#       Shape: (batch, seq_len, 2)
model.out_head = torch.nn.Linear(
    in_features=BASE_CONFIG["emb_dim"],  # 输入：768维嵌入向量
    out_features=num_classes,            # 输出：2个类别的得分
)
# 注意：新创建的 Linear 层默认 requires_grad=True
#       这意味着它会在训练中被更新

print(f"✅ Replaced output layer: {BASE_CONFIG['emb_dim']} → {num_classes}")

# 🔓 步骤3：解冻最后一个 Transformer 块的参数（允许微调）
# 为什么只解冻最后一层？
# - 底层学习通用语言特征（语法、词义），不需要改变
# - 高层学习任务特定特征，需要适应新任务
# - 这是一种平衡：既利用预训练知识，又适应新任务
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True

print(f"✅ Unfroze last Transformer block")

# 🔓 步骤4：解冻最后的 LayerNorm 层
# LayerNorm 的参数也需要微调，以适配新的任务
for param in model.final_norm.parameters():
    param.requires_grad = True

print(f"✅ Unfroze final LayerNorm")

# 📊 统计可训练参数数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n📊 Parameter statistics:")
print(f"   Total parameters: {total_params:,}")
print(f"   Trainable parameters: {trainable_params:,}")
print(f"   Frozen parameters: {total_params - trainable_params:,}")
print(f"   Training ratio: {trainable_params/total_params*100:.2f}%")

# 🧪 测试新的分类器结构
inputs = tokenizer.encode("Do you have time")  # 分词
inputs = torch.tensor(inputs).unsqueeze(0)  # 转换为张量并添加 batch 维度
# Shape: (1, seq_len)

# print("Inputs:", inputs)  # 打印输入 token IDs（已注释）
# print("Inputs dimensions:", inputs.shape)  # 形状：(1, seq_len)（已注释）

# 🔮 前向传播：获取分类结果
with torch.no_grad():  # 不计算梯度（仅测试）
    outputs = model(inputs)
# outputs Shape: (1, seq_len, 2)
# - 1: batch size
# - seq_len: 序列长度
# - 2: 两个类别的 logits

# print("Outputs:\n", outputs)  # 打印原始输出（已注释）
# print("Outputs dimensions:", outputs.shape)  # 形状：(1, seq_len, 2)（已注释）

# 🎯 对于分类任务，我们只关心序列末尾的分类结果
# print("Last output token:", outputs[:, -1, :])  # Shape: (1, 2)（已注释）

# 📝 以下代码展示了如何将 logits 转换为类别标签（已注释，供参考）
# probas = torch.softmax(outputs[:, -1, :], dim=-1)  # 转换为概率
# label = torch.argmax(probas)  # 选择概率最高的类别
# print("Class label:", label.item())

# logits = outputs[:, -1,:]  # 直接取最后一个 token 的 logits
# label = torch.argmax(logits)  # argmax 对 logits 和 probabilities 结果相同
# print("Class label:", label.item())

print(f"✅ Model modification complete!")


# ============================================================================
# 📊 第七部分：评估指标计算
# ============================================================================
# 代码清单 6-8：计算分类准确率
#
# 🎯 目标：在训练前评估模型的初始性能
#
# 💡 为什么要评估？
#   - 了解模型的起点（未训练时的表现）
#   - 作为基线，对比训练后的提升
#   - 验证模型加载和改造是否正确
#
# 📈 预期结果：
#   - 未训练的模型准确率应该在 50% 左右（随机猜测）
#   - 因为二分类任务，随机猜对的概率是 50%
# ============================================================================

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    """
    计算模型在数据集上的分类准确率
    
    工作原理：
    1. 遍历数据批次
    2. 对每个批次进行预测
    3. 统计正确预测的数量
    4. 计算准确率 = 正确数 / 总数
    
    参数:
        data_loader: 数据加载器（提供批次数据）
        model: 分类模型
        device: 计算设备（CPU 或 GPU）
        num_batches: 评估的批次数量
                   - None: 使用所有批次
                   - 整数: 只评估指定数量的批次（加速）
    
    返回:
        accuracy: 准确率（0-1 之间的浮点数）
                 - 0.0 表示全部错误
                 - 1.0 表示全部正确
                 - 0.5 表示一半正确（随机猜测水平）
    
    示例:
        >>> accuracy = calc_accuracy_loader(test_loader, model, device)
        >>> print(f"Accuracy: {accuracy*100:.2f}%")
        Accuracy: 92.35%
    """
    # 📝 步骤1：设置模型为评估模式
    # eval() 会禁用 dropout 等训练特定行为
    model.eval()
    
    # 🔢 步骤2：初始化计数器
    correct_predictions = 0  # 正确预测的数量
    num_examples = 0         # 总样本数量
    
    # 📏 步骤3：确定要评估的批次数量
    if num_batches is None:
        num_batches = len(data_loader)  # 使用所有批次
    else:
        # 限制批次数量（用于快速评估）
        # min() 确保不超过实际批次数量
        num_batches = min(num_batches, len(data_loader))
    
    # 🔄 步骤4：遍历数据批次
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            # 📱 将数据移动到指定设备（CPU/GPU）
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            
            # 🔮 前向传播：获取模型预测
            with torch.no_grad():  # 评估时不需要计算梯度
                logits = model(input_batch)[:, -1, :]
            # logits Shape: (batch_size, 2)
            # - 取最后一个 token 的输出（[:, -1, :]）
            # - 因为分类任务只看序列末尾
            
            # 🏆 将 logits 转换为预测标签
            # argmax 返回最大值的索引（即预测的类别）
            # predicted_labels Shape: (batch_size,)
            predicted_labels = torch.argmax(logits, dim=-1)
            
            # 📊 统计正确预测的数量
            num_examples += predicted_labels.shape[0]  # 累加样本数
            
            # ✅ 比较预测与真实标签
            # (predicted_labels == target_batch) 返回布尔张量
            # .sum() 计算 True 的数量（正确预测数）
            # .item() 将单元素张量转换为 Python 数值
            correct_predictions += (
                (predicted_labels == target_batch).sum().item()
            )
        else:
            break  # 达到指定批次数量，提前退出
    
    # 🎯 步骤5：计算并返回准确率
    accuracy = correct_predictions / num_examples
    return accuracy


# 🖥️ 设置计算设备：优先使用 GPU，如果没有则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🖥️ Using device: {device}")

# 📱 将模型移动到指定设备
model.to(device)

# 🧪 在训练前评估模型的初始性能（使用 10 个批次）
print("\n📊 Evaluating initial model performance...")
torch.manual_seed(123)  # 设置随机种子

train_accuracy = calc_accuracy_loader(
    train_loader, model, device, num_batches=10
)
val_accuracy = calc_accuracy_loader(
    val_loader, model, device, num_batches=10
)
test_accuracy = calc_accuracy_loader(
    test_loader, model, device, num_batches=10
)

print(f"Initial Training accuracy: {train_accuracy * 100:.2f}%")
print(f"Initial Validation accuracy: {val_accuracy * 100:.2f}%")
print(f"Initial Test accuracy: {test_accuracy * 100:.2f}%")
print(f"💡 Expected: ~50% (random guessing level)")


# ============================================================================
# 📉 计算分类损失
# ============================================================================
# 代码清单 6-9：计算分类损失
#
# 🎯 目标：定义损失函数，用于衡量模型预测与真实标签的差距
#
# 📐 使用的损失函数：交叉熵损失（Cross-Entropy Loss）
#   - 分类任务的标准损失函数
#   - 衡量预测概率分布与真实分布的差异
#   - 值越小，预测越准确
#
# 💡 为什么用交叉熵？
#   - 对错误预测给予更大的惩罚
#   - 梯度平滑，利于优化
#   - 与 softmax 配合良好
# ============================================================================

def calc_loss_batch(input_batch, target_batch, model, device):
    """
    计算一个批次的交叉熵损失
    
    工作流程：
    1. 将数据移动到设备
    2. 前向传播获取 logits
    3. 计算交叉熵损失
    
    参数:
        input_batch: 输入 token IDs
                    Shape: (batch_size, seq_len)
        target_batch: 真实标签
                     Shape: (batch_size,)
                     值：0 或 1
        model: 分类模型
        device: 计算设备
    
    返回:
        loss: 标量损失值（torch.Tensor）
    
    示例:
        >>> loss = calc_loss_batch(input_batch, target_batch, model, device)
        >>> print(f"Loss: {loss.item():.4f}")
        Loss: 0.6931
    """
    # 📱 将数据移动到指定设备
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    
    # 🔮 前向传播：获取 logits
    logits = model(input_batch)[:, -1, :]
    # Shape: (batch_size, 2)
    # 取最后一个 token 的输出
    
    # 📐 计算交叉熵损失
    # cross_entropy 内部会自动应用 softmax
    # 所以输入是 logits 而不是 probabilities
    # 公式：Loss = -Σ y_true * log(y_pred)
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    
    return loss


def cala_loss_loader(data_loader, model, device, num_batches=None):
    """
    计算数据加载器上的平均损失
    
    工作流程：
    1. 遍历所有批次
    2. 计算每个批次的损失
    3. 累加损失
    4. 计算平均值
    
    参数:
        data_loader: 数据加载器
        model: 分类模型
        device: 计算设备
        num_batches: 计算的批次数量（None 表示全部）
    
    返回:
        average_loss: 平均损失值（float）
    
    注意:
        - 损失值越小，模型性能越好
        - 完美模型的损失接近 0
        - 随机猜测的损失约为 -ln(0.5) ≈ 0.693
    """
    total_loss = 0.  # 累计损失
    
    # ⚠️ 处理空数据加载器的情况
    if len(data_loader) == 0:
        return float("nan")  # 返回 NaN（Not a Number）
    elif num_batches is None:
        num_batches = len(data_loader)  # 使用所有批次
    else:
        # 限制批次数量
        num_batches = min(num_batches, len(data_loader))
    
    # 🔄 遍历批次并累加损失
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            # 计算当前批次的损失
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            # .item() 将张量转换为 Python 数值
            total_loss += loss.item()
        else:
            break
    
    # 🎯 返回平均损失
    average_loss = total_loss / num_batches
    return average_loss


# 🧪 在训练前评估模型的初始损失（使用 5 个批次）
print("\n📉 Calculating initial loss...")
with torch.no_grad():  # 禁用梯度以提高效率
    train_loss = cala_loss_loader(
        train_loader, model, device, num_batches=5
    )
    val_loss = cala_loss_loader(
        val_loader, model, device, num_batches=5
    )
    test_loss = cala_loss_loader(
        test_loader, model, device, num_batches=5
    )

print(f"Initial Training loss: {train_loss:.3f}")
print(f"Initial Validation loss: {val_loss:.3f}")
print(f"Initial Test loss: {test_loss:.3f}")
print(f"💡 Expected: ~0.693 (random guessing level)")


# ============================================================================
# 🏋️ 第八部分：微调模型进行垃圾信息分类
# ============================================================================
# 代码清单 6-10：训练分类器
#
# 🎯 目标：通过微调让模型学会区分垃圾邮件和正常邮件
#
# 📋 训练流程：
#   1. 前向传播：计算预测和损失
#   2. 反向传播：计算梯度
#   3. 参数更新：优化器调整权重
#   4. 定期评估：监控训练进度
#
# 🔑 关键概念：
#   - Epoch（轮次）：遍历整个训练集一次
#   - Batch（批次）：一次处理的样本子集
#   - Step（步数）：处理一个批次
#   - Learning Rate（学习率）：参数更新的步长
# ============================================================================

def train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs, eval_freq, eval_iter):
    """
    简单的分类器训练函数
    
    工作流程：
    对于每个 epoch：
      对于每个批次：
        1. 清零梯度
        2. 前向传播（计算损失）
        3. 反向传播（计算梯度）
        4. 更新参数
        5. 定期评估
      计算 epoch 结束时的准确率
    
    参数:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 优化器（如 Adam）
        device: 计算设备（CPU/GPU）
        num_epochs: 训练轮数（遍历整个数据集的次数）
        eval_freq: 评估频率（每多少个 step 评估一次）
        eval_iter: 评估时使用的批次数量
    
    返回:
        train_losses: 训练损失列表（记录每次评估的损失）
        val_losses: 验证损失列表
        train_accs: 训练准确率列表（每个 epoch 结束时的准确率）
        val_accs: 验证准确率列表
        examples_seen: 看到的样本总数
    
    类比：就像学生学习备考
         - epoch: 复习一遍所有课本
         - batch: 每次复习一章
         - eval_freq: 每复习几章做一次小测
         - 最终考试：epoch 结束时的大测
    """
    # 📊 初始化跟踪列表，用于记录训练过程中的指标
    train_losses = []  # 训练损失
    val_losses = []    # 验证损失
    train_accs = []    # 训练准确率
    val_accs = []      # 验证准确率
    
    # 🔢 初始化计数器
    examples_seen = 0  # 已看到的样本总数
    global_step = -1   # 全局步数（从 -1 开始，因为后面会先 +1）
    
    print(f"\n🏋️ Starting training for {num_epochs} epochs...")
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    print(f"   Eval frequency: every {eval_freq} steps")
    print("-" * 60)

    # 🔄 主训练循环：遍历所有 epoch
    for epoch in range(num_epochs):
        # 📝 设置模型为训练模式
        # train() 会启用 dropout 等训练特定行为
        model.train()
        
        print(f"\n📚 Epoch {epoch+1}/{num_epochs}")
        
        # 📦 遍历训练数据的所有批次
        for input_batch, target_batch in train_loader:
            # 🧹 步骤1：清零梯度
            # 为什么需要清零？
            # - PyTorch 默认会累积梯度
            # - 如果不清零，梯度会累加上一次的
            # - 这会导致错误的更新方向
            optimizer.zero_grad()
            
            # 🔮 步骤2：计算当前批次的损失
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            # loss 是一个标量张量，例如：tensor(0.6931)
            
            # ⬅️ 步骤3：反向传播：计算梯度
            # backward() 会自动计算所有 requires_grad=True 的参数的梯度
            # 梯度存储在 param.grad 中
            loss.backward()
            
            # ✏️ 步骤4：更新模型参数
            # optimizer.step() 会根据梯度和学习率更新参数
            # 公式：param = param - learning_rate * gradient
            optimizer.step()
            
            # 📊 步骤5：更新计数器
            examples_seen += input_batch.shape[0]  # 累加已见样本数
            global_step += 1  # 全局步数递增

            # 📈 步骤6：定期评估模型性能
            # 每隔 eval_freq 个 step 评估一次
            if global_step % eval_freq == 0:
                # 计算当前的训练损失和验证损失
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                
                # 记录损失
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
                # 打印进度
                print(f"  Step {global_step:06d}: "
                      f"Train Loss {train_loss:.3f}, "
                      f"Val Loss {val_loss:.3f}"
                )

        # 🏁 每个 epoch 结束后计算准确率
        train_accuracy = calc_accuracy_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_accuracy = calc_accuracy_loader(
            val_loader, model, device, num_batches=eval_iter
        )

        # 打印 epoch 结果
        print(f"  ✅ Epoch {epoch+1} Complete:")
        print(f"     Training accuracy: {train_accuracy * 100:.2f}%")
        print(f"     Validation accuracy: {val_accuracy * 100:.2f}%")
        
        # 记录准确率
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    print("-" * 60)
    print(f"✅ Training completed!")
    print(f"   Total examples seen: {examples_seen:,}")
    print(f"   Final training accuracy: {train_accs[-1] * 100:.2f}%")
    print(f"   Final validation accuracy: {val_accs[-1] * 100:.2f}%")

    return train_losses, val_losses, train_accs, val_accs, examples_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    评估模型在训练集和验证集上的损失
    
    参数:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 计算设备
        eval_iter: 评估使用的批次数量
    
    返回:
        tuple: (train_loss, val_loss) 训练损失和验证损失
    
    注意:
        - 评估时会临时将模型设为 eval() 模式
        - 评估完成后恢复为 train() 模式
    """
    # 📝 设置为评估模式（禁用 dropout）
    model.eval()
    
    # 🔮 计算损失（不计算梯度）
    with torch.no_grad():
        train_loss = cala_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = cala_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    
    # 📝 恢复训练模式
    model.train()
    
    return train_loss, val_loss


# ============================================================================
# 🚀 第九部分：执行训练
# ============================================================================

import time  # 用于计时

# ⏱️ 记录开始时间
start_time = time.time()

# 🎲 设置随机种子，确保结果可复现
torch.manual_seed(123)

# 🛠️ 创建优化器
# Adam: Adaptive Moment Estimation
# - 自适应学习率优化器
# - 适合大多数深度学习任务
# - 结合了 Momentum 和 RMSProp 的优点
optimizer = torch.optim.Adam(
    model.parameters(),  # 要优化的参数（只有 requires_grad=True 的会被更新）
    lr=5e-5,             # 学习率：0.00005
                         # - 较小的值避免破坏预训练权重
                         # - 太大会导致震荡，太小会训练慢
    weight_decay=0.1     # L2 正则化系数
                         # - 防止过拟合
                         # - 惩罚大的权重值
)

print(f"\n⚙️ Optimizer configuration:")
print(f"   Type: Adam")
print(f"   Learning rate: {5e-5}")
print(f"   Weight decay: {0.1}")

# 📅 设置训练轮数
num_epochs = 5  # 训练 5 个 epoch
# 每个 epoch = 遍历整个训练集一次
# 5 个 epoch 通常足够微调任务

print(f"   Epochs: {num_epochs}")

# 🏋️ 执行训练
train_losses, val_losses, train_accs, val_accs, examples_seen = \
    train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, 
        eval_freq=50,   # 每 50 个 step 评估一次
        eval_iter=5     # 评估时使用 5 个批次
    )

# ⏱️ 记录结束时间并计算训练时长
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60  # 转换为分钟

print(f"\n⏱️ Training completed in {execution_time_minutes:.2f} minutes.")
print(f"   That's {execution_time_minutes*60:.0f} seconds.")


# ============================================================================
# 📈 第十部分：可视化训练过程
# ============================================================================
# 代码清单 6-11：绘制分类损失和准确率曲线
#
# 🎯 目标：通过可视化了解训练过程中的变化趋势
#
# 📊 可视化的作用：
#   - 检测过拟合（训练 loss ↓，验证 loss ↑）
#   - 检测欠拟合（训练 loss 不下降）
#   - 判断是否需要更多训练
#   - 比较不同超参数的效果
#
# 📉 理想的曲线：
#   - 训练损失：持续下降
#   - 验证损失：下降后趋于平稳
#   - 两者差距不大（不过拟合）
# ============================================================================

import matplotlib.pyplot as plt  # Python 的绘图库

def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    """
    绘制训练和验证指标随时间的变化曲线
    
    参数:
        epochs_seen: epoch 数量数组（x 轴）
        examples_seen: 看到的样本数量数组（第二 x 轴）
        train_values: 训练指标值列表（y 轴）
        val_values: 验证指标值列表（y 轴）
        label: 指标名称（"loss" 或 "accuracy"）
    
    返回:
        None（直接显示图形）
    
    图形特点：
        - 双 x 轴：上轴显示样本数，下轴显示 epoch
        - 两条线：实线=训练，虚线=验证
        - 自动保存为 PDF 文件
    """
    # 🎨 步骤1：创建图形和坐标轴
    # figsize=(5, 3) 设置图形大小（宽 5 英寸，高 3 英寸）
    fig, ax1 = plt.subplots(figsize=(5, 3))
    
    # 📉 步骤2：绘制训练集和验证集指标与 epoch 的关系
    # ax1.plot(x, y, ...) 绘制折线图
    ax1.plot(
        epochs_seen,           # x 轴：epoch 数量
        train_values,          # y 轴：训练指标值
        label=f"Training {label}"  # 图例标签
    )
    ax1.plot(
        epochs_seen,           # x 轴：epoch 数量
        val_values,            # y 轴：验证指标值
        linestyle="-.",        # 线型：点划线（区分训练和验证）
        label=f"Validation {label}"  # 图例标签
    )
    
    # 🏷️ 步骤3：设置轴标签和图例
    ax1.set_xlabel("Epochs")  # x 轴标签
    ax1.set_ylabel(label.capitalize())  # y 轴标签（首字母大写）
    ax1.legend()  # 显示图例（自动选择最佳位置）
    
    # 🔄 步骤4：创建第二个 x 轴，显示看到的样本数量
    # twiny() 创建一个共享 y 轴但独立 x 轴的坐标系
    ax2 = ax1.twiny()
    
    # 绘制一条不可见的线，仅用于对齐刻度
    # alpha=0 表示完全透明（看不见）
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel("Examples seen")  # 第二个 x 轴标签
    
    # 📐 步骤5：自动调整布局，避免标签重叠
    fig.tight_layout()
    
    # 💾 步骤6：保存图形为 PDF 文件
    plt.savefig(f"{label}-plot.pdf", dpi=300, bbox_inches='tight')
    print(f"💾 Saved: {label}-plot.pdf")
    
    # 👁️ 步骤7：显示图形
    plt.show()


# 📉 准备损失曲线的绘图数据
# torch.linspace(start, end, steps) 生成均匀分布的数字
# 例如：linspace(0, 5, 10) → [0, 0.56, 1.11, ..., 5]
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))

print("\n📈 Plotting training curves...")

# 绘制损失曲线
plot_values(
    epochs_tensor, 
    examples_seen_tensor, 
    train_losses,  # 训练损失列表
    val_losses,    # 验证损失列表
    label="loss"
)

# 📈 准备准确率曲线的绘图数据
epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))

# 绘制准确率曲线
plot_values(
    epochs_tensor, 
    examples_seen_tensor, 
    train_accs,  # 训练准确率列表
    val_accs,    # 验证准确率列表
    label="accuracy"
)

print("✅ Plots generated successfully!")


# ============================================================================
# 🧪 第十一部分：最终评估
# ============================================================================

print("\n🧪 Final evaluation on full datasets...")

# 📊 在完整数据集上评估最终模型性能（不使用 num_batches 限制）
train_accuracy = calc_accuracy_loader(train_loader, model, device)
val_accuracy = calc_accuracy_loader(val_loader, model, device)
test_accuracy = calc_accuracy_loader(test_loader, model, device)

print(f"\n🏆 Final Results:")
print(f"   Training accuracy: {train_accuracy * 100:.2f}%")
print(f"   Validation accuracy: {val_accuracy * 100:.2f}%")
print(f"   Test accuracy: {test_accuracy * 100:.2f}%")

# 💡 结果解读：
# - 训练准确率 > 验证准确率：可能存在轻微过拟合
# - 三者接近：模型泛化能力好
# - 测试准确率应该在 90% 以上（对于这个任务）
print(f"\n💡 Interpretation:")
if test_accuracy > 0.9:
    print(f"   ✅ Excellent! Test accuracy > 90%")
elif test_accuracy > 0.8:
    print(f"   👍 Good! Test accuracy > 80%")
else:
    print(f"   ⚠️ Consider training for more epochs or tuning hyperparameters")


# ============================================================================
# 📱 第十二部分：实际应用 - 使用模型对新文本进行分类
# ============================================================================
# 代码清单 6-12：分类新文本
#
# 🎯 目标：将训练好的模型应用于实际场景
#
# 💡 应用场景：
#   - 自动过滤垃圾邮件
#   - 实时短信分类
#   - 集成到应用程序中
# ============================================================================

def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    """
    对单条文本进行分类（垃圾邮件检测）
    
    工作流程：
    1. 分词：文本 → token IDs
    2. 截断：如果太长，截取前面部分
    3. 填充：如果太短，补充 padding
    4. 前向传播：获取预测结果
    5. 返回分类标签
    
    参数:
        text (str): 输入文本字符串
        model: 训练好的分类模型
        tokenizer: 分词器
        device: 计算设备
        max_length (int): 最大序列长度
        pad_token_id (int): 填充 token 的 ID（默认 50256）
    
    返回:
        str: "spam" 或 "not spam (ham)"
    
    示例:
        >>> classify_review("You won $1000!", model, tokenizer, device, 120)
        'spam'
        >>> classify_review("Hey, how are you?", model, tokenizer, device, 120)
        'not spam (ham)'
    """
    # 📝 步骤1：设置模型为评估模式
    model.eval()

    # 🔤 步骤2：分词
    # 将文本转换为 token IDs
    input_ids = tokenizer.encode(text)
    # 例如："Hello world" → [15496, 995]
    
    # 📏 获取模型支持的最大上下文长度
    # pos_emb.weight.shape[0] = 1024（GPT-2 的最大序列长度）
    supported_context_length = model.pos_emb.weight.shape[0]

    # ✂️ 步骤3：截断过长的序列
    # 取 max_length 和模型支持长度的较小值
    # 确保不超过模型的处理能力
    input_ids = input_ids[:min(max_length, supported_context_length)]
    # 例如：如果有 200 个 tokens，max_length=120
    #      → 只保留前 120 个

    # 🔲 步骤4：填充到指定长度
    # 在末尾添加 padding tokens，使所有输入长度一致
    # 例如：有 50 个 tokens，max_length=120
    #      → 添加 70 个 padding tokens
    input_ids += [pad_token_id] * (max_length - len(input_ids))

    # 🔄 步骤5：转换为张量并移动到设备
    input_tensor = torch.tensor(
        input_ids, device=device
    ).unsqueeze(0)  # 添加 batch 维度：[seq_len] → [1, seq_len]
    # Shape: (1, max_length)

    # 🔮 步骤6：前向传播获取预测结果
    with torch.no_grad():  # 推理时不需要计算梯度
        logits = model(input_tensor)[:, -1, :]
    # logits Shape: (1, 2)
    # 取最后一个 token 的输出

    # 🏆 步骤7：获取预测类别
    # argmax 返回最大值的索引（0 或 1）
    # .item() 将单元素张量转换为 Python 整数
    predicted_label = torch.argmax(logits, dim=-1).item()

    # 📝 步骤8：返回分类结果
    if predicted_label == 1:
        return "spam"  # 垃圾邮件
    else:
        return "not spam (ham)"  # 正常邮件


# 🧪 测试 1：典型的垃圾邮件
print("\n📱 Testing classification on new texts...")
text_1 = (
    "You have a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award."
)
print(f"\nText 1: {text_1}")
result_1 = classify_review(
    text_1, model, tokenizer, device, 
    max_length=train_dataset.max_length
)
print(f"Prediction: {result_1}")
print(f"Expected: spam ✅")

# 🧪 测试 2：正常的短信
text_2 = (
    "Hey, just wanted to check if we're still on"
    " for dinner tonight? Let me know!"
)
print(f"\nText 2: {text_2}")
result_2 = classify_review(
    text_2, model, tokenizer, device,
    max_length=train_dataset.max_length
)
print(f"Prediction: {result_2}")
print(f"Expected: not spam (ham) ✅")


# ============================================================================
# 💾 第十三部分：保存和加载模型
# ============================================================================

# 💾 保存模型权重
print("\n💾 Saving model...")
torch.save(model.state_dict(), "review_classifier.pth")
print(f"✅ Model saved to 'review_classifier.pth'")
print(f"   File size: {os.path.getsize('review_classifier.pth') / 1e6:.2f} MB")

# 📥 加载模型（示例代码，已注释）
# 使用时取消注释以下两行
# print("\n📥 Loading model...")
# model_state_dict = torch.load("review_classifier.pth", map_location=device)
# model.load_state_dict(model_state_dict)
# print("✅ Model loaded successfully!")

# ⚠️ 注意：
# 1. 加载前需要先创建相同结构的模型实例
# 2. map_location 指定加载到哪介设备（CPU/GPU）
# 3. state_dict() 只保存权重，不保存模型结构

print("\n" + "="*60)
print("🎉 CHAPTER 6 COMPLETE!")
print("="*60)
print("\n📚 Summary:")
print("   ✅ Data preparation and balancing")
print("   ✅ Dataset and DataLoader creation")
print("   ✅ GPT-2 model loading and modification")
print("   ✅ Model training and evaluation")
print("   ✅ Visualization of training progress")
print("   ✅ Real-world application")
print("   ✅ Model saving and loading")
print("\n🎯 Key Takeaways:")
print("   • Transfer learning saves time and data")
print("   • Fine-tuning only a few layers is effective")
print("   • Balanced datasets improve fairness")
print("   • Monitoring both loss and accuracy is important")
print("\n🚀 Next Steps:")
print("   • Try different model sizes (medium, large)")
print("   • Experiment with learning rates")
print("   • Add more data augmentation")
print("   • Deploy the model as a web service")
print("="*60)

# ============================================================================
# 📦 模型组件类定义（来自 U4/U5 章节）
# ============================================================================
# 以下类定义了 GPT-2 模型的所有组件
# 这些已经在之前的章节中详细讲解过，这里直接使用

import numpy as np  # NumPy 用于数值计算和数组操作

# -------------------------- 多头注意力机制 --------------------------
class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制（Multi-Head Self-Attention）
    
    工作原理：
    1. 将输入投影到 Query、Key、Value 三个空间
    2. 分成多个"头"，每个头独立计算注意力
    3. 合并所有头的输出
    
    类比：就像有多个专家同时看一篇文章
         - 专家1关注语法结构
         - 专家2关注情感色彩
         - 专家3关注主题内容
         最后综合所有专家的意见
    """
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # 每个头的维度
        
        # 创建 Query、Key、Value 的线性变换层
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # 输出投影层
        self.dropout = nn.Dropout(dropout)
        
        # 注册因果掩码（causal mask）
        # 确保每个位置只能看到它之前的位置（不能看到未来）
        self.register_buffer(
            "mask", 
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        
        # 步骤1：计算 Query、Key、Value
        keys = self.W_key(x)    # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        # 步骤2：拆分为多个头
        # (b, num_tokens, d_out) → (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        
        # 步骤3：转置，使 num_heads 成为第二维
        # (b, num_tokens, num_heads, head_dim) → (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # 步骤4：计算注意力分数（scaled dot-product attention）
        # queries @ keys.transpose(2, 3): 矩阵乘法
        # Shape: (b, num_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(2, 3)
        
        # 步骤5：应用因果掩码（遮住未来的位置）
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)  # 遮住的设为负无穷
        
        # 步骤6：softmax 归一化 + dropout
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5,  # 缩放因子
            dim=-1
        )
        attn_weights = self.dropout(attn_weights)
        
        # 步骤7：加权求和得到上下文向量
        # Shape: (b, num_heads, num_tokens, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)
        
        # 步骤8：合并所有头
        # (b, num_tokens, num_heads, head_dim) → (b, num_tokens, d_out)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # 最终投影
        
        return context_vec


# -------------------------- 层归一化 --------------------------
class LayerNorm(nn.Module):
    """
    层归一化（Layer Normalization）
    
    作用：
    - 稳定训练过程
    - 加速收敛
    - 减少内部协变量偏移
    
    公式：
        norm_x = (x - mean) / sqrt(var + eps)
        output = scale * norm_x + shift
    
    类比：就像标准化考试成绩
         - 减去平均分（中心化）
         - 除以标准差（缩放）
         - 再调整到合适的范围
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # 防止除以零的小常数
        self.scale = nn.Parameter(torch.ones(emb_dim))   # 可学习的缩放参数
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 可学习的偏移参数
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)  # 计算均值
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # 计算方差
        norm_x = (x - mean) / torch.sqrt(var + self.eps)  # 归一化
        return self.scale * norm_x + self.shift  # 缩放和偏移


# -------------------------- GELU 激活函数 --------------------------
class GELU(nn.Module):
    """
    GELU（Gaussian Error Linear Unit）激活函数
    
    特点：
    - 比 ReLU 更平滑
    - 在负值区域也有小的梯度
    - GPT-2 使用的激活函数
    
    公式：
        GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    
    类比：像一个柔和的开关
         - 正值时几乎线性通过
         - 负值时逐渐关闭，但不完全为零
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))


# -------------------------- 前馈神经网络 --------------------------
class FeedForward(nn.Module):
    """
    前馈神经网络（Feed-Forward Network）
    
    结构：
        Linear → GELU → Linear
    
    维度变化：
        emb_dim → emb_dim * 4 → emb_dim
    
    作用：
    - 对每个位置独立处理
    - 增加模型的非线性表达能力
    
    类比：就像一个小型的多层感知机
         对每个词元进行特征提取和转换
    """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], cfg["emb_dim"] * 4),  # 扩展 4 倍
            GELU(),                                          # 激活函数
            nn.Linear(cfg["emb_dim"] * 4, cfg["emb_dim"]),  # 压缩回原维度
        )
    
    def forward(self, x):
        return self.layers(x)


# -------------------------- Transformer 块 --------------------------
class TransformerBlock(nn.Module):
    """
    Transformer 块（Transformer Block）
    
    结构：
        Input → [LayerNorm → Attention → Dropout + Skip] → 
                [LayerNorm → FeedForward → Dropout + Skip] → Output
    
    关键特性：
    1. 残差连接（Skip Connection）：shortcut + x
    2. 层归一化（LayerNorm）：在子层之前
    3. Dropout：防止过拟合
    
    类比：就像一个双层处理流水线
         第一层：注意力机制（理解上下文）
         第二层：前馈网络（提取特征）
         每层都有"快捷通道"保留原始信息
    """
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
    
    def forward(self, x):
        # 第一部分：自注意力 + 残差连接
        shortcut = x  # 保存原始输入
        x = self.norm1(x)      # 层归一化
        x = self.att(x)        # 自注意力
        x = self.drop_shortcut(x)  # Dropout
        x = shortcut + x       # 残差连接：加回原始输入
        
        # 第二部分：前馈网络 + 残差连接
        shortcut = x  # 保存当前输出
        x = self.norm2(x)      # 层归一化
        x = self.ff(x)         # 前馈网络
        x = self.drop_shortcut(x)  # Dropout
        x = shortcut + x       # 残差连接
        
        return x


# -------------------------- GPT 模型 --------------------------
class GPTModel(nn.Module):
    """
    GPT 模型（Generative Pre-trained Transformer）
    
    完整架构：
        Token Embedding + Position Embedding → 
        [Transformer Block × n_layers] → 
        LayerNorm → Output Head
    
    工作流程：
    1. 输入 token IDs
    2. 转换为嵌入向量 + 位置编码
    3. 经过多层 Transformer 处理
    4. 输出 logits（词汇表大小的向量）
    
    类比：就像一个深度理解文本的大脑
         - 嵌入层：将单词转换为向量表示
         - Transformer 层：理解上下文和语义
         - 输出层：预测下一个词或分类
    """
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])  # 词元嵌入
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])  # 位置嵌入
        self.drop_emb = nn.Dropout(cfg["drop_rate"])  # 嵌入 dropout
        
        # 堆叠多个 Transformer 块
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        
        self.final_norm = LayerNorm(cfg["emb_dim"])  # 最终层归一化
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False  # 输出头
        )
    
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        
        # 步骤1：获取词元嵌入
        tok_embeds = self.tok_emb(in_idx)  # Shape: (batch, seq_len, emb_dim)
        
        # 步骤2：获取位置嵌入
        # torch.arange(seq_len) 生成 [0, 1, 2, ..., seq_len-1]
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )  # Shape: (seq_len, emb_dim)
        
        # 步骤3：相加并应用 dropout
        x = tok_embeds + pos_embeds  # 广播加法
        x = self.drop_emb(x)
        
        # 步骤4：通过所有 Transformer 块
        x = self.trf_blocks(x)
        
        # 步骤5：最终层归一化
        x = self.final_norm(x)
        
        # 步骤6：输出 logits
        logits = self.out_head(x)  # Shape: (batch, seq_len, vocab_size)
        
        return logits

# ... [模型类定义部分保持不变] ...
