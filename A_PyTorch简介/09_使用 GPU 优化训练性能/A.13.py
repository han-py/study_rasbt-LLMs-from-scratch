import torch
from torch.utils.data import DataLoader

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs): # 将输入和输出的数量编码为变量，使我们可以在具有不同特征数量和类别数量的数据集上重复相同的代码
        super().__init__()

        self.layers = torch.nn.Sequential(
            # 第一个隐藏层
            torch.nn.Linear(num_inputs, 30), # 线性层将输入结点和输出结点的数量作为参数
            torch.nn.ReLU(), # 非线性激活函数被放置在隐藏层之间

            # 第二个隐藏层
            torch.nn.Linear(30, 20), # 下一个隐藏层的输出节点数量必须与下一层的输入节点数量相匹配
            torch.nn.ReLU(),

            # 输出层
            torch.nn.Linear(20, num_outputs)
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits # 最后一层的输出称为 logits

from torch.utils.data import Dataset

import torch

x_train = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5],
])
y_train = torch.tensor([0, 0, 0, 1, 1])

x_test = torch.tensor([
    [-0.8, 2.8],
    [2.6, -1.6],
])
y_test = torch.tensor([0, 1])

class ToyDataset(Dataset):
    def __init__(self, x, y):
        self.features = x
        self.labels = y

    # 检索一条数据记录及其对应标签的说明
    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self):
        return self.labels.shape[0] # 返回数据集总长度的说明

train_ids = ToyDataset(x_train, y_train)
test_ids = ToyDataset(x_test, y_test)

train_loader = DataLoader(
    dataset=train_ids,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    drop_last=True, # 丢弃最后一个批次的数据
)

test_loader = DataLoader(
    dataset=test_ids,
    batch_size=2,
    shuffle=False, # 测试数据不需要打乱
    num_workers=0,
)

torch.manual_seed(123)
model = NeuralNetwork(num_inputs=2, num_outputs=2)

device =torch.device("cuda") #定义一个默认使用 GPU 的设备变量
model = model.to(device) # 将模型移动到 GPU 上

optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

num_epochs = 3

for epoch in range(num_epochs):

    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):
        features, labels = features.to(device), labels.to(device) # 将数据转移到  GPU 上
        logits = model(features)
        loss = torch.nn.functional.cross_entropy(logits, labels) # Loss fuction

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ### LOGGING
        print(f"Epoch {epoch + 1: 03d}/{num_epochs: 03d}"
              f" | Batch {batch_idx: 03d}/{len(train_loader): 03d}"
              f" | Train Loss: {loss:.2f}")

    model.eval()
    # 插入可选的模型评估代码

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

def compute_accuracy(model, dataloader):

    model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):

        with torch.no_grad():
            logits = model(features)

        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions # 根据标签是否匹配，返回一个 True/False 值的张量
        correct += torch.sum(compare) # 求和操作计算 True 值的数量
        total_examples += len(compare)

    return (correct / total_examples).item() # 正确预测的比例是一个介于 0 和 1 之间的值。调用 .item() 会将张量的值以 Python 浮点数的形式返回

import os
import torch

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost" # 主节点的地址
    os.environ["MASTER_PORT"] = "12345" # 机器上的任何空闲端口
    init_process_group(
        backend="nccl", # nccl 代表NVIDIA集体通信库
        rank=rank, # rank 指的是我们想要使用的GPU 的索引
        world_size=world_size # world_size是要使用的GPU 数量
    )
    torch.cuda.set_device(rank) # 设置当前的GPU 设备，以便在其上分配张量并执行操作

def prepare_dataset():
    # 插入数据集准备代码
    train_loader = DataLoader(
    dataset=train_ids,
    batch_size=2,
    shuffle=False, # DistributedSampler 现在负责打乱数据
    pin_memory=True, # 在 GPU 上训练时启用更快的内存传输
    drop_last=True,
    sampler=DistributedSampler(train_ids) # 将数据集分割成不同且不重叠的子集，以供每个进程（GPU）使用
    )
    return train_loader, test_loader

def main(rank, world_size, num_epochs): # 运行模型训练的主函数
    ddp_setup(rank, world_size)
    train_loader, test_loader = prepare_dataset()
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    model.to(rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    model = DDP(model, device_ids=[rank])
    for epoch in range(num_epochs):
        for features, labels in train_loader:
            features, labels = features.to(rank), labels.to(rank) # rank 是 GPU 的 ID
            # 插入模型预测和反向传播代码
            print(f"[GPU{rank}] Epoch: {epoch+1:03d}/{num_epochs:03d}"
                f" | Batchsize {labels.shape[0]:03d}"
                f" | Train/Val Loss: {loss:.2f}")
    model.eval()
    train_acc = compute_accuracy(model, train_loader, device=rank)
    print(f"[GPU{rank}] Training accuracy", train_acc)
    test_acc = compute_accuracy(model, test_loader, device=rank)
    print(f"[GPU{rank}] Test accuracy", test_acc)
    destroy_process_group() # 清理资源分配

if __name__ == "__main__":
    print("Number of GPUs available:", torch.cuda.device_count())
    torch.manual_seed(123)
    num_epochs = 3
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, num_epochs), nprocs=world_size) # 使用多个进程启动主函数，其中 nprocs=world_size 意味着每个 GPU 一个进程