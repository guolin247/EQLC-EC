import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import DenseNet121_Weights

# 设置随机种子以保证重复性
seed = 43  # 你可以选择任何整数作为随机种子
torch.manual_seed(seed)
np.random.seed(seed)

# 定义数据集类
class NumpyDataset(Dataset):
    def __init__(self, data_path, labels_path):
        self.data = np.load(data_path)  # 加载数据
        self.labels = np.load(labels_path)  # 加载标签

        # 定义数据预处理转换
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 将数组转换为 Tensor，格式为 (H, W) -> (C, H, W)
            transforms.Lambda(lambda x: x.to(torch.float32))  # 确保数据类型为 float32
        ])

    def __len__(self):
        return len(self.labels)  # 返回数据集大小

    def __getitem__(self, idx):
        sample = self.data[idx]  # 获取样本
        sample = self.transform(sample)  # 应用转换

        label = self.labels[idx]  # 获取标签
        label = torch.tensor(label, dtype=torch.long)  # 转换为 Long Tensor

        return sample, label  # 返回样本和标签

# 打印数据集信息的函数
def print_dataset_info(dataset, dataset_name):
    all_labels = [dataset[i][1].item() for i in range(len(dataset))]
    print(f"{dataset_name} size: {len(dataset)} samples")
    print(f"{dataset_name} shape (first sample): {dataset[0][0].shape}")
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    print(f"{dataset_name} class distribution:")
    for label, count in zip(unique_labels, counts):
        print(f" - Class {label}: {count} samples")

# 定义使用 DenseNet121 的模型
class DenseNet121TripleChannel(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet121TripleChannel, self).__init__()
        self.densenet = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        
        # 修改第一层卷积的输入通道数从 1 改为 3
        self.densenet.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # 修改分类头
        num_ftrs = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.densenet(x)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据文件路径
data_folder = r"D:\work_GuoLin\machine_learning\machinelearning\src\guolin\CNN_single_ICC\stratified_split_data"
train_data_path = os.path.join(data_folder, 'X_train.npy')
train_labels_path = os.path.join(data_folder, 'y_train.npy')
test_data_path = os.path.join(data_folder, 'X_test.npy')
test_labels_path = os.path.join(data_folder, 'y_test.npy')

# 加载数据集
train_dataset = NumpyDataset(train_data_path, train_labels_path)
test_dataset = NumpyDataset(test_data_path, test_labels_path)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True, worker_init_fn=lambda _: np.random.seed(seed))
test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False)

# 打印数据集信息
print_dataset_info(train_dataset, "Training set")
print_dataset_info(test_dataset, "Testing set")

# 计算类别权重
def compute_class_weights(dataset):
    all_labels = [dataset[i][1].item() for i in range(len(dataset))]
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    weights = 1.0 / counts  
    weights = torch.tensor(weights, dtype=torch.float32).to(device)
    return weights

# 获取类别权重
class_weights = compute_class_weights(train_dataset)

# 使用交叉熵损失函数和类别权重
criterion = nn.CrossEntropyLoss(weight=class_weights)

# 添加评估步骤
def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    loss = 0.0
    all_preds = []
    all_labels = []
    probabilities = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss += criterion(outputs, labels).item()

            # 存储 softmax 概率用于 AUC 计算
            probs = torch.softmax(outputs, dim=1)
            probabilities.append(probs[:, 1].cpu().numpy())  # 只保留正类的概率
            _, predicted = torch.max(outputs.data, 1)
            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    probabilities = np.concatenate(probabilities)

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')

    # 计算 AUC
    auc = roc_auc_score(all_labels, probabilities)

    return loss / len(data_loader), acc, precision, recall, f1, auc

# 训练模型
def train_model(model, train_loader, test_loader, num_epochs=50, learning_rate=0.001, criterion=None):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.99, patience=10)

    model.train()
    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}/{num_epochs}, Current learning rate: {current_lr:.6f}")
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 评估测试集
        test_loss, test_acc, test_precision, test_recall, test_f1, test_auc = evaluate_model(model, test_loader, criterion, device)

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {running_loss / len(train_loader):.4f}, '
              f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, '
              f'Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, '
              f'F1: {test_f1:.4f}, AUC: {test_auc:.4f}')

# 主程序
num_classes = 2  # 设置为二分类任务
model = DenseNet121TripleChannel(num_classes=num_classes).to(device)

# 训练模型
train_model(model, train_loader, test_loader, num_epochs=200, learning_rate=0.0001, criterion=criterion)
