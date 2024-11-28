import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import pandas as pd  # 导入pandas库以处理数据帧

# 设置随机种子以保证重复性
seed = 41
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True  # 确保在使用GPU时有可复现性
torch.backends.cudnn.benchmark = False  # 禁用以获得稳定运行时间

# 定义数据集类
class NumpyDataset(Dataset):
    def __init__(self, data_path, labels_path):
        self.data = np.load(data_path)
        self.labels = np.load(labels_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        sample = torch.tensor(sample, dtype=torch.float32).squeeze()
        label = torch.tensor(label, dtype=torch.long)
        return sample, label

# 打印数据集信息的函数
def print_dataset_info(dataset, dataset_name):
    all_labels = [dataset[i][1].item() for i in range(len(dataset))]
    print(f"{dataset_name} size: {len(dataset)} samples")
    print(f"{dataset_name} shape (first sample): {dataset[0][0].shape}")
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    print(f"{dataset_name} class distribution:")
    for label, count in zip(unique_labels, counts):
        print(f" - Class {label}: {count} samples")

# 定义一维卷积神经网络
class Simple1DCNN(nn.Module):
    def __init__(self, input_size, num_classes, dropout_value=0.05):
        super(Simple1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=2)  # 32 filters
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=2)  # 64 filters
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)  # 池化层
        
        # 计算通过卷积和池化后的输入特征图的大小
        # 第一次卷积
        self.conv1_out_size = (input_size + 2 * 2 - 3) + 1  # 1D卷积输出计算
        # 第一次池化
        self.pool1_out_size = self.conv1_out_size // 2        
        # 第二次卷积
        self.conv2_out_size = (self.pool1_out_size + 2 * 2 - 3) + 1  # 1D卷积
        # 第二次池化
        self.pool2_out_size = self.conv2_out_size // 2
        
        # 定义全连接层输入的大小
        self.fc1 = nn.Linear(64 * self.pool2_out_size, 128)  # Adjust to match
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=dropout_value)  # 使用dropout_value

    def forward(self, x):
        x = x.unsqueeze(1)  # 添加通道维度
        x = self.pool(torch.relu(self.conv1(x)))  # 经过第一个卷积层和池化
        x = self.pool(torch.relu(self.conv2(x)))  # 经过第二个卷积层和池化
        x = x.view(-1, 64 * self.pool2_out_size)  # 展平，确保尺寸匹配
        x = self.dropout(x)  # 应用dropout
        x = torch.relu(self.fc1(x))  # 全连接层
        x = self.fc2(x)  # 输出层
        return x

# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据文件路径
data_folder = r"D:\work_GuoLin\machine_learning\machinelearning\src\guolin\CNN_CHD\stratified_split_data"
train_data_path = os.path.join(data_folder, 'X_train_combat.npy')
train_labels_path = os.path.join(data_folder, 'y_train_combat.npy')
test_data_path = os.path.join(data_folder, 'X_test_combat.npy')
test_labels_path = os.path.join(data_folder, 'y_test_combat.npy')

# 加载数据集
train_dataset = NumpyDataset(train_data_path, train_labels_path)
test_dataset = NumpyDataset(test_data_path, test_labels_path)

# 创建数据加载器
batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=lambda x: np.random.seed(seed + x))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
    auc = roc_auc_score(all_labels, probabilities)

    return loss / len(data_loader), acc, precision, recall, f1, auc

# 训练模型
# 训练模型
# 训练模型
def train_model(model, train_loader, test_loader, num_epochs=50, learning_rate=0.1, criterion=None, weight_decay=1e-5, dropout_value=0.0):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train()

    # 创建输出文件名
    model_name = "Simple1DCNN"
    csv_file_name = f"training_log_b{batch_size}_do{dropout_value}_lr{learning_rate}_wd{weight_decay}.csv"
    
    # 创建一个空的 DataFrame，用于保存训练结果
    columns = ['Epoch', 'Train Loss', 'Test Loss', 'Test Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    log_df = pd.DataFrame(columns=columns)

    for epoch in range(num_epochs):
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 每个epoch结束后评估模型
        test_loss, test_acc, test_precision, test_recall, test_f1, test_auc = evaluate_model(model, test_loader, criterion, device)

        # 创建日志信息并添加到DataFrame
        log_df.loc[epoch] = [epoch + 1, running_loss / len(train_loader), test_loss, test_acc, test_precision, test_recall, test_f1, test_auc]

        # 打印到控制台
        log_info = (f'Epoch [{epoch + 1:2d}/{num_epochs:2d}], '
                    f'Train Loss: {running_loss / len(train_loader):8.4f}, '
                    f'Test Loss: {test_loss:8.4f}, '
                    f'Test Accuracy: {test_acc:6.4f}, '
                    f'Precision: {test_precision:6.4f}, '
                    f'Recall: {test_recall:6.4f}, '
                    f'F1: {test_f1:6.4f}, '
                    f'AUC: {test_auc:6.4f}\n')
        
        print(log_info, end='')

        # 更新学习率
        if (epoch + 1) % 5 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.95 # 学习率乘以0.99
            print(f'Learning rate updated to {optimizer.param_groups[0]["lr"]:.6f}')

    # 训练结束后保存日志为CSV文件
    log_df.to_csv(csv_file_name, index=False)
    print(f'Training log saved to {csv_file_name}')

    # 训练结束后获取最终准确率
    final_accuracy = test_acc

    # 保存完整模型
    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # 使用随机种子和最终准确率命名模型
    model_save_path = os.path.join(model_dir, 
        f"{model_name}_{learning_rate}_{batch_size}_{num_epochs}_{dropout_value:.2f}_seed{seed}_acc{final_accuracy:.4f}.pt")  
    
    torch.save(model, model_save_path)  # 保存完整模型
    print(f'Model saved to {model_save_path}')

# 主程序部分
num_classes = 2
input_size = 199  # 根据你的数据调整
dropout_value = 0.5  # 将 dropout 值作为参数传递
model = Simple1DCNN(input_size=input_size, num_classes=num_classes, dropout_value=dropout_value).to(device)

# 训练模型，并设置L2正则化的权重衰减参数
train_model(model, train_loader, test_loader, num_epochs=200, learning_rate=0.00006, criterion=criterion, weight_decay=0.001, dropout_value=dropout_value)
