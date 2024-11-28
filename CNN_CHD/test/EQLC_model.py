import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from datetime import datetime

# 定义一维卷积神经网络的类（假设此定义已经存在）
class Simple1DCNN(nn.Module):
    def __init__(self, input_size, num_classes, dropout_value=0.05):
        super(Simple1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=2)  # 32 filters
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=2)  # 64 filters
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)  # 池化层
        self.fc1 = nn.Linear(64 * 51, 128)  # Update to match the output size from conv+pool
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=dropout_value)  # 使用dropout_value

    def forward(self, x):
        x = x.unsqueeze(1)  # 添加通道维度
        x = self.pool(torch.relu(self.conv1(x)))  # 经过第一个卷积层和池化
        x = self.pool(torch.relu(self.conv2(x)))  # 经过第二个卷积层和池化
        x = x.view(-1, 64 * 51)  # 展平，确保尺寸匹配
        x = self.dropout(x)  # 应用dropout
        x = torch.relu(self.fc1(x))  # 全连接层
        x = self.fc2(x)  # 输出层
        return x

# 定义数据集类
class NumpyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        sample = torch.tensor(sample, dtype=torch.float32).squeeze()
        label = torch.tensor(label, dtype=torch.long)
        return sample, label

# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据文件路径
data_folder = r"D:\work_GuoLin\machine_learning\machinelearning\src\guolin\CNN_CHD\stratified_split_data"
test_data_path = os.path.join(data_folder, 'X_test_combat.npy')
test_labels_path = os.path.join(data_folder, 'y_test_combat.npy')

# 加载测试数据集
test_dataset = NumpyDataset(np.load(test_data_path), np.load(test_labels_path))
X_test, y_test = test_dataset.data, test_dataset.labels

# 加载特定模型
models_dir = r"D:\work_GuoLin\machine_learning\machinelearning\src\guolin\CNN_CHD\models"
model_files = [f for f in os.listdir(models_dir) if 'Simple1DCNN_5e-05_256_200_0.50_seed41' in f and f.endswith('.pt')]  # 筛选出特定模型
assert len(model_files) == 1, "应该有且仅有一个匹配的模型文件"
model_path = os.path.join(models_dir, model_files[0])
model = torch.load(model_path)
model.eval()  # 设定为评估模式
model = model.to(device)  # 移动到GPU（如果有）

# Bootstrap 方法设置
n_bootstrap_samples = 100  # 指定生成100个样本
n_test_samples = 1000  # 测试集样本数量
results = []

# 对多个 bootstrap 测试集进行预测
for i in range(n_bootstrap_samples):
    # 随机选取样本索引
    bootstrap_indices = np.random.choice(X_test.shape[0], n_test_samples, replace=True)
    X_bootstrap = X_test[bootstrap_indices]
    y_bootstrap = y_test[bootstrap_indices]
    
    # 创建新的测试数据集和数据加载器
    bootstrap_dataset = NumpyDataset(X_bootstrap, y_bootstrap)
    test_loader = DataLoader(bootstrap_dataset, batch_size=1024, shuffle=False)

    preds = []
    probs = []

    with torch.no_grad():
        # 对 bootstrap 测试集进行预测
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            preds.append(predicted.cpu().numpy())
            probs.append(torch.softmax(outputs, dim=1).cpu().numpy())

    final_preds = np.concatenate(preds)
    final_probs = np.concatenate(probs)[:, 1]

    # 计算评估指标
    accuracy = accuracy_score(y_bootstrap, final_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_bootstrap, final_preds, average='binary')
    auc = roc_auc_score(y_bootstrap, final_probs)
    results.append({
        'Bootstrap Sample': i + 1,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC': auc
    })

# 将结果保存到 CSV 文件
results_df = pd.DataFrame(results)

# 保存文件路径
result_folder = r"D:\work_GuoLin\machine_learning\machinelearning\src\guolin\CNN_CHD\test"
os.makedirs(result_folder, exist_ok=True)  # 创建文件夹（如果不存在）

# 保存结果
result_file_path = os.path.join(result_folder, 'EQLC_.csv')
results_df.to_csv(result_file_path, index=False)
print(f'Results saved to {result_file_path}')
