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
        self.fc1 = nn.Linear(64 * 617, 128)  # Update to match the output size from conv+pool
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=dropout_value)  # 使用dropout_value

    def forward(self, x):
        x = x.unsqueeze(1)  # 添加通道维度
        x = self.pool(torch.relu(self.conv1(x)))  # 经过第一个卷积层和池化
        x = self.pool(torch.relu(self.conv2(x)))  # 经过第二个卷积层和池化
        x = x.view(-1, 64 * 617)  # 展平，确保尺寸匹配
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
data_folder = r"D:\work_GuoLin\machine_learning\machinelearning\src\guolin\CNN_single_HIP_CER\stratified_split_data"
test_data_path = os.path.join(data_folder, 'X_test.npy')
test_labels_path = os.path.join(data_folder, 'y_test.npy')

# 加载测试数据集
test_dataset = NumpyDataset(np.load(test_data_path), np.load(test_labels_path))
X_test, y_test = test_dataset.data, test_dataset.labels

# 加载所有模型
models_dir = r"D:\work_GuoLin\machine_learning\machinelearning\src\guolin\CNN_single_HIP_CER\models_for_ensemble"
model_files = [f for f in os.listdir(models_dir) if f.endswith('.pt')]  # 假设模型文件以.pt结尾
models = []
for model_file in model_files:
    model_path = os.path.join(models_dir, model_file)
    model = torch.load(model_path)
    model.eval()  # 设定为评估模式
    models.append(model.to(device))  # 移动到GPU（如果有）

# Bootstrap 方法设置
n_bootstrap_samples = 100  # 指定生成100个样本
n_test_samples = 240  # 测试集样本数量
results_hard_voting = []
results_soft_voting = []

# 对多个 bootstrap 测试集进行预测
for i in range(n_bootstrap_samples):
    # 随机选取样本索引
    bootstrap_indices = np.random.choice(X_test.shape[0], n_test_samples, replace=True)
    X_bootstrap = X_test[bootstrap_indices]
    y_bootstrap = y_test[bootstrap_indices]
    
    # 创建新的测试数据集和数据加载器
    bootstrap_dataset = NumpyDataset(X_bootstrap, y_bootstrap)
    test_loader = DataLoader(bootstrap_dataset, batch_size=1024, shuffle=False)

    # 存储所有模型的预测
    all_preds_hard = []
    all_probs_soft = []

    with torch.no_grad():
        # 对 bootstrap 测试集进行预测
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 对每个模型进行预测
            batch_preds_hard = []
            batch_probs_soft = []
            for model in models:
                outputs = model(inputs)
                # 硬投票
                _, predicted = torch.max(outputs.data, 1)
                batch_preds_hard.append(predicted.cpu().numpy())
                # 软投票
                batch_probs_soft.append(torch.softmax(outputs, dim=1).cpu().numpy())
            
            all_preds_hard.append(np.array(batch_preds_hard).T)
            all_probs_soft.append(np.mean(batch_probs_soft, axis=0))

    # 最终硬投票预测
    all_preds_hard = np.concatenate(all_preds_hard, axis=0)
    final_preds_hard = [np.argmax(np.bincount(preds)) for preds in all_preds_hard]

    # 最终软投票预测
    final_probs_soft = np.concatenate(all_probs_soft, axis=0)
    final_preds_soft = np.argmax(final_probs_soft, axis=1)

    # 计算硬投票评估指标
    accuracy_hard = accuracy_score(y_bootstrap, final_preds_hard)
    precision_hard, recall_hard, f1_hard, _ = precision_recall_fscore_support(y_bootstrap, final_preds_hard, average='binary')
    auc_hard = roc_auc_score(y_bootstrap, final_preds_hard)
    results_hard_voting.append({
        'Bootstrap Sample': i + 1,
        'Accuracy': accuracy_hard,
        'Precision': precision_hard,
        'Recall': recall_hard,
        'F1 Score': f1_hard,
        'AUC': auc_hard
    })

    # 计算软投票评估指标
    accuracy_soft = accuracy_score(y_bootstrap, final_preds_soft)
    precision_soft, recall_soft, f1_soft, _ = precision_recall_fscore_support(y_bootstrap, final_preds_soft, average='binary')
    auc_soft = roc_auc_score(y_bootstrap, final_probs_soft[:, 1])
    results_soft_voting.append({
        'Bootstrap Sample': i + 1,
        'Accuracy': accuracy_soft,
        'Precision': precision_soft,
        'Recall': recall_soft,
        'F1 Score': f1_soft,
        'AUC': auc_soft
    })

# 将结果保存到 CSV 文件
results_hard_voting_df = pd.DataFrame(results_hard_voting)
results_soft_voting_df = pd.DataFrame(results_soft_voting)

# 保存文件路径
result_folder = r"D:\work_GuoLin\machine_learning\machinelearning\src\guolin\CNN_single_HIP_CER\test"
os.makedirs(result_folder, exist_ok=True)  # 创建文件夹（如果不存在）
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # 获取当前时间

# 保存硬投票结果
hard_voting_result_file_path = os.path.join(result_folder, f'HardVoting_{current_time}.csv')
results_hard_voting_df.to_csv(hard_voting_result_file_path, index=False)
print(f'Hard voting results saved to {hard_voting_result_file_path}')

# 保存软投票结果
soft_voting_result_file_path = os.path.join(result_folder, f'SoftVoting_{current_time}.csv')
results_soft_voting_df.to_csv(soft_voting_result_file_path, index=False)
print(f'Soft voting results saved to {soft_voting_result_file_path}')