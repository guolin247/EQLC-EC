'''
直接在models文件夹进行的测试
'''

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
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

# 加载所有模型
models_dir = r"D:\work_GuoLin\machine_learning\machinelearning\src\guolin\CNN_CHD\models"
model_files = [f for f in os.listdir(models_dir) if f.endswith('.pt')]  # 假设模型文件以.pt结尾
models = []
model_names = []
for model_file in model_files:
    model_path = os.path.join(models_dir, model_file)
    model = torch.load(model_path)
    model.eval()  # 设定为评估模式
    models.append(model.to(device))  # 移动到GPU（如果有）
    model_names.append(model_file)

# 存储各个模型和投票处理的结果
results_overall = []

# 评估每个单独的模型在整个测试集上的表现
with torch.no_grad():
    for model, model_name in zip(models, model_names):
        all_preds = []
        all_probs = []
        all_labels = []
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.append(predicted.cpu().numpy())
            all_probs.append(torch.softmax(outputs, dim=1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)

        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
        auc = roc_auc_score(all_labels, all_probs[:, 1])

        results_overall.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1_score,
            'AUC': auc
        })

# 计算软投票和硬投票的结果
all_preds_hard = []
all_probs_soft = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        batch_preds_hard = []
        batch_probs_soft = []
        for model in models:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            batch_preds_hard.append(predicted.cpu().numpy())
            batch_probs_soft.append(torch.softmax(outputs, dim=1).cpu().numpy())

        all_preds_hard.append(np.array(batch_preds_hard).T)
        all_probs_soft.append(np.mean(batch_probs_soft, axis=0))
        all_labels.append(labels.cpu().numpy())

all_preds_hard = np.concatenate(all_preds_hard, axis=0)
final_preds_hard = [np.argmax(np.bincount(preds)) for preds in all_preds_hard]

final_probs_soft = np.concatenate(all_probs_soft, axis=0)
final_preds_soft = np.argmax(final_probs_soft, axis=1)
all_labels = np.concatenate(all_labels)

accuracy_hard = accuracy_score(all_labels, final_preds_hard)
precision_hard, recall_hard, f1_hard, _ = precision_recall_fscore_support(all_labels, final_preds_hard, average='binary')
auc_hard = roc_auc_score(all_labels, final_preds_hard)

results_overall.append({
    'Model': 'Hard Voting',
    'Accuracy': accuracy_hard,
    'Precision': precision_hard,
    'Recall': recall_hard,
    'F1 Score': f1_hard,
    'AUC': auc_hard
})

accuracy_soft = accuracy_score(all_labels, final_preds_soft)
precision_soft, recall_soft, f1_soft, _ = precision_recall_fscore_support(all_labels, final_preds_soft, average='binary')
auc_soft = roc_auc_score(all_labels, final_probs_soft[:, 1])

results_overall.append({
    'Model': 'Soft Voting',
    'Accuracy': accuracy_soft,
    'Precision': precision_soft,
    'Recall': recall_soft,
    'F1 Score': f1_soft,
    'AUC': auc_soft
})

# 保存每个模型的整体测试表现
results_overall_df = pd.DataFrame(results_overall)
result_folder = r"D:\work_GuoLin\machine_learning\machinelearning\src\guolin\CNN_CHD\test"
os.makedirs(result_folder, exist_ok=True)
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
overall_result_file_path = os.path.join(result_folder, f'ModelPerformance_{current_time}.csv')
results_overall_df.to_csv(overall_result_file_path, index=False)
print(f'Model performance results saved to {overall_result_file_path}')
