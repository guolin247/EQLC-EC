import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import time

# 设置随机种子以保证重复性
seed = 46
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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

# 定义一维卷积神经网络
class Simple1DCNN(nn.Module):
    def __init__(self, input_size, num_classes, dropout_value=0.05):
        super(Simple1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1_out_size = (input_size + 2 * 2 - 3) + 1
        self.pool1_out_size = self.conv1_out_size // 2        
        self.conv2_out_size = (self.pool1_out_size + 2 * 2 - 3) + 1
        self.pool2_out_size = self.conv2_out_size // 2
        self.fc1 = nn.Linear(64 * self.pool2_out_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=dropout_value)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * self.pool2_out_size)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据文件路径
data_folder = r"D:\work_GuoLin\machine_learning\machinelearning\src\guolin\CNN_single_HIP_CER\stratified_split_data"
train_data_path = os.path.join(data_folder, 'X_train.npy')
train_labels_path = os.path.join(data_folder, 'y_train.npy')
test_data_path = os.path.join(data_folder, 'X_test.npy')
test_labels_path = os.path.join(data_folder, 'y_test.npy')

# 加载数据集
train_dataset = NumpyDataset(train_data_path, train_labels_path)
test_dataset = NumpyDataset(test_data_path, test_labels_path)

# 创建数据加载器
batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=lambda x: np.random.seed(seed + x))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义模型和优化器
num_classes = 2
input_size = 2463
dropout_value = 0.5
model = Simple1DCNN(input_size=input_size, num_classes=num_classes, dropout_value=dropout_value).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0004, weight_decay=0.001)
criterion = nn.CrossEntropyLoss()

# 统计运行时间
epochs = 100
repeats = 11
train_epoch_times = []
predict_times = []

for repeat in range(repeats):
    print(f"Run {repeat + 1}/{repeats}")
    model.train()
    total_train_time = 0.0

    # 训练时间统计
    for epoch in range(epochs):
        epoch_start_time = time.perf_counter()
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        epoch_end_time = time.perf_counter()
        total_train_time += epoch_end_time - epoch_start_time

    avg_train_epoch_time = total_train_time / epochs
    if repeat > 0:  # 跳过第1次预热结果
        train_epoch_times.append(avg_train_epoch_time)

    # 预测时间统计
    model.eval()
    start_predict_time = time.perf_counter()
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            _ = model(inputs)
    end_predict_time = time.perf_counter()
    
    predict_time = end_predict_time - start_predict_time
    if repeat > 0:  # 跳过第1次预热结果
        predict_times.append(predict_time)

    print(f"Run {repeat + 1} - Avg Train Epoch Time: {avg_train_epoch_time:.4f} seconds, Predict Time: {predict_time:.4f} seconds")

# 保存结果到CSV文件
df_results = pd.DataFrame({
    'Run': list(range(2, repeats + 1)),  # 从第2次到第11次
    'Avg_Train_Epoch_Time': train_epoch_times,
    'Predict_Time': predict_times
})
df_results['Avg_Train_Epoch_Time_Mean'] = df_results['Avg_Train_Epoch_Time'].mean()
df_results['Avg_Train_Epoch_Time_Std'] = df_results['Avg_Train_Epoch_Time'].std()
df_results['Predict_Time_Mean'] = df_results['Predict_Time'].mean()
df_results['Predict_Time_Std'] = df_results['Predict_Time'].std()

output_file = 'deep_learning_model_runtime_with_prediction.csv'
df_results.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")
