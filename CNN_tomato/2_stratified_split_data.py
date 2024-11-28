import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 定义路径
base_folder = r"D:\work_GuoLin\machine_learning\machinelearning\src\guolin\CNN_TOMATO"
data_folder = os.path.join(base_folder, 'processed_data')
output_folder = os.path.join(base_folder, 'stratified_split_data')

# 加载标签和文件名
labels_path = os.path.join(data_folder, 'labels.npy')
labels_data = np.load(labels_path, allow_pickle=True)

# 提取 sample_id 和对应的标签
file_names = labels_data[:, 0]  # 文件名
labels = labels_data[:, 1].astype(int)  # 标签
sample_ids = labels_data[:, 2]  # sample_id

# 创建 DataFrame 以方便处理
df = pd.DataFrame({
    'file_name': file_names,
    'label': labels,
    'sample_id': sample_ids
})

# 生成唯一的 sample_id 和对应标签
unique_ids = df[['sample_id', 'label']].drop_duplicates()

# 确保标签数量和sample_id数量相同，用于后续的分层分配
assert unique_ids['sample_id'].nunique() == unique_ids.shape[0], "标签数量应等于唯一sample_id数量"

# 使用 stratified sampling 进行分层划分
train_ids, test_ids = train_test_split(
    unique_ids,
    test_size=0.29,  # 手动设定测试集比例
    stratify=unique_ids['label'],
    # random_state=172
    random_state=42
)

# 从 DataFrame 中提取对应的文件名
train_sample_ids = train_ids['sample_id'].values
test_sample_ids = test_ids['sample_id'].values

# 根据sample_id从原始DataFrame提取文件名和标签
train_files = df[df['sample_id'].isin(train_sample_ids)]
test_files = df[df['sample_id'].isin(test_sample_ids)]

# 将文件名和标签提取为列表
train_file_names = train_files['file_name'].values
train_labels = train_files['label'].values
test_file_names = test_files['file_name'].values
test_labels = test_files['label'].values

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# Function to load .npy files and combine them into a single array
def load_and_combine(file_list):
    combined_data = []
    for file_name in file_list:
        file_path = os.path.join(data_folder, file_name)
        data = np.load(file_path)
        combined_data.append(data)
    return np.array(combined_data)

# 加载训练和测试数据
X_train = load_and_combine(train_file_names)
X_test = load_and_combine(test_file_names)

# 保存数据和标签到新的 .npy 文件中
np.save(os.path.join(output_folder, 'X_train.npy'), X_train)
np.save(os.path.join(output_folder, 'X_test.npy'), X_test)
np.save(os.path.join(output_folder, 'y_train.npy'), train_labels)
np.save(os.path.join(output_folder, 'y_test.npy'), test_labels)

print("数据集已分割并保存为单个 .npy 文件：")
print(f"输出文件夹: {output_folder}")

