import os
import numpy as np

# 定义路径
base_folder = r"D:\work_GuoLin\machine_learning\machinelearning\src\guolin\CNN_CHD"
data_folder = os.path.join(base_folder, 'output_for_cnn')
output_folder = os.path.join(base_folder, 'stratified_split_data')

# 加载标签和文件名
labels_path = os.path.join(data_folder, 'labels.npy')
labels_data = np.load(labels_path, allow_pickle=True)

# 从 labels.npy 中提取文件名、对应的标签和批次
file_names = labels_data[:, 0]  # 文件名
labels = labels_data[:, 1].astype(int)  # 数字化标签
batches = labels_data[:, 2].astype(int)  # 批次信息

# 获取所有数据文件的名称
data_files = sorted(
    [f for f in os.listdir(data_folder) if f.endswith(".npy") and f != 'labels.npy']
)

# 确保标签中的文件名与实际文件夹中的文件匹配
assert set(file_names) == set(data_files), "文件名和数据文件不匹配"

# 手动划分数据集
train_files = []
train_labels = []
test_files = []
test_labels = []

# 指定批次划分
for idx, batch in enumerate(batches):
    if batch in [1, 3]:  # 选择批次数作为训练集
        train_files.append(file_names[idx])
        train_labels.append(labels[idx])
    elif batch in [4]:  # 选择批次数作为测试集
        test_files.append(file_names[idx])
        test_labels.append(labels[idx])

# 转换为 NumPy 数组
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

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

# 加载并合并训练和测试数据
X_train = load_and_combine(train_files)
X_test = load_and_combine(test_files)

# 保存数据和标签到新的 .npy 文件中
np.save(os.path.join(output_folder, 'X_train.npy'), X_train)
np.save(os.path.join(output_folder, 'X_test.npy'), X_test)
np.save(os.path.join(output_folder, 'y_train.npy'), train_labels)
np.save(os.path.join(output_folder, 'y_test.npy'), test_labels)

print("数据集已分割并保存为单个 .npy 文件：")
print(f"输出文件夹: {output_folder}")
