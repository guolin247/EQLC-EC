import os
import numpy as np

# 定义路径
base_folder = r"D:\work_GuoLin\machine_learning\machinelearning\src\guolin\CNN_MI"
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

# 手动划分开发集和测试集
dev_files = []
dev_labels = []
test_files = []
test_labels = []

# 指定批次划分
for idx, batch in enumerate(batches):
    if batch in [2, 3]:  # 批次2和3作为开发集
        dev_files.append(file_names[idx])
        dev_labels.append(labels[idx])
    elif batch == 1:  # 批次1作为测试集
        test_files.append(file_names[idx])
        test_labels.append(labels[idx])

# 转换为 NumPy 数组
dev_labels = np.array(dev_labels)
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

# 加载并合并开发集和测试数据
X_dev = load_and_combine(dev_files)
X_test = load_and_combine(test_files)

# 拆分开发集为训练集和验证集（例如，80%为训练集，20%为验证集）
def split_dev_set(X_dev, dev_labels, train_ratio=0.8):
    num_train = int(len(X_dev) * train_ratio)
    indices = np.random.permutation(len(X_dev))
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    X_train = X_dev[train_indices]
    X_val = X_dev[val_indices]
    y_train = dev_labels[train_indices]
    y_val = dev_labels[val_indices]

    return X_train, X_val, y_train, y_val

# # 划分训练集和验证集
# X_train, X_val, y_train, y_val = split_dev_set(X_dev, dev_labels)
X_train = X_dev
y_train = dev_labels
# 保存数据和标签到新的 .npy 文件中
np.save(os.path.join(output_folder, 'X_train.npy'), X_train)
# np.save(os.path.join(output_folder, 'X_val.npy'), X_val)
np.save(os.path.join(output_folder, 'X_test.npy'), X_test)
np.save(os.path.join(output_folder, 'y_train.npy'), y_train)
# np.save(os.path.join(output_folder, 'y_val.npy'), y_val)
np.save(os.path.join(output_folder, 'y_test.npy'), test_labels)

print("数据集已分割并保存为单个 .npy 文件：")
print(f"输出文件夹: {output_folder}")
