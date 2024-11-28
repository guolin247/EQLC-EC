import os
import numpy as np
from sklearn.model_selection import train_test_split

# 定义路径
data_folder = r"D:\work_GuoLin\machine_learning\machinelearning\src\guolin\CNN_single_ICC\output_for_cnn"
output_folder = r"D:\work_GuoLin\machine_learning\machinelearning\src\guolin\CNN_single_ICC\stratified_split_data"

# 加载标签和文件名
labels_path = os.path.join(data_folder, 'labels.npy')
labels_data = np.load(labels_path, allow_pickle=True)

# 从 labels.npy 中提取文件名和对应的标签
file_names = labels_data[:, 0]  # 文件名
labels = labels_data[:, 1].astype(int)  # 标签

# 获取所有数据文件的名称
data_files = sorted(
    [f for f in os.listdir(data_folder) if f.endswith(".npy") and f != 'labels.npy']
)

# 确保标签中的文件名与实际文件夹中的文件匹配
assert set(file_names) == set(data_files), "文件名和数据文件不匹配"

# 使用 stratify 参数进行分层划分，直接按照标签进行划分
train_files, temp_files, train_labels, temp_labels = train_test_split(
    file_names, labels, test_size=0.2, stratify=labels, random_state=46
)

# # 对验证集和测试集再进行划分
# val_files, test_files, val_labels, test_labels = train_test_split(
#     temp_files, temp_labels, test_size=0.9, stratify=temp_labels, random_state=42
# )

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

# 加载并合并训练、验证和测试数据
X_train = load_and_combine(train_files)
# X_val = load_and_combine(val_files)
# X_test = load_and_combine(test_files)
X_test = load_and_combine(temp_files)
test_labels = temp_labels
# 保存数据和标签到新的 .npy 文件中
np.save(os.path.join(output_folder, 'X_train.npy'), X_train)
# np.save(os.path.join(output_folder, 'X_val.npy'), X_val)
np.save(os.path.join(output_folder, 'X_test.npy'), X_test)
np.save(os.path.join(output_folder, 'y_train.npy'), np.array(train_labels))
# np.save(os.path.join(output_folder, 'y_val.npy'), np.array(val_labels))
np.save(os.path.join(output_folder, 'y_test.npy'), np.array(test_labels))

print("数据集已分割并保存为单个 .npy 文件：")
print(f"输出文件夹: {output_folder}")
