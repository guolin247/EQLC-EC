import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from neuroCombat import neuroCombat

# 设置随机种子以保证重复性
seed = 52
np.random.seed(seed)

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

# 手动指定训练集和测试集的批次
train_batches = [3, 4]  # 可以修改为包含多个批次的列表，例如 [1, 2]
test_batches = [2]  # 可以修改为包含多个批次的列表，例如 [3]

# 根据指定的批次划分训练集和测试集
train_indices = [i for i, batch in enumerate(batches) if batch in train_batches]
test_indices = [i for i, batch in enumerate(batches) if batch in test_batches]

train_data = []
train_labels = []
for idx in train_indices:
    file_path = os.path.join(data_folder, file_names[idx])
    sample = np.load(file_path)  # 加载样本数据，假设形状为 (1, 199)
    train_data.append(sample)
    train_labels.append(labels[idx])

train_data = np.array(train_data)  # 训练集数据形状 (n_train, 1, 199)
train_labels = np.array(train_labels)

test_data = []
test_labels = []
for idx in test_indices:
    file_path = os.path.join(data_folder, file_names[idx])
    sample = np.load(file_path)  # 加载样本数据，假设形状为 (1, 199)
    test_data.append(sample)
    test_labels.append(labels[idx])

test_data = np.array(test_data)  # 测试集数据形状 (n_test, 1, 199)
test_labels = np.array(test_labels)

# 将数据重塑为二维数组以适应Combat方法
train_data_reshaped = train_data.reshape(train_data.shape[0], -1)  # 形状 (n_train, 199)
test_data_reshaped = test_data.reshape(test_data.shape[0], -1)  # 形状 (n_test, 199)

# 合并训练集和测试集数据，以应用Combat进行批次效应消除
combined_data = np.concatenate((train_data_reshaped, test_data_reshaped), axis=0)

# 创建批次变量，训练集批次为指定的训练批次，测试集批次为指定的测试批次
batch_labels = pd.Series(np.concatenate((batches[train_indices], batches[test_indices])))

# 将数据转换为DataFrame，以适应neuroCombat方法
data_df = pd.DataFrame(combined_data)
covars_df = pd.DataFrame({'batch': batch_labels})

# 对数据进行标准化
scaler = StandardScaler()
combined_data_scaled = scaler.fit_transform(data_df)

# 使用Combat进行批次效应消除
combat_result = neuroCombat(dat=combined_data_scaled.T, covars=covars_df, batch_col='batch')
combined_data_combat = combat_result['data'].T

# 将数据分回训练集和测试集
train_data_combat = combined_data_combat[:train_data.shape[0], :].reshape(train_data.shape)  # 形状 (n_train, 1, 199)
test_data_combat = combined_data_combat[train_data.shape[0]:, :].reshape(test_data.shape)  # 形状 (n_test, 1, 199)

# 保存新的训练集和测试集
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

train_data_combat_path = os.path.join(output_folder, 'X_train_combat.npy')
train_labels_combat_path = os.path.join(output_folder, 'y_train_combat.npy')
test_data_combat_path = os.path.join(output_folder, 'X_test_combat.npy')
test_labels_combat_path = os.path.join(output_folder, 'y_test_combat.npy')

np.save(train_data_combat_path, train_data_combat)
np.save(train_labels_combat_path, train_labels)
np.save(test_data_combat_path, test_data_combat)
np.save(test_labels_combat_path, test_labels)

print(f"Combat-corrected training data saved to {train_data_combat_path}")
print(f"Combat-corrected training labels saved to {train_labels_combat_path}")
print(f"Combat-corrected testing data saved to {test_data_combat_path}")
print(f"Combat-corrected testing labels saved to {test_labels_combat_path}")
