import os
import numpy as np

# 定义路径
output_folder = r"D:\work_GuoLin\machine_learning\machinelearning\src\guolin\CNN_single_ICC\stratified_split_data"
x_train_path = os.path.join(output_folder, 'X_train.npy')
y_train_path = os.path.join(output_folder, 'y_train.npy')

# 加载原始训练集和标签
X_train = np.load(x_train_path, allow_pickle=True)
y_train = np.load(y_train_path)

# 获取样本数量
n_samples = X_train.shape[0]

def bootstrap_sample(X, y, random_seed):
    """
    使用Bootstrap方法对数据集进行重采样
    :param X: 原始特征数据
    :param y: 原始标签
    :param random_seed: 随机种子用于重采样
    :return: 重采样后的特征数据和标签
    """
    np.random.seed(random_seed)
    indices = np.random.choice(n_samples, size=n_samples, replace=True)  # 重采样
    X_sample = X[indices]  # 生成重采样的特征数据
    y_sample = y[indices]  # 生成重采样的标签
    return X_sample, y_sample

# 设置要生成的随机种子数量
num_samples = 10  # 可以根据需要修改
random_seeds = range(42, 42 + num_samples)  # 从42开始，每次增加1的随机种子

# 创建输出文件夹（如果不存在）
resampled_folder = os.path.join(output_folder, 'bootstrap_samples')
os.makedirs(resampled_folder, exist_ok=True)

# 循环生成每个随机种子的重采样数据
for seed in random_seeds:
    X_resampled, y_resampled = bootstrap_sample(X_train, y_train, seed)

    # 保存重采样的数据和标签
    np.save(os.path.join(resampled_folder, f'X_train_bootstrap_{seed}.npy'), X_resampled)
    np.save(os.path.join(resampled_folder, f'y_train_bootstrap_{seed}.npy'), y_resampled)

    print(f"重采样后的数据集已保存：")
    print(f"训练集特征文件: {os.path.join(resampled_folder, f'X_train_bootstrap_{seed}.npy')}")
    print(f"训练集标签文件: {os.path.join(resampled_folder, f'y_train_bootstrap_{seed}.npy')}")
