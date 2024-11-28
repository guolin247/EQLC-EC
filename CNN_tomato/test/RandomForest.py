import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # 导入随机森林分类器
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from datetime import datetime

# 设置随机种子以保证重复性
seed = 42
np.random.seed(seed)

# 定义数据集类
class NumpyDataset:
    def __init__(self, data_path, labels_path):
        self.data = np.load(data_path)  # 加载 numpy 数据
        self.labels = np.load(labels_path)  # 加载标签
    
    def get_data(self):
        return self.data, self.labels

# 数据文件路径
data_folder = r"D:\work_GuoLin\machine_learning\machinelearning\src\guolin\CNN_single_HIP_CER\stratified_split_data"
train_data_path = os.path.join(data_folder, 'X_train.npy')
train_labels_path = os.path.join(data_folder, 'y_train.npy')
test_data_path = os.path.join(data_folder, 'X_test.npy')
test_labels_path = os.path.join(data_folder, 'y_test.npy')

# 加载数据集
train_dataset = NumpyDataset(train_data_path, train_labels_path)
test_dataset = NumpyDataset(test_data_path, test_labels_path)

# 获取训练和测试数据
X_train, y_train = train_dataset.get_data()
X_test, y_test = test_dataset.get_data()

# 按需修改数据形状
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# 定义最佳参数
best_params = (500, None, 5, 2, 'sqrt')  # (n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features)
n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features = best_params

# 创建随机森林模型
rf_model = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    max_features=max_features,
    random_state=seed
)

# 训练模型
rf_model.fit(X_train, y_train)

# Bootstrap 方法设置
n_bootstrap_samples = 100  # 指定生成 50 个样本
# n_test_samples = X_test.shape[0]  # 测试集样本数量
n_test_samples = 240  # 测试集样本数量
results = []

for i in range(n_bootstrap_samples):
    # 随机选取样本索引
    bootstrap_indices = np.random.choice(n_test_samples, n_test_samples, replace=True)
    
    # 生成新的测试集
    X_bootstrap = X_test[bootstrap_indices]
    y_bootstrap = y_test[bootstrap_indices]

    # 在新的测试集上进行预测
    y_pred = rf_model.predict(X_bootstrap)
    y_proba = rf_model.predict_proba(X_bootstrap)[:, 1]  # 预测概率

    # 计算指标
    test_accuracy = accuracy_score(y_bootstrap, y_pred)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(y_bootstrap, y_pred, average='binary')
    test_auc = roc_auc_score(y_bootstrap, y_proba)

    # 保存结果
    results.append({
        'Bootstrap Sample': i + 1,
        'Accuracy': test_accuracy,
        'Precision': test_precision,
        'Recall': test_recall,
        'F1 Score': test_f1,
        'AUC': test_auc
    })

# 将结果保存到 CSV 文件
results_df = pd.DataFrame(results)

# 保存文件路径
result_folder = r"D:\work_GuoLin\machine_learning\machinelearning\src\guolin\CNN_single_HIP_CER\test"
os.makedirs(result_folder, exist_ok=True)  # 创建文件夹（如果不存在）
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # 获取当前时间
result_file_path = os.path.join(result_folder, f'RandomForest_{current_time}.csv')

# 保存到 CSV 文件
results_df.to_csv(result_file_path, index=False)

# 输出完成信息
print(f'Results saved to {result_file_path}')
