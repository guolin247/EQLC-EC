'''
Best Parameters: (num_leaves=31, max_depth=20, learning_rate=0.1, n_estimators=200) -> Best Cross-Validated AUC: 0.8273
'''

import os
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import cross_val_score, KFold
from itertools import product
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
data_folder = r"D:\work_GuoLin\machine_learning\machinelearning\src\guolin\CNN_single_ICC\stratified_split_data"
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

# 定义参数搜索空间
num_leaves = [31, 50, 100]  # 叶子节点数
max_depth = [-1, 10, 20, 30]  # 最大深度（-1意味着无限制）
learning_rate = [0.01, 0.05, 0.1, 0.2]  # 学习率
n_estimators = [50, 100, 200]  # 迭代次数

# 初始化最佳指标
best_auc = 0
best_params = None

# 初始化 KFold 进行交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=seed)

# 准备存储结果
results = []

# 迭代所有参数组合
for leaves, depth, lr, estimators in product(num_leaves, max_depth, learning_rate, n_estimators):
    # 创建 LightGBM 模型
    lgbm_model = LGBMClassifier(
        num_leaves=leaves,
        max_depth=depth,
        learning_rate=lr,
        n_estimators=estimators,
        random_state=seed,
        verbose=-1,
        n_jobs=-1
    )
    
    # 执行交叉验证以计算指标
    accuracy_scores = cross_val_score(lgbm_model, X_train, y_train, cv=kf, scoring='accuracy', n_jobs=-1)
    f1_scores = cross_val_score(lgbm_model, X_train, y_train, cv=kf, scoring='f1', n_jobs=-1)
    auc_scores = cross_val_score(lgbm_model, X_train, y_train, cv=kf, scoring='roc_auc', n_jobs=-1)

    average_accuracy = accuracy_scores.mean()
    average_f1 = f1_scores.mean()
    average_auc = auc_scores.mean()

    # 用整个训练集训练模型进行评估
    lgbm_model.fit(X_train, y_train)

    # 在测试集上进行测试
    y_pred = lgbm_model.predict(X_test)
    y_proba = lgbm_model.predict_proba(X_test)[:, 1]  # 预测概率

    # 计算测试指标
    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    test_auc = roc_auc_score(y_test, y_proba)

    # 保存交叉验证和测试结果
    result_line = (f'Params: (num_leaves={leaves}, max_depth={depth}, learning_rate={lr}, '
                   f'n_estimators={estimators}) -> '
                   f'Cross-Validated Accuracy: {average_accuracy:.4f}, '
                   f'Cross-Validated F1: {average_f1:.4f}, '
                   f'Cross-Validated AUC: {average_auc:.4f} | '
                   f'Test Accuracy: {test_accuracy:.4f}, Test Precision: {test_precision:.4f}, '
                   f'Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}, Test AUC: {test_auc:.4f}')
    
    results.append(result_line)
    print(result_line)  # 输出每次参数组合的结果

    # 记录最佳参数
    if average_auc > best_auc:
        best_auc = average_auc
        best_params = (leaves, depth, lr, estimators)

# 准备最佳参数和AUC的输出
best_result = (f'Best Parameters: (num_leaves={best_params[0]}, max_depth={best_params[1]}, '
               f'learning_rate={best_params[2]}, n_estimators={best_params[3]}) -> '
               f'Best Cross-Validated AUC: {best_auc:.4f}')
print(best_result)

# 保存结果到文件
result_folder = "result_cross"
os.makedirs(result_folder, exist_ok=True)  # 创建文件夹（如果不存在）
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # 获取当前时间
result_file_path = os.path.join(result_folder, f"LightGBM_{current_time}.txt")

with open(result_file_path, 'w') as f:
    for line in results:
        f.write(line + '\n')
    f.write(best_result + '\n')  # 保存最佳结果
