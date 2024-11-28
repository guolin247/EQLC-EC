'''
Best Parameters: (n_estimators=200, learning_rate=1.0, max_depth=1) -> Best Cross-Validated AUC: 0.7722
'''

import os
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
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
n_estimators = [50, 100, 200]  # 基础分类器数量
learning_rate = [0.01, 0.1, 1.0]  # 学习率
max_depth = [1, 2, 3]  # 基础分类器的最大深度

# 初始化最佳指标
best_auc = 0
best_params = None

# 初始化 KFold 进行交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=seed)

# 准备存储结果
results = []

# 迭代所有参数组合
for n, lr, depth in product(n_estimators, learning_rate, max_depth):
    # 创建 AdaBoost 模型，基础分类器为决策树
    base_estimator = DecisionTreeClassifier(max_depth=depth, random_state=seed)
    ada_model = AdaBoostClassifier(
        n_estimators=n,
        learning_rate=lr,
        random_state=seed,
        algorithm='SAMME'  # 显式选择 SAMME 算法
    )
    
    # 执行交叉验证以计算指标
    accuracy_scores = cross_val_score(ada_model, X_train, y_train, cv=kf, scoring='accuracy', n_jobs=-1)
    f1_scores = cross_val_score(ada_model, X_train, y_train, cv=kf, scoring='f1', n_jobs=-1)
    auc_scores = cross_val_score(ada_model, X_train, y_train, cv=kf, scoring='roc_auc', n_jobs=-1)

    average_accuracy = accuracy_scores.mean()
    average_f1 = f1_scores.mean()
    average_auc = auc_scores.mean()

    # 用整个训练集训练模型进行评估
    ada_model.fit(X_train, y_train)

    # 在测试集上进行测试
    y_pred = ada_model.predict(X_test)
    y_proba = ada_model.predict_proba(X_test)[:, 1]  # 预测概率

    # 计算测试指标
    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    test_auc = roc_auc_score(y_test, y_proba)

    # 保存交叉验证和测试结果
    result_line = (f'Params: (n_estimators={n}, learning_rate={lr}, max_depth={depth}) -> '
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
        best_params = (n, lr, depth)

# 准备最佳参数和AUC的输出
best_result = (f'Best Parameters: (n_estimators={best_params[0]}, learning_rate={best_params[1]}, '
               f'max_depth={best_params[2]}) -> '
               f'Best Cross-Validated AUC: {best_auc:.4f}')
print(best_result)

# 保存结果到文件
result_folder = "result_cross"
os.makedirs(result_folder, exist_ok=True)  # 创建文件夹（如果不存在）
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # 获取当前时间
result_file_path = os.path.join(result_folder, f"AdaBoost_{current_time}.txt")

with open(result_file_path, 'w') as f:
    for line in results:
        f.write(line + '\n')
    f.write(best_result + '\n')  # 保存最佳结果
