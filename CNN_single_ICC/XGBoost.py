import os
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

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

# 按需修改数据形状，XGBoost 需要二维数组
# 如果 X_train 原本是形式 (样本数, 特征数, 1)，通过 reshape 转换为 (样本数, 特征数)
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# 训练 XGBoost 模型，设置 n_estimators 和 learning_rate
model = xgb.XGBClassifier(n_estimators=500, learning_rate=0.1, random_state=seed, eval_metric='logloss')

# 训练模型
model.fit(X_train, y_train)

# 评估模型
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # 预测概率

# 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
auc = roc_auc_score(y_test, y_proba)

print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}')
