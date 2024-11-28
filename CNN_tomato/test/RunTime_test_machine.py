import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

# 定义数据集类
class NumpyDataset:
    def __init__(self, data_path, labels_path):
        self.data = np.load(data_path)  # 加载 numpy 数据
        self.labels = np.load(labels_path)  # 加载标签
    
    def get_data(self):
        return self.data, self.labels

# 数据路径
train_data_path = 'D:\\work_GuoLin\\machine_learning\\machinelearning\\src\\guolin\\CNN_TOMATO\\stratified_split_data\\X_train.npy'
train_labels_path = 'D:\\work_GuoLin\\machine_learning\\machinelearning\\src\\guolin\\CNN_TOMATO\\stratified_split_data\\y_train.npy'
test_data_path = 'D:\\work_GuoLin\\machine_learning\\machinelearning\\src\\guolin\\CNN_TOMATO\\stratified_split_data\\X_test.npy'
test_labels_path = 'D:\\work_GuoLin\\machine_learning\\machinelearning\\src\\guolin\\CNN_TOMATO\\stratified_split_data\\y_test.npy'

# 加载数据集
train_dataset = NumpyDataset(train_data_path, train_labels_path)
test_dataset = NumpyDataset(test_data_path, test_labels_path)

# 获取训练和测试数据
X_train, y_train = train_dataset.get_data()
X_test, y_test = test_dataset.get_data()

# 检查并调整数据维度
if X_train.ndim > 2:
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

# 模型参数设置
seed = 42
models = {
    'RandomForest': RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=5, min_samples_leaf=2, max_features='sqrt', random_state=seed),
    'SVM': SVC(C=10, gamma=0.01, kernel='rbf', random_state=seed, probability=True, cache_size=2000),
    'AdaBoost': AdaBoostClassifier(n_estimators=200, learning_rate=1.0, random_state=seed, algorithm='SAMME'),
    'XGBoost': XGBClassifier(n_estimators=500, learning_rate=0.1, random_state=seed, eval_metric='logloss')
}

# 5折交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

# 初始化结果存储
results = []

# 运行时间测试
for model_name, model in models.items():
    print(f"Running {model_name} model...")
    train_times = []
    predict_times = []
    for i in range(11):  # 重复11次，第1次为预热
        fold_train_times = []
        fold_predict_times = []
        for train_idx, test_idx in cv.split(X_train, y_train):
            X_train_cv, X_val_cv = X_train[train_idx], X_train[test_idx]
            y_train_cv, y_val_cv = y_train[train_idx], y_train[test_idx]
            
            # 训练时间测量
            start_train = time.time()
            model.fit(X_train_cv, y_train_cv)
            end_train = time.time()
            fold_train_times.append(end_train - start_train)
            
            # 预测时间测量
            start_predict = time.time()
            y_pred = model.predict(X_val_cv)
            end_predict = time.time()
            fold_predict_times.append(end_predict - start_predict)
        
        # 记录5折的平均训练和预测时间
        if i > 0:  # 跳过预热的第1次结果
            train_times.append(np.mean(fold_train_times))
            predict_times.append(np.mean(fold_predict_times))
        
        print(f"Iteration {i + 1} complete for {model_name}")

    # 计算每个模型的平均训练时间和预测时间（从第2次到第11次的结果）
    avg_train_time = np.mean(train_times)
    avg_predict_time = np.mean(predict_times)
    
    results.append({
        'Model': model_name,
        'Train_Time_Mean': avg_train_time,
        'Train_Time_Std': np.std(train_times),
        'Predict_Time_Mean': avg_predict_time,
        'Predict_Time_Std': np.std(predict_times),
        'All_Train_Times': train_times,
        'All_Predict_Times': predict_times
    })

# 将结果保存为CSV文件
df_results = pd.DataFrame(results)
df_results.to_csv('model_runtime_results.csv', index=False)
print("Results saved to model_runtime_results.csv")
