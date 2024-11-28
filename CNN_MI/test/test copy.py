import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# CSV 文件夹路径
folder_path = r"D:\work_GuoLin\machine_learning\machinelearning\src\guolin\CNN_single_HIP_CER\test\result"

# 收集算法名称和准确率
accuracy_data = []
algorithm_names = []

# 遍历文件夹中的所有 CSV 文件
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        # 加载数据
        file_path = os.path.join(folder_path, file_name)
        data = pd.read_csv(file_path)
        
        # 收集每个文件的准确率
        accuracy_data.append(data['Accuracy'].values)
        
        # 从文件名中提取算法名称
        algorithm_name = file_name.split('_')[0]
        algorithm_names.append(algorithm_name)

# 计算每个算法准确率的中位数
median_accuracies = [np.median(acc) for acc in accuracy_data]

# 根据中位数进行排序
sorted_indices = np.argsort(median_accuracies)
sorted_accuracy_data = [accuracy_data[i] for i in sorted_indices]
sorted_algorithm_names = [algorithm_names[i] for i in sorted_indices]

# 绘制准确率的箱线图
plt.figure(figsize=(10, 6))
plt.boxplot(sorted_accuracy_data, labels=sorted_algorithm_names)
plt.title('Boxplot of Accuracy across Models (Sorted by Median)')
plt.ylabel('Accuracy')
plt.xlabel('Models')
plt.grid(axis='y')
plt.xticks(rotation=45)  # 旋转 x 轴标签以便于阅读
plt.show()

# 计算并打印每种算法的方差
for algorithm_name, accuracies in zip(sorted_algorithm_names, sorted_accuracy_data):
    variance = np.var(accuracies)
    print(f'{algorithm_name} Variance: {variance:.8f}')
# 计算并打印每种算法的标准差
for algorithm_name, accuracies in zip(sorted_algorithm_names, sorted_accuracy_data):
    std = np.std(accuracies)
    print(f'{algorithm_name} Standard Deviation: {std:.4f}')
