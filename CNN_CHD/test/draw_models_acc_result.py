import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 定义包含子文件夹的目录路径
directory_path = r'D:\work_GuoLin\machine_learning\machinelearning\src\guolin\CNN_CHD\test\result'

# 初始化一个用于存放绘图数据的字典
data = {}

# 遍历目录下的每个子文件夹
for subfolder in os.listdir(directory_path):
    subfolder_path = os.path.join(directory_path, subfolder)

    # 检查是否是目录
    if os.path.isdir(subfolder_path):
        # 查找子文件夹中的CSV文件
        for file in os.listdir(subfolder_path):
            if 'ModelPerformance' in file and file.endswith('.csv'):
                filepath = os.path.join(subfolder_path, file)
                # 读取CSV文件
                df = pd.read_csv(filepath)
                # 假设第一列是算法名称，第二列是准确率
                for _, row in df.iterrows():
                    model_name = str(row.iloc[0])  # Ensure model_name is a string
                    accuracy = row.iloc[1]  # Use iloc for positional indexing
                    
                    # 在判断之前，处理 model_name 的格式
                    if '_' in model_name:
                        # 提取符合指定格式的部分
                        model_name = '_'.join(model_name.split('_')[:-1])  # 去掉最后一个 underscore 之后的部分
                    # 仅保留 Papers，Hard Voting，Soft Voting，ComBat
                    # if model_name not in {'ComBat', 'Hard Voting', 'Soft Voting'}:
                    #     continue
                    
                    if model_name not in data:
                        data[model_name] = []
                    data[model_name].append(accuracy)

# 计算平均值和标准差
statistics = {}
for model_name, accuracies in data.items():
    average_accuracy = sum(accuracies) / len(accuracies)
    std_deviation = np.std(accuracies)
    statistics[model_name] = (average_accuracy, std_deviation)
    print(f'{model_name}: Average Accuracy = {average_accuracy:.8f}, Std Deviation = {std_deviation:.8f}')

# 将结果保存到CSV文件
statistics_df = pd.DataFrame(statistics).T
statistics_df.columns = ['Average Accuracy', 'Std Deviation']
statistics_df.index.name = 'Model Name'
statistics_df.to_csv('model_performance_statistics.csv')  # 保存为CSV文件

# 绘图
plt.figure(figsize=(10, 4))

# 设置字体大小
axis_label_fontsize = 18
axis_ticks_fontsize = 16
legend_fontsize = 12

# 遍历数据字典，绘制每个算法的准确率
for model_name, accuracies in data.items():
    # 获取对应的文件夹名称
    subfolders = [subfolder for subfolder in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, subfolder))]
    
    # 确保绘制时 x 轴上各个算法对应的子文件夹数量、顺序一致，也解决绘图点上的子文件夹名称问题
    accuracies_by_subfolder = [accuracies[subfolders.index(subfolder)] if subfolder in subfolders else float('nan') for subfolder in subfolders]
    
    # 区分 ComBat 的标记符号
    marker = 's' if model_name == 'ComBat' else 'o'
    
    plt.plot(subfolders, accuracies_by_subfolder, marker=marker, label=model_name)

# 添加标题和标签
plt.title('Model Accuracy', fontsize=axis_label_fontsize)
plt.xlabel('12_3 means Batch1 and Batch2 are training set, Batch3 is test set', fontsize=axis_label_fontsize, color='red')
plt.ylabel('Accuracy', fontsize=axis_label_fontsize)

# 设置刻度标签的字体大小和颜色
plt.xticks(rotation=36, fontsize=axis_ticks_fontsize)
plt.yticks(fontsize=axis_ticks_fontsize)

# 设置图例的字体大小
plt.legend(title='Model Name', fontsize=legend_fontsize, title_fontsize=legend_fontsize + 2)

# 去掉背景网格
plt.grid(False)

# 调整布局
plt.tight_layout()

# 显示图表
plt.show()
