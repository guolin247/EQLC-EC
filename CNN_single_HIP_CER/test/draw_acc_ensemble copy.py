import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 基础文件夹路径（可根据需要修改）
base_folder_path = r"D:\work_GuoLin\machine_learning\machinelearning\src\guolin"

# 相对路径 - 仅需要相对于 base_folder_path 的路径
relative_folder_path = r"CNN_single_HIP_CER\test\result"

# 合并路径以形成完整的 CSV 文件夹路径
folder_path = os.path.join(base_folder_path, relative_folder_path)

# 收集算法名称和准确率
accuracy_data = []
algorithm_names = []

# 遍历文件夹中的所有 CSV 文件
for file_name in os.listdir(folder_path):
    # 检查文件是否为 CSV 格式
    if file_name.endswith('.csv'):
        # 加载数据
        file_path = os.path.join(folder_path, file_name)
        data = pd.read_csv(file_path)

        # 收集每个文件的准确率
        accuracy_data.append(data['Accuracy'].values)

        # 从文件名中提取算法名称，假设算法名称和其他信息用下划线分隔
        algorithm_name = file_name.split('_')[0]
        algorithm_names.append(algorithm_name)

# 定义所需算法的固定顺序
desired_order = ['RandomForest', 'SVM', 'XGBoost', '1DCNN', 'SoftVoting', 'HardVoting']

# 根据所需顺序重新排列算法名称和准确率
ordered_acc_data = []
ordered_algorithm_names = []

for name in desired_order:
    if name in algorithm_names:
        index = algorithm_names.index(name)
        ordered_algorithm_names.append(algorithm_names[index])
        ordered_acc_data.append(accuracy_data[index])

# 绘制准确率的小提琴图
fig, ax = plt.subplots(figsize=(10, 6))

# 设置每种小提琴的最大宽度
violin_width = 0.5

# 绘制小提琴图并设置宽度
violin_parts = ax.violinplot(ordered_acc_data, 
                             showmeans=False, 
                             showextrema=False, 
                             showmedians=True, 
                             widths=violin_width)

# 为每个小提琴图设置颜色
colors = [
    'skyblue',       # 浅蓝色（RandomForest）
    'lightgreen',    # 浅绿色（SVM）
    'whitesmoke',    # 白烟色（XGBoost，保持透明）
    'lightpink',     # 浅粉色（1DCNN）
    'seagreen',      # 深绿色（SoftVoting）
    'dodgerblue'     # 深蓝色（HardVoting）
]

for i, violin in enumerate(violin_parts['bodies']):
    violin.set_facecolor(colors[i])
    violin.set_alpha(0.6)

# 定义背景区域的颜色和范围
ax.axvspan(0.5, 3.5, facecolor='white', alpha=0.3)
ax.axvspan(3.5, len(ordered_algorithm_names) + 0.5, facecolor='lightpink', alpha=0.3)

# 设置从第三个算法位置绘制一条平度的虚线
median_value = np.median(ordered_acc_data[2])  # 计算第三个算法的中位数
ax.axhline(median_value, linestyle='--', color='gray')

# 调整 x 轴范围，只显示 0.5 到 箱体数量 + 0.5 的部分
ax.set_xlim(0.5, len(ordered_algorithm_names) + 0.5)

# 设置 y 轴范围，y 轴数值只显示小数点后两位
ax.yaxis.set_major_formatter('{x:.2f}')

# 设置图表的标题和坐标说明
plt.ylabel('Accuracy', fontsize=12)

# 关闭网格线
plt.grid(False)

# 旋转 x 轴标签以便于阅读
plt.xticks(ticks=range(1, len(ordered_algorithm_names) + 1), labels=ordered_algorithm_names, rotation=30)
plt.subplots_adjust(bottom=0.13)

# 显示图形
plt.show()

# 保存图片
ax.figure.savefig('accuracy_violinplot.png', dpi=300)

# 计算并打印每种算法的方差和标准差
for algorithm_name, accuracies in zip(ordered_algorithm_names, ordered_acc_data):
    variance = np.var(accuracies)
    std = np.std(accuracies)
    print(f'{algorithm_name} Variance: {variance:.8f}, Standard Deviation: {std:.4f}')
