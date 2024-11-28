import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# CSV 文件夹路径
folder_path = r"D:\work_GuoLin\machine_learning\machinelearning\src\guolin\CNN_single_HIP_CER\test\result"

# 收集算法名称和 AUC
auc_data = []
algorithm_names = []

# 遍历文件夹中的所有 CSV 文件
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        # 加载数据
        file_path = os.path.join(folder_path, file_name)
        data = pd.read_csv(file_path)

        # 收集每个文件的 AUC 值
        auc_data.append(data['AUC'].values)

        # 从文件名中提取算法名称
        algorithm_name = file_name.split('_')[0]
        algorithm_names.append(algorithm_name)

# 定义所需算法的固定顺序
desired_order = ['RandomForest', 'SVM', 'AdaBoost', 'XGBoost', '1DCNN', 'SoftVoting', 'HardVoting']

# 根据所需顺序重新排列算法名称和 AUC 数据
ordered_auc_data = []
ordered_algorithm_names = []

for name in desired_order:
    if name in algorithm_names:
        index = algorithm_names.index(name)
        ordered_algorithm_names.append(algorithm_names[index])
        ordered_auc_data.append(auc_data[index])

# 绘制 AUC 的箱线图
fig, ax = plt.subplots(figsize=(10, 6))

# 定义背景区域的颜色和范围
left_background_color = 'white'
right_background_color = 'lightpink'
boundary_index = 4

# 绘制背景颜色
ax.axvspan(0.5, boundary_index + 0.5, facecolor=left_background_color, alpha=0.3)
ax.axvspan(boundary_index + 0.5, len(ordered_algorithm_names) + 0.5, facecolor=right_background_color, alpha=0.3)

# 定义每个箱体的颜色
box_colors = [
    'skyblue',       # 浅蓝色（左1）
    'lightgreen',    # 浅绿色（左2）
    'lightgray',     # 浅灰色（左3）
    'None',          # 不填充颜色（左4）
    'lightpink',     # 深粉色（右1）
    'seagreen',      # 深绿色（右2）
    'dodgerblue'     # 深蓝色（右3）
]

# 绘制箱线图并设置每个箱体的颜色
box = ax.boxplot(ordered_auc_data, labels=ordered_algorithm_names, patch_artist=True)

for patch, color in zip(box['boxes'], box_colors):
    patch.set_facecolor(color)

# 设置从第四个箱体位置绘制一条平度的虚线
median_value = np.median(ordered_auc_data[3])
ax.axhline(median_value, linestyle='--', color='gray')

# 设置轴范围和格式
ax.set_xlim(0.5, len(ordered_algorithm_names) + 0.5)
ax.yaxis.set_major_formatter('{x:.2f}')

# 设置图表标题
plt.ylabel('AUC', fontsize=12)
plt.grid(False)
plt.xticks(rotation=30)
plt.subplots_adjust(bottom=0.13)

# 显示图形
plt.show()

# 保存图片
ax.figure.savefig('auc_boxplot.png', dpi=300)

# 计算并打印每种算法的方差和标准差
for algorithm_name, aucs in zip(ordered_algorithm_names, ordered_auc_data):
    variance = np.var(aucs)
    std = np.std(aucs)
    print(f'{algorithm_name} Variance: {variance:.8f}, Standard Deviation: {std:.4f}')
