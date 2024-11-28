import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# CSV 文件夹路径
folder_path = r"D:\work_GuoLin\machine_learning\machinelearning\src\guolin\CNN_single_HIP_CER\test\result"

# 收集算法名称和 F1 分数
f1_score_data = []
algorithm_names = []

# 遍历文件夹中的所有 CSV 文件
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        # 加载数据
        file_path = os.path.join(folder_path, file_name)
        data = pd.read_csv(file_path)
        
        # 收集每个文件的 F1 分数
        f1_score_data.append(data['F1 Score'].values)

        # 从文件名中提取算法名称
        algorithm_name = file_name.split('_')[0]
        algorithm_names.append(algorithm_name)

# 定义所需算法的固定顺序
desired_order = ['RandomForest', 'SVM', 'XGBoost', 'EQLC', 'SoftVoting', 'HardVoting']

# 根据所需顺序重新排列算法名称和 F1 分数
ordered_f1_data = []
ordered_algorithm_names = []

for name in desired_order:
    if name in algorithm_names:
        index = algorithm_names.index(name)
        ordered_algorithm_names.append(algorithm_names[index])
        ordered_f1_data.append(f1_score_data[index])

# 绘制 F1 分数的箱线图
fig, ax = plt.subplots(figsize=(10, 5))  # 设置图表大小

# 定义背景区域的颜色和范围
left_background_color = 'none'            # 左侧背景颜色
right_background_color = '#F9B3AD'       # 右侧背景颜色
boundary_index = 3                        # 定义分割点

# 绘制背景颜色，对应不同的图表区域
ax.axvspan(0.5, boundary_index + 0.5, facecolor=left_background_color, alpha=0.3)
ax.axvspan(boundary_index + 0.5, len(ordered_algorithm_names) + 0.5, facecolor=right_background_color, alpha=0.3)

# 定义每个箱体的颜色
box_colors = [
    '#C79DC9',  # 浅紫色（左1）
    '#C2B4D3',  # 浅蓝色（左2）
    '#0076B9',  # 深蓝色（左4）
    '#EF607A',  # 深粉色（右1）
    '#EC3E31',  # 红色（右2）
    '#DB2834'   # 深红色（右3）
]

# 绘制箱线图并设置每个箱体的颜色
box = ax.boxplot(ordered_f1_data, labels=ordered_algorithm_names, patch_artist=True)
for patch, color in zip(box['boxes'], box_colors):
    patch.set_facecolor(color)

# 设置从第四个箱体位置绘制一条水平的虚线
median_value = np.median(ordered_f1_data[3])  # 计算第四个箱体的中位数
ax.axhline(median_value, linestyle='--', color='gray')

# 调整 x 轴范围
ax.set_xlim(0.5, len(ordered_algorithm_names) + 0.5)

# 设置 y 轴范围和格式，格式化为两位小数
ax.yaxis.set_major_formatter('{x:.2f}')
ax.tick_params(labelsize=18)

# 设置图表的标题和坐标说明
plt.ylabel('F1 Score', fontsize=24)

# 关闭网格线
plt.grid(False)

# 旋转 x 轴标签
plt.xticks(rotation=30, fontsize=18)
plt.subplots_adjust(left=0.11, right=0.99, top=0.99, bottom=0.25)

# 自定义 x 轴标签颜色
for label in ax.get_xticklabels():
    if label.get_text() == 'XGBoost':
        label.set_color('#0076B9')  # 深蓝色
    elif label.get_text() == 'SoftVoting':
        label.set_color('#EC3E31')  # 红色
    elif label.get_text() == 'HardVoting':
        label.set_color('#EC3E31')  # 红色

# 显示图形
plt.show()

# 保存图片
ax.figure.savefig('f1_score_boxplot.png', dpi=300)

# 计算并打印每种算法的方差和标准差
for algorithm_name, f1_scores in zip(ordered_algorithm_names, ordered_f1_data):
    variance = np.var(f1_scores)
    std = np.std(f1_scores)
    print(f'{algorithm_name} 方差: {variance:.8f}, 标准差: {std:.4f}')
