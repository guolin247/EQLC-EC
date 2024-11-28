import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 基础文件夹路径（可根据需要修改）
base_folder_path = r"D:\work_GuoLin\machine_learning\machinelearning\src\guolin"

# 相对路径 - 仅需要相对于 base_folder_path 的路径
relative_folder_path = r"CNN_CHD\test\result"

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
        file_path = os.path.join(folder_path, file_name)  # 使用 os.path.join 组合路径
        data = pd.read_csv(file_path)
        
        # 收集每个文件的准确率
        accuracy_data.append(data['Accuracy'].values)
        
        # 从文件名中提取算法名称，假设算法名称和其他信息用下划线分隔
        algorithm_name = file_name.split('_')[0]  # 取文件名的第一个部分作为算法名称
        algorithm_names.append(algorithm_name)

# 定义所需算法的固定顺序
desired_order = ['RandomForest', 'SVM', 'XGBoost', 'EQLC', 'SoftVoting', 'HardVoting']

# 根据所需顺序重新排列算法名称和准确率
ordered_acc_data = []
ordered_algorithm_names = []

for name in desired_order:
    if name in algorithm_names:
        index = algorithm_names.index(name)
        ordered_algorithm_names.append(algorithm_names[index])
        ordered_acc_data.append(accuracy_data[index])

# 绘制准确率的箱线图
fig, ax = plt.subplots(figsize=(6, 4))

# 定义背景区域的颜色和范围
left_background_color = 'white'
right_background_color = 'lightpink'
boundary_index = 3  # 定义分割点，例如前四个模型与后面两个模型的背景不同

# 绘制背景颜色
ax.axvspan(0.5, boundary_index + 0.5, facecolor=left_background_color, alpha=0.3)
ax.axvspan(boundary_index + 0.5, len(ordered_algorithm_names) + 0.5, facecolor=right_background_color, alpha=0.3)

# 定义每个箱体的颜色
box_colors = [
    '#0076B9',  # 深蓝色（左3）
    '#EC3E31',  # 红色（右2）
    '#DB2834'   # 深红色（右3）
]

# 绘制箱线图并设置每个箱体的颜色
box = ax.boxplot(ordered_acc_data, tick_labels=ordered_algorithm_names, patch_artist=True)

for patch, color in zip(box['boxes'], box_colors):
    patch.set_facecolor(color)

# # # # 设置从第三个箱体位置绘制一条平度的虚线
# median_value = 0.771  # 手动设置中位数值
# ax.axhline(median_value, linestyle='--', color='gray')
# # #增加阴影区域 0.9241±0.0406
# # ax.axhspan(0.9241-0.0406, 0.9241+0.0406, facecolor='gray', alpha=0.2)
# # 调整 x 轴范围，只显示 0.5 到 箱体数量 + 0.5 的部分
ax.set_xlim(0.5, len(ordered_algorithm_names) + 0.5)


# # 设置 y 轴范围，y 轴数值只显示小数点后两位,范围从0.8-1
ax.set_ylim(0.69, 0.84)
ax.yaxis.set_major_formatter('{x:.2f}')

# 设置图表的标题和坐标说明
plt.ylabel('Accuracy', fontsize=24)

# 关闭网格线
plt.grid(False)

# 旋转 x 轴标签以便于阅读
plt.xticks(rotation=30)
plt.subplots_adjust(left=0.18, right=0.99, top=0.95, bottom=0.25)
# 设置 y 轴范围和格式，格式化为两位小数
ax.yaxis.set_major_formatter('{x:.2f}')
ax.tick_params(labelsize=16)
# 在图的下方添加文本
text = 'Train Set: Batch1&4, Test Set:Batch3 '
plt.text(0.6, 0.32, text, ha='center', va='center', fontsize=16, transform=fig.transFigure)
# 显示图形
plt.show()

# 保存图片
ax.figure.savefig('accuracy_boxplot_14_3.png', dpi=300)

# 计算并打印每种算法的方差和标准差
for algorithm_name, accuracies in zip(ordered_algorithm_names, ordered_acc_data):
    variance = np.var(accuracies)
    std = np.std(accuracies)
    print(f'{algorithm_name} Variance: {variance:.8f}, Standard Deviation: {std:.4f}')
