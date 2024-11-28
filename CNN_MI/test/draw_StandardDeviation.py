import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle

# 读取Excel文件
file_path = r"D:\work_GuoLin\machine_learning\machinelearning\src\guolin\CNN_MI\test\Standard_Deviation.xlsx"
xls = pd.ExcelFile(file_path)

# 定义一个函数来读取每个sheet的数据
def read_data(sheet_name):
    df = pd.read_excel(xls, sheet_name=sheet_name)
    return df

# 获取所有的sheet名称
sheet_names = xls.sheet_names

# 创建空字典用于存储Accuracy和F1数据
acc_data = {}
f1_data = {}
labels = []

# 从每个sheet读取数据
for sheet in sheet_names:
    df = read_data(sheet)
    acc_data[sheet] = df['Accuracy'].values
    f1_data[sheet] = df['F1_Score'].values
    labels = df['distribution'].values  # 假设第一列是所需的标签

# 分割标签，使其换行
split_labels = [label.replace(' ', '\n') for label in labels]

# 颜色选择（使用英文颜色名称）
colors = ['#6BBC47', '#EC3E31', '#0076B9']

# 颜色选择（使用英文颜色名称）
line_color = 'gray'
marker_color = 'black'
marker_size = 100  # 设置圆点和叉号的大小

# 绘图函数，包含背景色和平均值虚线
def plot_data(data, y_label, y_lim=(0.7, 0.9)):
    fig, ax = plt.subplots(figsize=(18, 9))
    width = 0.05  # 每个柱子的宽度
    spacing = 0.005  # 每个柱子之间的间距
    group_spacing = 0.1  # 每个组之间的间距
    x_labels = list(data.keys())
    x = np.arange(len(x_labels)) * (3 * width + 2 * spacing + group_spacing)  # 延长每组总长度，使各组有间距

    # 确定 hardvoting 的索引以设置背景色
    hardvoting_index = [i for i, sheet in enumerate(x_labels) if 'hardvoting' in sheet.lower()]

    # 添加背景色区分硬投票区域
    for idx in hardvoting_index:
        rect = Rectangle((x[idx] - (3 * width + 2 * spacing + group_spacing) / 2, 0.7),
                         3 * width + 2 * spacing + group_spacing, 0.9 - 0.7, color='lightgray', alpha=0.2)
        ax.add_patch(rect)

    # 绘制每个sheet的数值
    for i, sheet in enumerate(x_labels):
        for j, label in enumerate(labels):
            color = colors[j % len(colors)]  # 为每个label分配颜色
            ax.bar(x[i] - (3 * width + 2 * spacing) / 2 + j * (width + spacing) + width / 2, data[sheet][j], width=width, color=color, alpha=1)

    # 计算第一组的平均值并绘制横跨全图的虚线
    avg_value_first_group = np.mean(data[x_labels[0]])
    # 调整虚线的长度，延长到接近边缘
    ax.plot([x[0] - (3 * width + 2 * spacing + group_spacing) / 2, x[-1] + (3 * width + 2 * spacing + group_spacing) / 2],
            [avg_value_first_group, avg_value_first_group], color=line_color, linestyle='--', label='Average of paper')

    # 计算后两组的平均值并用不同符号标记
    avg_hard_voting = np.mean(data[x_labels[1]])  # 第二个sheet的平均值
    avg_soft_voting = np.mean(data[x_labels[2]])  # 第三个sheet的平均值
    
    ax.scatter(x[1], avg_hard_voting, color=marker_color, marker='o', s=marker_size, label='Average of HardVoting')  # 第二个sheet绘制圆点
    ax.scatter(x[2], avg_soft_voting, color=marker_color, marker='X', s=marker_size, label='Average of SoftVoting')  # 第三个sheet绘制叉号

    ax.set_ylim(y_lim)  # 设置y轴范围
    ax.set_xticks(x)  # 设置x轴刻度
    ax.set_xticklabels(x_labels, fontsize=24)  # 调整x轴刻度字体大小
    ax.set_xlabel('Source of Model', fontsize=24)  # 调整x轴标题字体大小
    ax.set_ylabel(y_label, fontsize=24)  # 调整y轴标题字体大小
    ax.set_title(f'{y_label} Comparison by Sheets', fontsize=24)  # 调整图标题字体大小

    # 调整y轴刻度数字的字体大小
    ax.tick_params(axis='y', labelsize=20)  # 设置y轴刻度标签字体大小

    # 创建图例，将换行后的标签显示在图例中
    legend_elements = [Patch(facecolor=colors[i % len(colors)], edgecolor='black', label=split_labels[i]) for i in range(len(labels))]
    legend_elements.append(plt.Line2D([0], [0], color=line_color, linestyle='--', label='Average of paper'))
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label='Average of HardVoting', markerfacecolor=marker_color, markersize=marker_size//20))  # 增大大小
    legend_elements.append(plt.Line2D([0], [0], marker='X', color='w', label='Average of SoftVoting', markerfacecolor=marker_color, markersize=marker_size//10))  # 增大大小
    
    # 调整图例字体大小
    ax.legend(handles=legend_elements, title="Source-Target", loc='upper left', fontsize=16, title_fontsize=16)

    plt.tight_layout()
    plt.show()

# 绘制Accuracy柱状图，手动设置y轴范围
plot_data(acc_data, 'Accuracy', y_lim=(0.75, 0.90))

# 绘制F1 Score柱状图，手动设置y轴范围
plot_data(f1_data, 'F1 Score', y_lim=(0.80, 0.90))
