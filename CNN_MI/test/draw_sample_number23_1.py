import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# 指定包含标签数据的路径
labels_path = r'D:\work_GuoLin\machine_learning\machinelearning\src\guolin\CNN_MI\stratified_split_data'

# 文件名列表
file_names = ['y_1.npy', 'y_2.npy', 'y_3.npy']

# 设定每个 Batch 的类别（手动指定）
batch_types = ['Training Set', 'Training Set', 'Testing Set']  # Batch 1 和 2 为训练集，Batch 3 为测试集

# 准备绘图数据
batch_counts = []
batch_names = []

# 读取每个文件并统计标签
for file_name in file_names:
    file_path = f"{labels_path}\\{file_name}"
    labels = np.load(file_path, allow_pickle=True)
    
    # 检查标签数组的维度
    if labels.ndim == 1:
        labels_only = labels  # 直接使用一维数组
    elif labels.ndim == 2:
        labels_only = labels[:, 1]  # 提取每行的第二个元素
    else:
        raise ValueError(f"Unexpected number of dimensions: {labels.ndim}")

    # 统计标签数量
    label_counts = Counter(labels_only)
    
    # 计算样本数量并添加到列表
    total_count = sum(label_counts.values())
    batch_counts.append(total_count)

    # 添加对应的 Batch 名称
    batch_names.append(f'Batch {len(batch_names) + 1}')  # 构建 Batch 名称

# 总样本数
total_samples = sum(batch_counts)

# 创建绘图
fig, ax = plt.subplots(figsize=(8, 8))  # 调整图形大小

# 设置绘图数据
colors = ['#0076B9', '#EC3E3127', '#6BBC4727']  # 训练集使用不同的颜色

# 计算百分比
batch_percentages = [count / total_samples * 100 for count in batch_counts]

# 绘制外部环形饼图，并设置文字距离中心的距离
wedges, texts, autotexts = ax.pie(
    batch_counts,
    startangle=90,
    wedgeprops=dict(width=0.5, edgecolor='w'),
    colors=colors,
    autopct='',  # 不显示默认的百分比
    pctdistance=0.75  # 调整此值以设置文字距离中心的距离（0.85 只是示例值，可根据需要调整）
)

# 在中央放置总样本数
# ax.text(0, 0, 'Training Set-Batch1,2\nTesting Set-Batch3', ha='center', va='center', fontsize=24, bbox=dict(facecolor='white', alpha=0.8))

# 设置绘图细节
ax.set(aspect="equal")
# # 手动设置图例的颜色和标签
handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#0076B9', markersize=10, label='Testing Set'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#EC3E3127', markersize=10, label='Training Set'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#6BBC4727', markersize=10, label='Training Set')
]

# 添加图例，手动指定图例颜色和标签，并调整Testing Set字体颜色为红色
legend = ax.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.5), 
                   frameon=True, framealpha=0.8, fontsize=12, borderpad=1, labelspacing=1)
# 修改Testing Set的字体颜色为红色
for text in legend.get_texts():
    if text.get_text() == 'Testing Set':
        text.set_color("red")

# 更新环形每一区域的文本为 Batch 名称及其百分比
for i, a in enumerate(autotexts):
    a.set_text(f'{batch_names[i]}\n{batch_percentages[i]:.2f}%')
    a.set_fontsize(18)  # 调整文本大小

plt.tight_layout()  # 调整布局
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 手动调整子图位置

plt.show()

# 如果需要保存图片，取消下面这行的注释
# plt.savefig('datasets_distribution.png', dpi=300, bbox_inches='tight')