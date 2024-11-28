import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# 加载标签数据
labels_path = 'D:\\work_GuoLin\\machine_learning\\machinelearning\\src\\guolin\\CNN_single_HIP_CER\\output_for_cnn\\labels.npy'
labels = np.load(labels_path)

# 提取标签信息并计数
labels_only = labels[:, 1]
label_counts = Counter(labels_only)

# 准备绘图数据
labels_list = list(label_counts.keys())
counts_list = list(map(int, label_counts.values()))  # 将计数转换为整数
total_samples = sum(counts_list)

# 绘图
fig, ax = plt.subplots(figsize=(12, 6), ncols=2)

# 发现数据集的饼图
ax[0].pie(
    counts_list, 
    labels=counts_list, 
    startangle=90, 
    wedgeprops=dict(width=0.3, edgecolor='w'), 
    textprops={'fontsize': 48}, 
    colors=['#0076B9', '#EC3E31']
)
ax[0].set(aspect="equal", title='Complete Dataset')
# 中央文字
ax[0].text(0, 0, f'N={total_samples}', ha='center', va='center', fontsize=48)
# 标题字体大小
ax[0].title.set_fontsize(36)

# 独立测试数据集的饼图，假设测试集是发现集的20%
subset_counts = [int(count * 0.2) for count in counts_list]  # 假设测试集占20%
# 使用 explode 增加白色填充，以缩小右侧饼图的实际显示
ax[1].pie(
    subset_counts, 
    labels=subset_counts, 
    startangle=90, 
    wedgeprops=dict(width=0.15, edgecolor='w'),  # 减少楔块宽度
    textprops={'fontsize': 48}, 
    colors=['#0076B9', '#EC3E31'],
    radius=0.6  # 缩小右侧饼图的半径
)
ax[1].set(aspect="equal", title='Independent Testing Dataset')
# 中央文字
ax[1].text(0, 0, f'N={sum(subset_counts)}', ha='center', va='center', fontsize=48)
# 标题字体大小
ax[1].title.set_fontsize(36)

# 图例，保持两张图相同的位置和设置
handles, legend_labels = ax[0].get_legend_handles_labels()
labels = ['Cerebellar', 'Hippocampal']
ax[0].legend(handles, labels, loc="upper right", fontsize=24, bbox_to_anchor=(1.2, 1))
ax[1].legend(handles, labels, loc="upper right", fontsize=24, bbox_to_anchor=(1.2, 1))

# 在图的下方添加文本
text = 'Number of features per sample: '
plt.text(0.5, 0.12, text, ha='center', va='center', fontsize=48, transform=fig.transFigure)
plt.text(0.82, 0.12, '2464', ha='center', va='center', fontsize=48, color='red', transform=fig.transFigure)

# 调整布局使子图更对齐
# fig.subplots_adjust(wspace=0.5)

plt.show()
