import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# 标签文件路径
label_path = 'D:\\work_GuoLin\\machine_learning\\machinelearning\\src\\guolin\\CNN_MI\\stratified_split_data\\y_3.npy'

# 读取标签并计数
labels = np.load(label_path)
label_counts = Counter(labels)

# 准备绘图数据
fig, ax = plt.subplots(figsize=(8, 8))

title = 'Batch 3'

# 为数据集设置颜色
# colors = ['#FCC41E54', '#FCC41E']
colors = ['#6BBC4754', '#6BBC47']

# 为数据集设置标签
group_labels = ['Control', 'MI']

counts_list = [label_counts.get(0, 0), label_counts.get(1, 0)]  # 0对应未患病的对照组，1对应心肌梗塞病人
total_samples = sum(counts_list)

# 计算每个标签的比例
percentages = [f"{count / total_samples * 100:.1f}%" for count in counts_list]

wedges, texts = ax.pie(
    counts_list, 
    startangle=90, 
    wedgeprops=dict(width=0.3, edgecolor='w'), 
    textprops={'fontsize': 24}, 
    colors=colors  # 使用颜色
)

# 在扇形图的中间显示样本数
for i, wedge in enumerate(wedges):
    angle = (wedge.theta1 + wedge.theta2) / 2
    x = np.cos(np.radians(angle))
    y = np.sin(np.radians(angle))
    ax.text(x * 1.4, y * 0.5, str(counts_list[i]), ha='center', va='center', fontsize=36)
    
# 设置图例：每个部分的标签为组名和比例
legend_labels = [f"{group_labels[i]} : {percentages[i]}" for i in range(len(counts_list))]
ax.legend(
    wedges, 
    legend_labels, 
    loc='upper center', 
    fontsize=36, 
    bbox_to_anchor=(0.5, 0.0),  # 调整以将其放置在图的下方
    handletextpad=2
)
#加粗标题

ax.set(aspect="equal", title=title)
# ax.title.set_fontweight('bold')
# 中央文字
ax.text(0, 0, f'N={total_samples}', ha='center', va='center', fontsize=36)
# 标题字体大小
ax.title.set_fontsize(48)

# # 在图的下方添加文本
# text = 'Number of features per sample: '
# plt.text(0.45, 0.05, text, ha='center', va='center', fontsize=36, transform=fig.transFigure)
# plt.text(0.7, 0.05, '204', ha='center', va='center', fontsize=36, color='red', transform=fig.transFigure)

# 调整布局，减少周围白色区域
fig.tight_layout()

plt.show()
