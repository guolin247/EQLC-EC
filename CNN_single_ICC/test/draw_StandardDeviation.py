import pandas as pd
import matplotlib.pyplot as plt

# 读取 Excel 文件
file_path = r'D:\work_GuoLin\machine_learning\machinelearning\src\guolin\CNN_single_ICC\test\Standard_Deviation.xlsx'
data = pd.read_excel(file_path)

# 设置模型名称为索引
data.set_index('models', inplace=True)

# 手动定义每个算法的颜色
color_map = {
    'HardVoting': '#EC3E31',  # 红色
    'SoftVoting': '#EC3E31',  # 红色
    'EQLC': '#A6D0E6',    
    'XGBoost': '#4F99C9',  
    'AdaBoost': '#A6D0E6',  
    'SVM': '#A6D0E6',  
    'RandomForest': '#A6D0E6',  
}

# 设置默认颜色为灰色，如果在字典中找不到则使用默认颜色
colors = [color_map.get(model, 'gray') for model in data.index]

# 创建一个新的图形
plt.figure(figsize=(10, 8))

# 定义所有子图的x轴取值范围, 可以根据需要手动调整
x_limits = {
    'Accuracy': (0.02, 0.028),
    'Precision': (0.02, 0.044),
    'Recall': (0.02, 0.048),
    'F1_Score': (0.02, 0.04)
}

# 设置x轴和y轴标签的字体大小
x_label_fontsize = 16
y_label_fontsize = 16
x_ticks_fontsize = 12
y_ticks_fontsize = 16

# 绘制柱形图
metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score']

for i, metric in enumerate(metrics):
    ax = plt.subplot(2, 2, i + 1)  # 2行2列的子图
    plt.barh(data.index, data[metric], color=colors)
    
    plt.title(metric, fontsize=x_label_fontsize)
    plt.xlabel('Standard_Deviation', fontsize=x_label_fontsize)
    plt.ylabel('Models', fontsize=y_label_fontsize)
    plt.xticks(fontsize=x_ticks_fontsize)
    plt.yticks(fontsize=y_ticks_fontsize)

    # 设置x轴取值范围
    plt.xlim(x_limits[metric])

    # 去除网格线
    plt.grid(False)

    # 去掉边框
    for spine in ax.spines.values():
        spine.set_visible(False)

# 自动调整子图参数
plt.tight_layout()

# 保存图像
plt.savefig('model_stability_metrics_custom_colors.png', dpi=300, bbox_inches='tight')
plt.show()  # 显示图像
