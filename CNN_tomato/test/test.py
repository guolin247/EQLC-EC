import pandas as pd
import matplotlib.pyplot as plt

# CSV 文件路径
file_path = r"D:\work_GuoLin\machine_learning\machinelearning\src\guolin\CNN_single_ICC\test\RandomForest_20241016_143015.csv"

# 加载数据
data = pd.read_csv(file_path)

# 查看数据形式
print(data.head())  # 打印前几行数据以检查其形式

# 绘制准确率的箱线图
plt.figure(figsize=(10, 6))
plt.boxplot(data['Accuracy'])
plt.title('Boxplot of Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Models')
plt.grid(axis='x')
plt.show()
