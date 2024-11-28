import os
import numpy as np
import pandas as pd

# 文件夹路径
base_folder = r"D:\work_GuoLin\machine_learning\machinelearning\src\guolin\CNN_single_HIP_CER"
data_folder = os.path.join(base_folder, "data")
output_folder = os.path.join(base_folder, "output_for_cnn")

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 初始化标签映射字典
label_mapping = {}

# 用于保存所有样本的标签和文件名
all_labels = []

# 遍历每个样本的 csv 文件
for csv_file in os.listdir(data_folder):
    if csv_file.endswith(".csv"):
        csv_path = os.path.join(data_folder, csv_file)
        
        # 读取CSV文件，第一行是列名称，最后一列是标签
        df = pd.read_csv(csv_path)
        
        # 获取标签列并将其从数据中移除
        raw_labels = df.iloc[:, -1].values  # 最后一列是标签
        data = df.iloc[:, :-1].values       # 其余列是谱图数据
        
        # 确认数据行数是否为1544，列数是否为511
        assert data.shape == (1201, 2463), "数据形状不符合预期"

        # 数字化标签，如果标签未在映射中，则添加新的数字化映射
        unique_labels = np.unique(raw_labels)
        for label in unique_labels:
            if label not in label_mapping:
                label_mapping[label] = len(label_mapping)

        # 将原始标签转换为数字化标签
        numeric_labels = np.array([label_mapping[label] for label in raw_labels])

        # 遍历每个样本的谱图数据
        for idx, spectrum in enumerate(data):
            # 将每一行数据保存为三维张量 [1, 511]
            tensor = spectrum.reshape(1, -1)  # 变为 1 x 511 的数组

            # 处理文件名，移除 .csv 扩展名
            base_name = os.path.splitext(csv_file)[0]
            output_npy_path = os.path.join(output_folder, f"{base_name}_spectrum_{idx+1}.npy")
            np.save(output_npy_path, tensor)

            # 保存文件名和数字化标签到 all_labels，确保移除 .csv
            all_labels.append((f"{base_name}_spectrum_{idx+1}.npy", numeric_labels[idx]))

            print(f"保存了样本 {csv_file} 的第 {idx+1} 个谱图的张量为 .npy 文件。")

# 保存文件名和对应的数字化标签到 labels.npy
output_label_path = os.path.join(output_folder, "labels.npy")
np.save(output_label_path, all_labels)

# 打印原始标签和数字化标签的映射关系
print("原始标签和数字化标签的映射关系:")
for label, numeric in label_mapping.items():
    print(f"'{label}' -> {numeric}")

print("所有样本处理完成，标签文件已更新。")
