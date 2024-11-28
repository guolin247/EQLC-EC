import os
import pandas as pd
import numpy as np
import logging

def extract_and_save_data(input_file, output_folder):
    try:
        # 加载CSV数据，header=None确保不将第一行当作列名
        data = pd.read_csv(input_file, header=None)
        logging.info(f"成功加载文件: {input_file}")
    except Exception as e:
        logging.error(f"加载文件 {input_file} 时出错: {e}")
        return

    # 提取文件名并去掉".RAW"，然后加上".npy"作为文件名
    file_names = data.iloc[0, 1:].str.replace(".RAW", "", regex=False) + ".npy"

    # 提取类别标签
    label_names = data.iloc[1, 1:]

    # 创建标签到数字的映射
    unique_labels = label_names.unique()
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    labels = label_names.map(label_to_int)

    # 打印映射关系
    logging.info(f"标签到数字的映射: {label_to_int}")

    # 仅保存强度值，从第二列开始
    intensity_values = data.iloc[2:, 1:].astype(float)

    all_data_files = []
    all_labels = []
    all_sample_ids = []

    for i, file_name in enumerate(file_names):
        intensity = intensity_values.iloc[:, i].values
        
        # 保存仅有强度值的数组
        npy_file_path = os.path.join(output_folder, f"{file_name}")
        np.save(npy_file_path, intensity)
        logging.info(f"强度数据已保存到 {npy_file_path}")

        all_data_files.append(file_name)
        all_labels.append(labels.iat[i])

        # 解析 sample_id，假设 sample_id 在文件名的最后部分被下划线分隔
        sample_id = "_".join(file_name.split('_')[-3:-1])  # 获取倒数第二个和第三个部分作为sample_id
        all_sample_ids.append(sample_id)

    # 将文件名、标签和 sample_id 保存为 labels.npy
    labels_output = np.array(list(zip(all_data_files, all_labels, all_sample_ids)), dtype=object)
    labels_path = os.path.join(output_folder, 'labels.npy')
    np.save(labels_path, labels_output)
    logging.info(f"文件名、标签和 sample_id 已保存到 {labels_path}")

    # 将标签映射关系保存到一个文本文件中
    mapping_file_path = os.path.join(output_folder, 'label_mapping.txt')
    with open(mapping_file_path, 'w') as f:
        for label, idx in label_to_int.items():
            f.write(f"{label}: {idx}\n")
    logging.info(f"标签映射关系已保存到 {mapping_file_path}")

if __name__ == "__main__":
    # 设置日志记录
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    input_file = r"D:\work_GuoLin\machine_learning\machinelearning\src\guolin\CNN_tomato\raw\raw.csv"  # CSV数据文件
    output_folder = "processed_data"  # 用于保存.np文件的输出目录

    # 确保输出目录存在
    os.makedirs(output_folder, exist_ok=True)

    # 处理数据集
    extract_and_save_data(input_file, output_folder)
