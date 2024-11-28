import numpy as np
from PIL import Image
from torchvision import transforms
import os

# 加载数据
x_train = np.load(r'D:\work_GuoLin\machine_learning\machinelearning\src\guolin\CNN_single_ICC\stratified_split_data\X_train.npy')
y_train = np.load(r'D:\work_GuoLin\machine_learning\machinelearning\src\guolin\CNN_single_ICC\stratified_split_data\y_train.npy')

# 数据增强函数
def augment_data(images, labels, augmentations):
    augmented_images = []
    augmented_labels = []
    
    for i, img in enumerate(images):
        pil_img = Image.fromarray(img)
        
        for aug in augmentations:
            aug_img = aug(pil_img)
            augmented_images.append(np.array(aug_img))
            augmented_labels.append(labels[i])
    
    return np.array(augmented_images), np.array(augmented_labels)

# 定义增强操作
augmentations = [
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(),
    # 可以添加更多增强操作
]

# 执行数据增强
augmented_x_train, augmented_y_train = augment_data(x_train, y_train, augmentations)

# 合并增强后的数据
x_train_augmented = np.concatenate((x_train, augmented_x_train), axis=0)
y_train_augmented = np.concatenate((y_train, augmented_y_train), axis=0)

# 保存增强后的数据
np.save(r'D:\work_GuoLin\machine_learning\machinelearning\src\guolin\CNN_single_ICC\stratified_split_data\x_train_augmented.npy', x_train_augmented)
np.save(r'D:\work_GuoLin\machine_learning\machinelearning\src\guolin\CNN_single_ICC\stratified_split_data\y_train_augmented.npy', y_train_augmented)

print("Data augmentation completed and saved.")
