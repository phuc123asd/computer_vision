# split_dataset.py
import os
import random

# 🧭 Đường dẫn dữ liệu
root_dir = r"D:\workspace\hoc_python\dl_for_computer_vision\raw-img"
train_ratio = 0.8

def load_dataset():
    class_names = sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')
    ])
    label_dict = {name: idx for idx, name in enumerate(class_names)}
    train_data, test_data = [], []

    for label_name in class_names:
        class_dir = os.path.join(root_dir, label_name)
        images = [f for f in os.listdir(class_dir)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        random.shuffle(images)
        split_point = int(len(images) * train_ratio)
        train_imgs = images[:split_point]
        test_imgs = images[split_point:]

        for img in train_imgs:
            train_data.append((os.path.join(class_dir, img), label_dict[label_name]))
        for img in test_imgs:
            test_data.append((os.path.join(class_dir, img), label_dict[label_name]))

    return train_data, test_data, label_dict
