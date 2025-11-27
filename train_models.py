import os
import random
import numpy as np
from PIL import Image
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------
# 1️⃣ Đường dẫn dataset
# -----------------------------
root_dir = '/workspace/computer_vision/raw-img'

# Class labels 10 loài
label_dict = {
    "Con Chó":0, "Con Ngựa":1, "Con Voi":2, "Con Bướm":3, "Con Gà":4,
    "Con Mèo":5, "Con Bò":6, "Con Cừu":7, "Con Nhện":8, "Con Sóc":9
}

# -----------------------------
# 2️⃣ Load dataset và flatten ảnh
# -----------------------------
X = []
y = []

for label_name, label_idx in label_dict.items():
    class_dir = os.path.join(root_dir, label_name)
    if not os.path.exists(class_dir):
        continue
    images = [f for f in os.listdir(class_dir) if f.lower().endswith((".jpg",".jpeg",".png"))]
    for img_name in images:
        img_path = os.path.join(class_dir, img_name)
        img = Image.open(img_path).convert("RGB").resize((128,128))
        img_arr = np.array(img).flatten()
        X.append(img_arr)
        y.append(label_idx)

X = np.array(X)
y = np.array(y)

print(f"Dataset loaded: {X.shape[0]} images, each vector size {X.shape[1]}")

# -----------------------------
# 3️⃣ Split train/test
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 4️⃣ Train KNN
# -----------------------------
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_test)
acc_knn = accuracy_score(y_test, y_pred_knn)
print(f"KNN Accuracy: {acc_knn*100:.2f}%")

# Lưu KNN
with open("knn_model.pkl", "wb") as f:
    pickle.dump(knn_model, f)
print("KNN model saved to knn_model.pkl")

# -----------------------------
# 5️⃣ Train Linear Regression
# -----------------------------
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred_linear = np.rint(linear_model.predict(X_test)).astype(int)
# Clip giá trị về 0-9
y_pred_linear = np.clip(y_pred_linear, 0, 9)

acc_linear = accuracy_score(y_test, y_pred_linear)
print(f"Linear Regression Accuracy: {acc_linear*100:.2f}%")

# Lưu Linear Regression
with open("linear_model.pkl", "wb") as f:
    pickle.dump(linear_model, f)
print("Linear Regression model saved to linear_model.pkl")
