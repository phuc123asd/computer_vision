import os
import csv

# 🧭 Đường dẫn chứa dữ liệu ảnh
root_dir = r"D:\workspace\hoc_python\dl_for_computer_vision\raw-img"

# 📝 File CSV xuất ra
output_csv = os.path.join(root_dir, "labels.csv")

# 🧠 Danh sách kết quả
rows = []

# Duyệt qua các thư mục con (mỗi thư mục = 1 class)
for label_name in sorted(os.listdir(root_dir)):
    class_dir = os.path.join(root_dir, label_name)
    if not os.path.isdir(class_dir):
        continue

    for file_name in os.listdir(class_dir):
        if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            file_path = os.path.join(class_dir, file_name)
            rows.append([file_path, label_name])

# Ghi vào CSV
with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["file_path", "label"])
    writer.writerows(rows)

print(f"✅ Total images: {len(rows)}")
print(f"📄 Labels saved to: {output_csv}")
