import torch
from torchvision import transforms
from PIL import Image, ImageTk
from b import SimpleCNN
import tkinter as tk
from tkinter import filedialog
import torch.nn as nn

# -----------------------------
# 1️⃣ Load model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("cnn_animal_weights.pth", map_location=device))
model.eval()
print("✅ Model loaded and ready for prediction.")

# -----------------------------
# 2️⃣ Transform ảnh đầu vào
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# -----------------------------
# 3️⃣ Label dictionary
# -----------------------------
label_dict = {"Con Chó": 0, "Con Ngựa": 1, "Con Voi": 2, "Con Bướm": 3, "Con Gà": 4,
              "Con Mèo": 5, "Con Bò": 6, "Con Cừu": 7, "Con Nhện": 8, "Con Sóc": 9}


# -----------------------------
# 4️⃣ Hàm dự đoán
# -----------------------------
def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        
        # Kiểm tra lại hàm softmax
        softmax = nn.Softmax(dim=1)
        probs = softmax(outputs)
        
        _, predicted = torch.max(outputs.data, 1)
    label_name = list(label_dict.keys())[list(label_dict.values()).index(predicted.item())]
    return label_name, probs


# -----------------------------
# 5️⃣ Giao diện Tkinter - THIẾT KẾ MỚI
# -----------------------------
root = tk.Tk()
root.title("🐾 Phân loại súc vật - AI Pro")
root.geometry("850x700")
root.configure(bg="#1e1e2e")

# -----------------------------
# STYLE MỚI
# -----------------------------
style = {
    "bg": "#1e1e2e",
    "card_bg": "#2a2a3c",
    "accent": "#ff79c6",
    "success": "#50fa7b",
    "text": "#f8f8f2",
    "secondary": "#6272a4"
}

# -----------------------------
# HEADER
# -----------------------------
header_frame = tk.Frame(root, bg=style["bg"])
header_frame.pack(fill="x", pady=20)

title_label = tk.Label(header_frame, text="🤖 AI Classfication Animal", 
                      font=("Arial", 24, "bold"), 
                      bg=style["bg"], fg=style["accent"])
title_label.pack()

sub_label = tk.Label(header_frame, text="Chọn ảnh và xem AI đoán đúng không!", 
                    font=("Arial", 12), 
                    bg=style["bg"], fg=style["secondary"])
sub_label.pack(pady=5)

# -----------------------------
# MAIN CONTENT
# -----------------------------
main_frame = tk.Frame(root, bg=style["bg"])
main_frame.pack(fill="both", expand=True, padx=30)

# LEFT - IMAGE FRAME
left_frame = tk.Frame(main_frame, bg=style["card_bg"], relief="flat", bd=0)
left_frame.grid(row=0, column=0, padx=(0, 15), pady=10, sticky="nsew")

img_header = tk.Label(left_frame, text="🖼️ ẢNH ĐÃ CHỌN", 
                     font=("Arial", 12, "bold"), 
                     bg=style["card_bg"], fg=style["text"])
img_header.pack(pady=15)

# Frame ảnh với shadow effect
img_shadow = tk.Frame(left_frame, bg=style["accent"])
img_shadow.pack(pady=5, padx=20)

frame = tk.Frame(img_shadow, width=350, height=350, bg="#44475a", relief="flat", bd=0)
frame.pack(padx=2, pady=2)
frame.pack_propagate(False)

img_label = tk.Label(frame, bg="#44475a")
img_label.pack(expand=True)

# RIGHT - RESULTS FRAME
right_frame = tk.Frame(main_frame, bg=style["card_bg"], relief="flat", bd=0)
right_frame.grid(row=0, column=1, padx=(15, 0), pady=10, sticky="nsew")

results_header = tk.Label(right_frame, text="📊 KẾT QUẢ THÔNG MINH", 
                         font=("Arial", 12, "bold"), 
                         bg=style["card_bg"], fg=style["text"])
results_header.pack(pady=15)

# Kết quả chính
main_result_frame = tk.Frame(right_frame, bg=style["card_bg"])
main_result_frame.pack(fill="x", pady=15, padx=20)

result_label = tk.Label(main_result_frame, 
                       text="🎯 Chọn ảnh để AI đoán nè!",
                       font=("Arial", 16, "bold"), 
                       bg=style["card_bg"], fg=style["success"],
                       wraplength=300)
result_label.pack()

# Xác suất chi tiết
confidence_frame = tk.Frame(right_frame, bg=style["card_bg"])
confidence_frame.pack(fill="x", pady=10, padx=20)

confidence_title = tk.Label(confidence_frame, text="🔥 ĐỘ TIN CẬY:", 
                           font=("Arial", 11, "bold"), 
                           bg=style["card_bg"], fg=style["secondary"])
confidence_title.pack(anchor="w")

confidence_value = tk.Label(confidence_frame, text="0%", 
                           font=("Arial", 20, "bold"), 
                           bg=style["card_bg"], fg=style["accent"])
confidence_value.pack(pady=5)

# Progress bar đơn giản
progress_frame = tk.Frame(confidence_frame, bg=style["card_bg"])
progress_frame.pack(fill="x", pady=5)

progress_bg = tk.Frame(progress_frame, height=10, bg=style["secondary"])
progress_bg.pack(fill="x")
progress_bg.pack_propagate(False)

progress_fill = tk.Frame(progress_bg, height=10, bg=style["success"])
progress_fill.pack(side="left", fill="y")

# Thông tin thêm
info_frame = tk.Frame(right_frame, bg=style["card_bg"])
info_frame.pack(fill="x", pady=20, padx=20)

info_label = tk.Label(info_frame, 
                     text="💡 AI được huấn luyện nhận diện 10 loài động vật khác nhau",
                     font=("Arial", 10), 
                     bg=style["card_bg"], fg=style["secondary"],
                     wraplength=300, justify="left")
info_label.pack(anchor="w")

# -----------------------------
# BUTTON - THIẾT KẾ MỚI
# -----------------------------
button_frame = tk.Frame(root, bg=style["bg"])
button_frame.pack(fill="x", pady=25)

def choose_image():
    file_path = filedialog.askopenfilename(
        title="Selected Image - Pro Version",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")]
    )
    if file_path:
        # Load và hiển thị ảnh
        img = Image.open(file_path)
        img.thumbnail((340, 340), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(img)

        img_label.configure(image=tk_img)
        img_label.image = tk_img

        # Dự đoán
        prediction, probs = predict_image(file_path)
        confidence = torch.max(probs).item() * 100
        
        # Cập nhật kết quả
        result_label.configure(text=f"🎯 Tôi nghĩ là: {prediction}")
        confidence_value.configure(text=f"{confidence:.1f}%")
        
        # Cập nhật progress bar
        progress_width = int((confidence / 100) * 300)
        progress_fill.configure(width=progress_width)

choose_btn = tk.Button(button_frame, 
                      text="🚀 CHỌN ẢNH & XEM AI ĐOÁN", 
                      command=choose_image,
                      font=("Arial", 14, "bold"), 
                      bg=style["accent"], 
                      fg=style["text"],
                      relief="flat",
                      padx=25, 
                      pady=12,
                      cursor="hand2")
choose_btn.pack()

<<<<<<< HEAD
# -----------------------------
=======
# ----------------------------
>>>>>>> 200e96f (first commit)
# FOOTER
# -----------------------------
footer_frame = tk.Frame(root, bg=style["bg"])
footer_frame.pack(fill="x", pady=10)

footer_label = tk.Label(footer_frame, 
                       text="✨ Powered by PyTorch | Design by Pro ✨",
                       font=("Arial", 10), 
                       bg=style["bg"], 
                       fg=style["secondary"])
footer_label.pack()

# Configure grid weights
main_frame.grid_columnconfigure(0, weight=1)
main_frame.grid_columnconfigure(1, weight=1)
main_frame.grid_rowconfigure(0, weight=1)

root.resizable(False, False)
root.mainloop()