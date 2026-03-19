## 📌 Overview

**Animal Classification AI PRO** is a comprehensive computer vision project designed to classify **10 different animal species** using both **Deep Learning** and **Traditional Machine Learning** approaches.

The project is built to not only achieve high accuracy but also to **compare model behaviors**, **visualize predictions**, and provide a **real-time interactive experience** via a GUI application.

---

## 🎯 Objectives

* Build a **custom CNN architecture from scratch**
* Compare performance between:

  * Deep Learning models (CNN, Fully Connected)
  * Traditional ML models (KNN, Linear Regression)
* Develop a **real-time prediction system**
* Create a **clean and user-friendly GUI**
* Understand the **trade-offs between model types**

---

## 🧠 Models Implemented

### 🔹 Deep Learning

* **Custom CNN (PyTorch)**

  * 5 Convolutional Blocks:

    ```
    Conv2d → BatchNorm → ReLU → MaxPool
    ```
  * Input: `128x128 RGB images`
  * Strong feature extraction capability

* **Fully Connected Network**

  * Baseline deep learning model
  * Helps compare against CNN performance

---

## 🖥️ Demo

### 🎥 Demo Image

<img width="847" height="717" alt="Ảnh màn hình 2026-03-19 lúc 08 14 52" src="https://github.com/user-attachments/assets/5af1839a-467c-4e3b-923f-1b05e77dc3ae" />


---

### 🧪 Demo Preview (GUI)

<p align="center">
<a href="https://youtu.be/2rDx8kwE-kY">
<img src="https://img.youtube.com/vi/2rDx8kwE-kY/0.jpg" width="800">
</a>
</p>

```
[ User clicks "CHỌN ẢNH" ]
        ↓
[ Image Loaded ]
        ↓
[ Model Predicts ]
        ↓
[ Result Displayed ]

→ "Prediction: Elephant 🐘 (Confidence: 92%)"
```

---

## 📂 Project Structure

```bash
computer_vision/
│
├── raw-img/                        # Dataset (Animal 10)
│   ├── dog/
│   ├── horse/
│   ├── elephant/
│   └── ...
│
├── train_CNN_by_Animal.py          # Train CNN model
├── train_FC_by_Animal.py           # Train FC model
├── train_models.py                 # Train KNN & Linear Regression
│
├── test_CNN_last_train_AnimalDataset.py   # GUI Application
│
├── b.py                            # CNN architecture
├── fc.py                           # FC architecture
├── CustomImageDataset.py           # Dataset loader
│
├── cnn_animal_weights.pth          # Saved model weights
├── requirements.txt
└── README.md
```

---

## 🛠️ Installation

### 1. Clone repository

```bash
git clone <repository-url>
cd computer_vision
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset Setup

Download **Animal 10 Dataset** and place it like this:

```bash
computer_vision/
└── raw-img/
    ├── dog/
    ├── cat/
    ├── elephant/
    ├── horse/
    └── ... (10 classes)
```

---

## 🚀 Usage

### 🔹 Train CNN (Recommended)

```bash
python train_CNN_by_Animal.py
```

→ Output:

```
cnn_animal_weights.pth
```

---

### 🔹 Train Traditional Models

```bash
python train_models.py
```

---

### 🔹 Run GUI Application

```bash
python test_CNN_last_train_AnimalDataset.py
```

---

## 📈 Performance Insights

| Model             | Strengths             | Weaknesses                 |
| ----------------- | --------------------- | -------------------------- |
| CNN               | High accuracy, robust | Slower training            |
| Fully Connected   | Simple baseline       | Poor spatial understanding |
| KNN               | Easy to implement     | Slow inference             |
| Linear Regression | Fast                  | Low accuracy               |

---

## 🔍 Technical Highlights

* Built **from scratch CNN** (no pretrained models)
* Applied **Batch Normalization** for stability
* Used **MaxPooling** for spatial reduction
* Implemented **custom dataset pipeline**
* Compared **feature-based vs pixel-based learning**

---

## 💡 Future Improvements

* Add **Transfer Learning (ResNet, EfficientNet)**
* Deploy as **Web App (Flask / FastAPI)**
* Add **Model Explainability (Grad-CAM)**
* Optimize with **GPU training**
* Convert to **Mobile App**

---

## 👨‍💻 Author

**Phuc Vo**
AI / ML Engineer (Aspiring)

* Passionate about **Deep Learning & Computer Vision**
* Focused on understanding models at a **fundamental level**
