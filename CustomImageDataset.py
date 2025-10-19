from split_and_label_images import load_dataset
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch

# 🧱 Tạo Dataset tuỳ chỉnh từ 2 list
class CustomImageDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label