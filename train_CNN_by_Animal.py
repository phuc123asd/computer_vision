from split_and_label_images import load_dataset
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
import torch
from b import SimpleCNN
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor
from CustomImageDataset import CustomImageDataset

if __name__ == '__main__':
    # -----------------------------
    # 1️⃣ Device configuration
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 🧠 Gọi hàm để lấy dữ liệu
    train_data, test_data, label_dict = load_dataset()

    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    print("Classes:", label_dict)

    # 🎨 Tiền xử lý ảnh
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


    # 🔹 Tạo DataLoader
    train_loader = DataLoader(CustomImageDataset(train_data, transform), batch_size=32, shuffle=True)
    test_loader = DataLoader(CustomImageDataset(test_data, transform), batch_size=32)
    
    # -----------------------------
    # 3️⃣ Initialize model, loss, optimizer
    # -----------------------------
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    # -----------------------------
    # 4️⃣ Training loop
    # -----------------------------
    num_epochs = 100
    num_iters = len(train_loader)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward + optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{num_iters}], Loss: {loss.item():.4f}")

        print(f"Epoch [{epoch+1}/{num_epochs}] finished, Average loss: {running_loss / num_iters:.4f}")
    
    # -----------------------------
    # 💾 6️⃣ Save trained weights
    # -----------------------------
    torch.save(model.state_dict(), "cnn_animal_weights.pth")
    print("✅ Model weights saved to cnn_animal_weights.pth")
    # -----------------------------
    # 5️⃣ Testing phase
    # -----------------------------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"\n✅ Test Accuracy: {accuracy:.2f}%")

