from split_and_label_images import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from fc import FC
import torch.optim as optim
import torch.nn as nn
from torchvision.transforms import ToTensor
from CustomImageDataset import CustomImageDataset

if __name__ == '__main__':
    # -----------------------------
    # 1️ Device configuration
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Gọi hàm để lấy dữ liệu
    train_data, test_data, label_dict = load_dataset()
    print("Train samples:", train_data[0])
    print("Classes:", label_dict)
    
    # Tiền xử lý ảnh
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    # Tạo DataLoader
    train_loader = DataLoader(CustomImageDataset(train_data, transform), batch_size=32, shuffle=True)
    test_loader = DataLoader(CustomImageDataset(test_data, transform), batch_size=32)
    # -----------------------------
    # 3️ Initialize model, loss, optimizer
    # -----------------------------
    model = FC().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    # -----------------------------
    # 4️ Training loop
    # -----------------------------
    num_epochs = 10
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
    # Save trained weights
    # -----------------------------
    torch.save(model.state_dict(), "fc_animal_weights.pth")
    print("Model weights saved to cnn_animal_weights.pth")
    # -----------------------------
    # 5️ Testing phase
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
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    