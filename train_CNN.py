import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor
from a import MyDataset
from b import SimpleCNN

if __name__ == '__main__':
    # -----------------------------
    # 1️⃣ Device configuration
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -----------------------------
    # 2️⃣ Load datasets
    # -----------------------------
    train_dataset = MyDataset(root="data/cifar-10-batches-py", train=True)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    test_dataset = MyDataset(root="data/cifar-10-batches-py", train=False)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

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
    num_iters = len(train_dataloader)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_dataloader):
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
    # 5️⃣ Testing phase
    # -----------------------------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"\n✅ Test Accuracy: {accuracy:.2f}%")
