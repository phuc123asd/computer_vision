from dl_for_computer_vision.a import MyDataset
from dl_for_computer_vision.b import SimpleNN
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim
from torchvision.transforms import ToTensor

if __name__ == '__main__':
    
    train_dataset = MyDataset(root="data/cifar-10-batches-py", train=True)
    train_dataloader = DataLoader(
        dataset= train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    test_dataset = MyDataset(root="data/cifar-10-batches-py", train=False)
    train_dataloader = DataLoader(
        dataset= test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )
    model = SimpleNN(number_class=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    num_iters = len(train_dataloader)
    for epoch in range(100):
        model.train()
        for iter, (images, labels) in enumerate(train_dataloader):
            output = model(images)
            loss_value = criterion(output, labels)
            print("Epoch{}/{}. Iteration {}/{}. Loss {}".format(epoch+1, 100, iter+1, num_iters, loss_value))
            
            #backward and optimizer
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
