import torch
import torch.nn as nn

class FC(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=49152 , out_features=512),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=512 , out_features=1024),
            nn.ReLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=1024 , out_features=num_classes),
            nn.ReLU(),
        )
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
if __name__ == '__main__':
    model = FC()
    input_data = torch.rand(1, 3, 128, 128)
    if torch.cuda.is_available():
        model.cuda()
        input_data = input_data.cuda()

    result = model(input_data) 
    print("Model output shape:", result)
