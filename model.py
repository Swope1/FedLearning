import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(32 * 32 * 3, 256)
        self.dense2 = nn.Linear(256, 64)
        self.dense3 = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        return self.dense3(x)

class Net2(nn.Module):
    def __init__(self) -> None:
        super(Net2, self).__init__()
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(32 * 32 * 3, 1024)
        self.dense2 = nn.Linear(1024, 256)
        self.dense3 = nn.Linear(256, 64)
        self.dense4 = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
        return self.dense4(x)

class ConvNet(nn.Module):

    def __init__(self) -> None:
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.dense1 = nn.Linear(16 * 5 * 5, 120)
        self.dense2 = nn.Linear(120, 84)
        self.dense3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        return self.dense3(x)

class ConvNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True)
        self.pool1 = nn.MaxPool2d(3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.pool2 = nn.MaxPool2d(3)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.pool3 = nn.MaxPool2d(3)
        self.drop = nn.Dropout(0.1)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(64, 1024)
        self.dense2 = nn.Linear(1024, 10)
        self.softmax = nn.Softmax(1)
    
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.drop(x)
        x = self.flatten(x)
        x = F.relu(self.dense1(x))
        return self.dense2(x)
            
def train(client_net, train_loader, optimizer, loss_fn):
    client_net.train()
    for batch in train_loader:
        imgs, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
        optimizer.zero_grad()
        loss_fn(client_net(imgs), labels).backward()
        optimizer.step()

@torch.no_grad()
def test(client_net, test_loader, loss_fn):
    client_net.eval()
    correct, loss = 0, 0.0
    for batch in test_loader:
        imgs, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
        predictions = client_net(imgs)
        loss += loss_fn(predictions, labels).item()
        correct += (torch.max(predictions.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(test_loader.dataset)
    return loss, accuracy