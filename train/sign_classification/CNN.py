import torch 
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import Subset, DataLoader

import numpy as np
import matplotlib.pyplot as plt

from dataset import SignsDataset

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.convolution_part = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Bloc 2 : Contours internes (48x48 -> 24x24)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            # Bloc 3 : Idem (24x24 -> 12x12)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # 🎯 Bloc 4 (Le Nouveau !) : Micro-détails (12x12 -> 6x6)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.flatten = nn.Flatten()

        self.dense_layers = nn.Sequential(
            nn.Linear(6*6*256, 120),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(84, 3)
        )

        # We do not use Softmax, because the loss function CrossEntropyLoss needs pure logits. 
    
    def forward(self, x):
        x = self.convolution_part(x)
        x = self.flatten(x)
        x = self.dense_layers(x)
        return x

def train_loop(dataloader, model, loss_function, optimizer, batch_size):
    size = len(dataloader.dataset)
    # Model in the train state
    model.train()

    sum_losses = 0
    for batch_idx, (batch, labels) in enumerate(dataloader):
        # Compute predictions
        pred = model(batch)
        loss = loss_function(pred, labels)

        # Optimization of the parameters
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch_idx % 10 == 0:
            loss, current = loss.item(), batch_idx * batch_size + len(batch)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        sum_losses += loss
    return sum_losses/len(dataloader)
        


def test_loop(dataloader, model, loss_function):
    # Model in the evaluation state
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # 
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_function(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss    

if __name__ == "__main__":
    model = CNN()
    # Definition of the hyper parameters
    epochs = 25
    learning_rate = 0.001
    batch_size = 32

    img_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Resize((96, 96))])

    train_data = SignsDataset("train_dataset", 3, transform=img_transform) # Size --> 2525
    test_data = SignsDataset("test_dataset", 3, transform=img_transform) # Size --> 1199

    train_dataloader = DataLoader(train_data, batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    train_losses = []
    test_losses = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss = train_loop(train_dataloader, model, loss_fn, optimizer, batch_size).item()
        test_loss = test_loop(test_dataloader, model, loss_fn)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
    print("Practise done !")

    print("Saving the weights ...")
    torch.save(model.state_dict(), "weights.pth")

    # Result in a graph
    train_losses = np.array(train_losses)
    test_losses = np.array(test_losses)

    X = np.array([(i+1) for i in range(epochs)])
    plt.plot(X, train_losses, 'bo')
    plt.plot(X, test_losses, 'ro')
    plt.title("Training results")
    plt.xlabel("Epoch")
    plt.ylabel("losses")
    plt.show()
