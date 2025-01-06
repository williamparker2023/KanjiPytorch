import torch
import torch.nn as nn
import torch.optim as optim

class KanjiModel(nn.Module):
    def __init__(self, num_classes=3832):
        super(KanjiModel, self).__init__()
        
        # Define the layers of the CNN
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)  # You may need to adjust this depending on your image size
        self.fc2 = nn.Linear(512, num_classes)  # The number of output classes (Kanji characters)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # Apply Conv1 + ReLU + MaxPool
        x = self.pool(torch.relu(self.conv2(x)))  # Apply Conv2 + ReLU + MaxPool
        x = self.pool(torch.relu(self.conv3(x)))  # Apply Conv3 + ReLU + MaxPool
        x = x.view(-1, 128 * 8 * 8)  # Flatten the tensor for the fully connected layer
        x = torch.relu(self.fc1(x))  # Fully connected layer 1
        x = self.fc2(x)  # Fully connected layer 2 (output layer)
        return x
