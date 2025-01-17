import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import deeplake
from model import KanjiModel  # Import your model definition
from load_data import KanjiDataset  # Import the dataset
import time


# Load Kuzushiji Kanji dataset
ds = deeplake.load("hub://activeloop/kuzushiji-kanji")

# Define transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),
])


# Create the dataset and dataloader
dataset = KanjiDataset(ds, transform=transform)
dataloader = DataLoader(dataset, batch_size=1028, shuffle=True, num_workers=12, pin_memory=True)


# Initialize the model
model = KanjiModel(num_classes=3832)  # The number of classes corresponds to the number of Kanji characters


# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Set the model to training mode
model.train()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(next(model.parameters()).device)


def train_model():
    # Number of epochs to train
    num_epochs = 10

    for epoch in range(num_epochs):
        running_loss = 0.0
        i=0
        for images, labels in dataloader:
            if i%2==0:
                print(i)
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
                model.cuda()

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            i+=1

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader)}")

    # Save the model
    torch.save(model.state_dict(), 'kanji_model.pth')

if __name__ == '__main__':
    train_model()