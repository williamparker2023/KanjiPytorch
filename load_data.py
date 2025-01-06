import deeplake
from torch.utils.data import Dataset
import torch

# Load Kuzushiji Kanji dataset
ds = deeplake.load("hub://activeloop/kuzushiji-kanji")



# Define your custom dataset class
class KanjiDataset(Dataset):
    def __init__(self, deeplake_ds, transform=None):
        self.ds = deeplake_ds
        self.transform = transform
        
    def __len__(self):
        return len(self.ds['images'])
    
    def __getitem__(self, idx):
        images = self.ds['images'][idx].numpy()
        labels = self.ds['labels'][idx].numpy()

        labels = labels.squeeze()
        labels = torch.tensor(labels, dtype=torch.long)
        
        if self.transform:
            images = self.transform(images)
        

        return images, labels

# Test loading the data
if __name__ == "__main__":
    print(ds['images'])  # This should print the images tensor
    print(ds['labels'])  # This should print the labels tensor
