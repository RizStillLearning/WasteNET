from torch.utils.data import Dataset

class WasteDataset(Dataset):
    def __init__(self, dataset, transform=None, target_transform=None):
        super().__init__()
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset['image'])

    def __getitem__(self, idx):
        image = self.dataset['image'][idx]
        label = self.dataset['label'][idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label