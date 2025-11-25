import torch
from torchvision import transforms

target_size = (128, 128)
transform = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(target_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.ConvertImageDtype(torch.float),
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(target_size),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.ConvertImageDtype(torch.float),
    ]),
}

target_transform = transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.long))