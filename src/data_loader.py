import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(image_size=224, augment=True):
    if augment:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_dataloaders(dataset_path, batch_size=32, num_workers=0, image_size=224):
    train_data = datasets.ImageFolder(os.path.join(dataset_path, "train"), transform=get_transforms(image_size, True))
    val_data   = datasets.ImageFolder(os.path.join(dataset_path, "val"),   transform=get_transforms(image_size, False))
    test_data  = datasets.ImageFolder(os.path.join(dataset_path, "test"),  transform=get_transforms(image_size, False))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_data, val_data, test_data
