import torchvision
import torch
from torch.utils.data import DataLoader

def data_loader(args):
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(args.image_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor()
    ])

    test_set11_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256,256)),
        torchvision.transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.ImageFolder(args.data_path,transform=train_transforms)
    train_loader = DataLoader(train_dataset,batch_size=args.batchSize,shuffle=True,num_workers=4,
                              pin_memory=False,drop_last=False)

    test_set11 = torchvision.datasets.ImageFolder(args.set11_path,transform=test_set11_transforms)
    test_loader = DataLoader(test_set11,batch_size=1,shuffle=False,num_workers=4,
                                    pin_memory=False,drop_last = False)

    return train_loader, test_loader


