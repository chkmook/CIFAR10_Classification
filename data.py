import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets 

def TrainLoader(data_path):
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = dsets.CIFAR10(root=data_path,
                                train=True, 
                                transform=transform_train,
                                download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=32, 
                                            shuffle=True)

    return train_dataset, train_loader


def TestLoader(data_path):

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_dataset = dsets.CIFAR10(root=data_path, 
                                train=False, 
                                transform=transform_test)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                            batch_size=32, 
                                            shuffle=False)

    return test_dataset, test_loader