import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as dsets 
import torchvision
import time
import os
from tqdm.auto import tqdm


from data import TrainLoader, TestLoader
from model import ResNet, VGG

if __name__ == "__main__":

    data_path='/datasets/cifar10/' # dataset path

    train_dataset, train_loader = TrainLoader(data_path)
    test_dataset, test_loader = TestLoader(data_path)
    
    print("The number of training images : ", len(train_dataset))
    print("The number of test images : ", len(test_dataset))


    # Can choose model

    #model = VGG(11)
    #model = VGG(13)
    #model = VGG(16)
    model = VGG(19)
    #model = ResNet(18)
    #model = ResNet(34)
    #model = ResNet(50)
    #model = ResNet(101)
    #model = ResNet(152)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    learning_rate = 1e-2
    momentum = 0.9
    weight_decay = 5e-4

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum = momentum, weight_decay = weight_decay)

    num_epochs = 20
    
    for epoch in range(num_epochs):

        avgloss = 0

        for i, (images, labels) in tqdm(enumerate(train_loader), total = len(train_loader)):
        
            images = images.view(-1, 3, 32, 32).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            avgloss += loss.item()
            loss.backward()
            optimizer.step()
            
        avgloss = avgloss/(i+1)

        correct = 0
        total = 0

        for images, labels in test_loader:
            images = images.view(-1, 3, 32, 32).to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            if torch.cuda.is_available():
                correct += (predicted.cpu() == labels.cpu()).sum()
            else:
                correct += (predicted == labels).sum()
    
        accuracy = 100 * correct.item() / total

        print('Epoch: {}. Loss: {}. Accuracy: {}.'.format(epoch+1, avgloss, accuracy))