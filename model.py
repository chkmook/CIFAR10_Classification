import torch.nn as nn
import torchvision.models as models



def ResNet(layers):
    
    if layers == 18:
        model = models.resnet18()
        model.fc = nn.Linear(512, 10)
    elif layers == 34:
        model = models.resnet34()
        model.fc = nn.Linear(512, 10)
    elif layers == 50:
        model = models.resnet50()
        model.fc = nn.Linear(2048, 10)
    elif layers == 101:
        model = models.resnet101()
        model.fc = nn.Linear(2048, 10)
    elif layers == 152:
        model = models.resnet152()
        model.fc = nn.Linear(2048, 10)

    

    return model



def VGG(layers):
    
    if layers == 11:
        model = models.vgg11()
    elif layers == 13:
        model = models.vgg13()
    elif layers == 16:
        model = models.vgg16()
    elif layers == 19:
        model = models.vgg19()

    model.classifier[-1] = nn.Linear(4096, 10)

    return model