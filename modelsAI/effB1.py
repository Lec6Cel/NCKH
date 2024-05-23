import torchvision
from torch import nn


def effB1():
    weights = torchvision.models.EfficientNet_B1_Weights.DEFAULT
    model = torchvision.models.efficientnet_b1(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features=1280, out_features=3)
    )
    return model
