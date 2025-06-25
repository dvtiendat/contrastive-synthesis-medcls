import torch.nn as nn
import torch
import torchvision.models as models
from src.models.resnet import ResNet, BasicBlock

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    
    def forward(self, x):
        return x

# SimCLR
class SimCLR(nn.Module):
    def __init__(self, linear_eval=False):
        super().__init__()
        self.linear_eval = linear_eval
        #resnet18 = models.resnet18(weights=None)
        resnet18 = ResNet(BasicBlock, [2, 2, 2, 2])
        resnet18.fc = Identity()
        self.encoder = resnet18
        self.projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
    def forward(self, x):
        if not self.linear_eval:
            x = torch.cat(x, dim=0)
        
        encoding = self.encoder(x)
        projection = self.projection(encoding) 
        return projection
    
# Định nghĩa mô hình LinearEvaluation
class Identity(nn.Module):
    def forward(self, x):
        return x

class LinearEvaluation(nn.Module):
    def __init__(self, model, nu_classes):
        super().__init__()
        self.simclr = model
        self.simclr.linear_eval = True
        self.simclr.projection = Identity()
        for param in self.simclr.parameters():
            param.requires_grad = True
        self.linear = nn.Linear(512, nu_classes)

    def forward(self, x):
        encoding = self.simclr(x)
        return self.linear(encoding)
