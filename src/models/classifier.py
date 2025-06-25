import torch.nn as nn
import torch

class LinearClassifier(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(512, num_classes)  # 512: tùy thuộc backbone

    def forward(self, x):
        with torch.no_grad():
            h = self.backbone(x)
        return self.classifier(h)
