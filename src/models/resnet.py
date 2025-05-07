import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# ResNet block
# -------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

# -------------------------
# ResNet Backbone
# -------------------------
class ResNet(nn.Module):
    def __init__(self, block, layers, include_top=True, num_classes=1000):
        super().__init__()
        self.in_channels = 64
        self.include_top = include_top

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if include_top:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if self.include_top:
            x = self.fc(x)
        return x

# -------------------------
# ResNet constructor
# -------------------------
def resnet18(include_top=True, num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], include_top, num_classes)

# -------------------------
# Wrapper cho backbone
# -------------------------
def get_resnet_backbone(out_dim=128):
    """
    Trả về backbone ResNet18 + projection head.
    Dùng cho SimCLR.
    """
    class ProjectionHead(nn.Module):
        def __init__(self, in_dim=512, out_dim=128):
            super().__init__()
            self.proj = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU(),
                nn.Linear(in_dim, out_dim)
            )

        def forward(self, x):
            return self.proj(x)

    backbone = resnet18(include_top=False)
    model = nn.Sequential(
        backbone,
        ProjectionHead(in_dim=512, out_dim=out_dim)
    )
    return model
