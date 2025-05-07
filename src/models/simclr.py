import torch.nn as nn

class SimCLR(nn.Module):
    def __init__(self, backbone, out_dim):
        super(SimCLR, self).__init__()
        self.backbone = backbone[0]
        dim = backbone[1]
        self.projection_head = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        h = self.backbone(x)
        z = self.projection_head(h)
        return z
