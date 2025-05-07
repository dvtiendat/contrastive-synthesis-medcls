import torch.nn as nn

class FinetuneViT(nn.Module):
    """ Simple wrapper for ViT backbone + linear classification head. """
    def __init__(self, backbone, num_classes, freeze_backbone=False):
        super().__init__()
        self.backbone = backbone
        # Ensure backbone's own head is identity if it exists
        if hasattr(self.backbone, 'head') and not isinstance(self.backbone.head, nn.Identity):
             print("Replacing backbone's head with Identity.")
             self.backbone.head = nn.Identity()
        if hasattr(self.backbone, 'fc') and not isinstance(self.backbone.fc, nn.Identity):
            print("Replacing backbone's fc layer with Identity.")
            self.backbone.fc = nn.Identity()

        self.embed_dim = backbone.embed_dim
        self.head = nn.Linear(self.embed_dim, num_classes)

        if freeze_backbone == 'True':
            print("Freezing backbone parameters.")
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        embed = self.backbone(x)

        logits = self.head(embed)
        return logits