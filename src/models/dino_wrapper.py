import torch
import torch.nn as nn

class MultiCropWrapper(nn.Module):
    """ Handles forward pass for multi-crop augmentations."""
    def __init__(self, backbone, head):
        super().__init__()
        # backbone.fc = nn.Identity()
        # backbone.head = nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        if not isinstance(x, list):
            x = [x] 

        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)

        start_idx = 0
        output = torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx: end_idx]))

            if isinstance(_out, tuple):
                _out = _out[0]

            output = torch.cat((output, _out))
            start_idx = end_idx

        return self.head(output)
