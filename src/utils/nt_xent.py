import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5, device='cuda'):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, zis, zjs):
        """
        zis: tensor (batch_size, projection_dim) - output từ ảnh gốc
        zjs: tensor (batch_size, projection_dim) - output từ ảnh augment
        """
        batch_size = zis.shape[0]

        # Normalize embeddings
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        # Hợp nhất lại
        representations = torch.cat([zis, zjs], dim=0)  # (2N, D)
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=2
        )  # (2N, 2N)

        # Tạo mask loại bỏ chính nó (i == j)
        self_mask = torch.eye(2 * batch_size, dtype=torch.bool).to(self.device)
        positives_mask = torch.zeros_like(self_mask)
        for i in range(batch_size):
            positives_mask[i, i + batch_size] = 1
            positives_mask[i + batch_size, i] = 1

        # Positive similarities
        positives = similarity_matrix[positives_mask.bool()].view(2 * batch_size, 1)
        negatives = similarity_matrix[~(self_mask | positives_mask)].view(2 * batch_size, -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(2 * batch_size).long().to(self.device)

        logits /= self.temperature
        loss = F.cross_entropy(logits, labels)
        return loss
