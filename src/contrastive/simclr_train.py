import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from src.models.resnet import get_resnet_backbone
from models.simclr import SimCLR
from src.data.transforms import get_simclr_augmentation
from src.utils.helper import set_seed, save_checkpoint
from utils.nt_xent import NTXentLoss 
import os

def train_simclr(config):
    set_seed()
    transform = get_simclr_augmentation()
    dataset = ImageFolder(config["data_dir"], transform=transform)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    backbone = get_resnet_backbone()
    model = SimCLR(backbone, config["out_dim"]).to(config["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = NTXentLoss(config["temperature"]).to(config["device"])

    for epoch in range(config["epochs"]):
        total_loss = 0
        model.train()
        for (x, _), in loader:
            x = x.to(config["device"])
            xi, xj = torch.split(x, x.shape[0] // 2)

            zi = model(xi)
            zj = model(xj)

            loss = criterion(zi, zj)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"[SimCLR] Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        if not os.path.exists(config["save_path"]):
            os.makedirs(config["save_path"])
        save_checkpoint(model, optimizer, epoch, os.path.join(config["save_path"], f"simclr_epoch_{epoch}.pt"))
