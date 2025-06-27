import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from utils.helper import set_seed
from models.classifier import LinearClassifier 
from models.resnet import get_resnet_backbone

def train_classifier(config, checkpoint_path):
    set_seed()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = ImageFolder(config["data_dir"], transform=transform)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    backbone = get_resnet_backbone()[0]
    backbone.load_state_dict(torch.load(checkpoint_path)['model_state'])
    backbone.fc = torch.nn.Identity()
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False

    model = LinearClassifier(backbone, num_classes=len(dataset.classes)).to(config["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(config["epochs"]):
        total_loss, correct = 0, 0
        model.train()
        for x, y in loader:
            x, y = x.to(config["device"]), y.to(config["device"])
            out = model(x)
            loss = criterion(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (out.argmax(dim=1) == y).sum().item()

        acc = correct / len(dataset)
        print(f"[Classifier] Epoch {epoch+1}, Loss: {total_loss:.4f}, Acc: {acc:.4f}")
