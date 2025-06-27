import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models.inception import inception_v3

from tqdm import tqdm
from PIL import Image
from scipy.stats import entropy
from src.utils.helper import get_path
from src.models import (
    acgan,
    dcgan
)
LABEL = ['COVID', 'Lung_Opacity', 'Normal', 'Viral_Pneumonia']
def inception_score(gen_img_paths, device = "cpu", batch_size=32, resize=True, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    class dataset(torch.utils.data.Dataset):
        def __init__(self, gen_img_paths = gen_img_paths):
            self.gen_img_paths = gen_img_paths
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            self.data = os.listdir(self.gen_img_paths)
        def __getitem__(self, index):
            img_path = os.path.join(self.gen_img_paths, self.data[index])
            img = Image.open(img_path).convert('RGB') 
            return self.transform(img)

        def __len__(self):
            return len(self.data)
    imgs = dataset(gen_img_paths=gen_img_paths)
    N = len(imgs)
    assert batch_size > 0
    assert N > batch_size

    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').to(device)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()

    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.to(device)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]
        with torch.no_grad():
            preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

# Example usage
def run_is_evaluate(model_name, device = "cpu", path = get_path()):
    G, D = import_model(model_name)
    abs_path = get_path()
    for label in LABEL:
        gen_img_paths = os.path.join(path, f'images/gen_images/{model_name}/{label}')
        print(gen_img_paths)
        if not os.path.exists(gen_img_paths) or len(os.listdir(gen_img_paths)) != 1000:
            G.sample_images(label, os.path.join(path, 'images/gen_images'), 1000, 25, device = device)
        is_score = inception_score(gen_img_paths, device=device)
        print(f"Inception Score statistics for {model_name} on label {label}: {is_score[0]}")
def import_model(model_name: str):
    """
    Dynamically import model based on model name.
    """
    if model_name == "ACGAN":
        G, D = acgan.get_model({})
    elif model_name == "DCGAN":
        G, D = dcgan.get_model({})
    else:
        raise ValueError(f"Model {model_name} is not supported.") 
    return G, D
def parse_args():
    parser = argparse.ArgumentParser(description="Compute FID score")
    parser.add_argument(
        '--model_name', 
        type=str, 
        choices=["ACGAN", "DCGAN"], 
        default="ACGAN", 
        help="The model type ('ACGAN' or 'DCGAN'). Default to 'ACGAN'."
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the evaluation on ('cuda' or 'cpu'). Default: auto-select based on availability."
    )
    return parser.parse_args()
if __name__ == "__main__":
    args = parse_args()
    run_is_evaluate(args.model_name, args.device)