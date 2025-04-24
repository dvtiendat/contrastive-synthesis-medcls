import os
import cv2
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image

class Generator(nn.Module):
    def __init__(
            self, 
            n_classes, 
            latent_dim, 
            img_size, 
            lookup_label,
            out_channels
        ):
        super(Generator, self).__init__()
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.out_channels = out_channels
        self.lookup_label = lookup_label
        self.inv_lookup = {v: k for k, v in self.lookup_label.items()}

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),       # Resize to 224x224
            transforms.ToTensor(),               # Convert image to tensor
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize the channel
        ])
        self.inv_transform = transforms.Normalize(
            mean=[-0.5 / 0.5],  # Undo mean shift
            std=[1 / 0.5]  # Undo scaling
        )
        self.init_size = self.img_size // 4  # Initial size before upsampling
        # (224 * (img_size // 4) ** 2)
        self.l1 = nn.Sequential(nn.Linear(self.latent_dim, 128 * self.init_size ** 2))
        self.label_emb = nn.Embedding(self.n_classes, self.latent_dim)
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, self.out_channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

    def sample_images(self, label, save_dir, num_samples = 1000, batch_size = 20, device = "cpu"):
        assert label in self.lookup_label.keys(), f"Label must be of the following: {list(self.lookup_label.keys())}"
        save_folder = os.path.join(save_dir, "ACGAN", label)
        os.makedirs(save_folder, exist_ok=True)
        with torch.no_grad():
            for steps in tqdm.tqdm(range(num_samples // batch_size)):
                z = torch.FloatTensor(np.random.normal(0, 1, (batch_size, self.latent_dim))).to(device)
                gen_labels = torch.full((batch_size,), self.lookup_label[label], dtype=torch.long).squeeze(-1).to(device)
                generated_images = self(z, gen_labels)
                images = self.inv_transform(generated_images).permute(0, 2, 3, 1).cpu().numpy()
                st = batch_size * steps
                for i, img_array in enumerate(images):
                    if img_array.max() <= 1.0:
                        img_array = (img_array * 255).astype(np.uint8)
                    else:
                        img_array = img_array.astype(np.uint8)
                    img = Image.fromarray(gray_to_rgb_ndarray(img_array))
                    save_path = os.path.join(save_folder, f"image_{st + i}.png")      
                    img.save(save_path)   
        print("ACGAN Generated successfully!") 

class Discriminator(nn.Module):
    def __init__(self, n_classes, img_size, out_channels):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block
        self.out_channels = out_channels
        self.n_classes = n_classes
        self.img_size = img_size
        self.conv_blocks = nn.Sequential(
            *discriminator_block(self.out_channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = self.img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, self.n_classes))

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.reshape(out.shape[0], -1)
        validity = self.adv_layer(out).squeeze(-1)
        label = self.aux_layer(out).squeeze(-1)

        return validity, label

def gray_to_rgb_ndarray(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:  # (H, W)
        return np.stack([img]*3, axis=-1)  # (H, W, 3)
    elif len(img.shape) == 3 and img.shape[0] == 1:  # (1, H, W)
        return np.repeat(img, 3, axis=0)  # (3, H, W)
    elif len(img.shape) == 3 and img.shape[2] == 1:  # (H, W, 1)
        return np.repeat(img, 3, axis=2)  # (H, W, 3)
    else:
        return img  # Already 3 channels

def get_model(configs):
    LATENT_DIM = configs.get('LATENT_DIM', 128) 
    IMG_SIZE = configs.get('IMG_SIZE', 224) 
    NUM_CLASSES = configs.get('NUM_CLASSES', 4) 
    OUT_CHANNELS = configs.get('OUT_CHANNELS', 1)
    CHECKPOINT_PATH = configs.get('CHECKPOINT_PATH', 'ckpts/ACGAN_Checkpoint.pt') 
    LOOKUP_LABEL = configs.get('LOOKUP_LABEL', {
        "COVID": 0,
        "Lung_Opacity": 1,
        "Normal": 2,
        "Viral_Pneumonia": 3,
    })

    INV_LOOKUP = {v: k for k, v in LOOKUP_LABEL.items()}

    # Initialize the Generator and Discriminator with the provided configuration
    G = Generator(NUM_CLASSES, LATENT_DIM, IMG_SIZE, LOOKUP_LABEL, OUT_CHANNELS)
    D = Discriminator(NUM_CLASSES, IMG_SIZE, OUT_CHANNELS)
    if CHECKPOINT_PATH:
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
        G.load_state_dict(ckpt)
        # D.load_state_dict(ckpt['discriminator_state_dict'])
        print(f"Checkpoint loaded from {CHECKPOINT_PATH}")
        
    return G, D
'''
configs = {
    'LATENT_DIM': 128,
    'IMG_SIZE': 224,
    'NUM_CLASSES': 4,
    'OUT_CHANNELS': 1,
    'CHECKPOINT_PATH': '/path/to/your/model_checkpoint/ACGAN_Checkpoint_Deeper_1500.pt',
    'LOOKUP_LABEL': {
        "COVID": 0,
        "Lung_Opacity": 1,
        "Normal": 2,
        "Viral_Pneumonia": 3,
    }
}

G, D, INV_LOOKUP = get_model(configs)
'''