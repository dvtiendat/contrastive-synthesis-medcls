import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image

class Generator(nn.Module):
    def __init__(self, n_classes, latent_dim, img_size, out_channels):
        super(Generator, self).__init__()
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.out_channels = out_channels

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

def get_model(configs):
    # Unpack values from configs dictionary
    LATENT_DIM = configs.get('LATENT_DIM', 128)  # Default value is 128 if not provided
    IMG_SIZE = configs.get('IMG_SIZE', 224)  # Default value is 224 if not provided
    NUM_CLASSES = configs.get('NUM_CLASSES', 4)  # Default value is 4 if not provided
    OUT_CHANNELS = configs.get('OUT_CHANNELS', 1)  # Default value is 1 if not provided
    CHECKPOINT_PATH = configs.get('CHECKPOINT_PATH', None)  # Checkpoint path if provided
    # Lookup labels (can also be passed in configs if needed)
    LOOKUP_LABEL = configs.get('LOOKUP_LABEL', {
        "COVID": 0,
        "Lung_Opacity": 1,
        "Normal": 2,
        "Viral_Pneumonia": 3,
    })

    INV_LOOKUP = {v: k for k, v in LOOKUP_LABEL.items()}

    # Initialize the Generator and Discriminator with the provided configuration
    G = Generator(NUM_CLASSES, LATENT_DIM, IMG_SIZE, OUT_CHANNELS)
    D = Discriminator(NUM_CLASSES, IMG_SIZE, OUT_CHANNELS)
    if CHECKPOINT_PATH:
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
        G.load_state_dict(ckpt['generator_state_dict'])
        D.load_state_dict(ckpt['discriminator_state_dict'])
        print(f"Checkpoint loaded from {CHECKPOINT_PATH}")
        
    return G, D, INV_LOOKUP
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