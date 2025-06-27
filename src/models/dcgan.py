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
    def __init__(self, noise_dim, checkpoint_path):
        super().__init__()
        self.noise_dim = noise_dim
        self.checkpoint_path = checkpoint_path
        self.transform = transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize image to 224x224
        ])

        self.inv_transform = None 
        self.fc = nn.Linear(noise_dim, 1024 * 16 * 16, bias=False)
        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
        

        self.conv_transpose1 = nn.ConvTranspose2d(1024, 512, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        
        self.conv_transpose2 = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        
        self.conv_transpose3 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv_transpose4 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        
        self.conv_transpose5 = nn.ConvTranspose2d(64, 3, kernel_size=5, stride=1, padding=2, bias=False)
        
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        batch_size = x.size(0)
        
        x = self.fc(x)
        x = self.LeakyReLU(x)
        x = x.view(batch_size, 1024, 16, 16)
        
        x = self.conv_transpose1(x)
        x = self.bn1(x)
        x = self.LeakyReLU(x)
        
        x = self.conv_transpose2(x)
        x = self.bn2(x)
        x = self.LeakyReLU(x)
        
        x = self.conv_transpose3(x)
        x = self.bn3(x)
        x = self.LeakyReLU(x)
        
        x = self.conv_transpose4(x)
        x = self.bn4(x)
        x = self.LeakyReLU(x)
        
        x = self.conv_transpose5(x)
        
        x = self.tanh(x)
        
        return x
    def load_ckpt(self, label):
        assert label in ['COVID', 'Lung_Opacity', 'Normal', 'Viral_Pneumonia'], \
            f"Label must be one of the following: ['COVID', 'Lung_Opacity', 'Normal', 'Viral_Pneumonia']"
        
        checkpoint_name = f"DCGAN_Checkpoint_{label}.pth"
        checkpoint_path = os.path.join(self.checkpoint_path, checkpoint_name)
        
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Remove 'module.' prefix from the keys if present
        state_dict = checkpoint.get('state_dict', checkpoint)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # Load the state_dict into the model
        self.load_state_dict(state_dict)
        print(f"Checkpoint for {label} loaded successfully!")
    def sample_images(self, label, save_dir, num_samples = 1000, batch_size = 20, device = "cpu"):
        assert label in ['COVID', 'Lung_Opacity', 'Normal', 'Viral_Pneumonia'], f"Label must be of the following: ['COVID', 'Lung_Opacity', 'Normal', 'Viral_Pneumonia']"
        checkpoint_name = f"DCGAN_Checkpoint_{label}.pth"
        checkpoint_path = os.path.join(self.checkpoint_path, checkpoint_name)
        # self.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        self.load_ckpt(label)
        print(f"Checkpoint of DCGAN for label {label} loaded successfully!")
        save_folder = os.path.join(save_dir, "DCGAN", label)
        os.makedirs(save_folder, exist_ok=True)
        with torch.no_grad():
            for steps in tqdm.tqdm(range(num_samples // batch_size)):
                z = torch.randn(batch_size, self.noise_dim).to(device)
                generated_images = self.transform(self(z).data)
                # gen_labels = torch.full((batch_size,), self.lookup_label[label], dtype=torch.long).squeeze(-1).to(device)
                # generated_images = self(z, gen_labels)
                images = generated_images.permute(0, 2, 3, 1).cpu().numpy()
                images = images * 0.5 * 255 + 0.5 * 255  
                images = images.astype(np.uint8)  
                st = batch_size * steps
                for i, img_array in enumerate(images):
                    img = Image.fromarray(img_array)
                    save_path = os.path.join(save_folder, f"image_{st + i}.png")      
                    img.save(save_path)   
        print(f"DCGAN Generated images for {label} images successfully!") 

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
        self.dropout1 = nn.Dropout2d(p=0.4)
        self.dropout2 = nn.Dropout2d(p=0.4)
        self.dropout3 = nn.Dropout2d(p=0.4)
        self.dropout4 = nn.Dropout2d(p=0.4)
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn5 = nn.BatchNorm2d(1024)

        self.fc_final = nn.Linear(1024 * 8 * 8, 1)
        

        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        batch_size = x.size(0)
        

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.LeakyReLU(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.LeakyReLU(x)
        x = self.dropout2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.LeakyReLU(x)
        x = self.dropout3(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.LeakyReLU(x)
        x = self.dropout4(x)
        x = self.conv5(x)
        x = self.LeakyReLU(x)
        
        # Flatten
        x = x.view(batch_size, -1)  # Flatten to (batch_size, 1024 * 8 * 8)
        
        # Final fully connected layer
        x = self.fc_final(x)
        x = self.sigmoid(x)
        
        return x

def get_model(configs):
    """
    Initialize and return the Generator and Discriminator models based on the provided configurations.
    
    Args:
        configs (dict): Configuration dictionary containing model parameters like 'noise_dim' and device info.
        
    Returns:
        generator (nn.Module): Initialized Generator model.
        discriminator (nn.Module): Initialized Discriminator model.
        transform: 
    """
    
    # Set device (CUDA if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT_PATH = configs.get('CHECKPOINT_PATH', 'ckpts')  # Checkpoint path if provided
    CHECKPOINT_PATH_DISCRIMINATOR = configs.get('CHECKPOINT_PATH_DISCRIMINATOR', None)
    # Initialize Generator model
    noise_dim = configs.get('noise_dim', 100)
    G = Generator(noise_dim, checkpoint_path=CHECKPOINT_PATH).to(device)
    D = Discriminator().to(device)
        
    return G, D