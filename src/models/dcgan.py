import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim):
        super().__init__()
        self.noise_dim = noise_dim
        
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
        
        # Process noise
        x = self.fc(x)
        x = self.LeakyReLU(x)
        x = x.view(batch_size, 1024, 16, 16)
        
        # ConvTranspose2d layers
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