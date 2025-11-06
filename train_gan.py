"""
DCGAN Training Script for Lego Brick Images
Trains a Deep Convolutional GAN on grayscale lego brick images (64x64)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import imageio.v2 as imageio
from pathlib import Path
from tqdm import tqdm


# ============================================================================
# Hyperparameters
# ============================================================================
EPOCHS = 300
LR = 2e-4
BETA_1 = 0.5
BETA_2 = 0.999
BATCH_SIZE = 128
LATENT_DIM = 100
DATASET_PATH = 'data/lego-brick-images/dataset'


# ============================================================================
# Helper Functions
# ============================================================================
def calculate_same_padding(kernel_size, stride=2):
    """Calculate padding for 'same' convolution"""
    padding = math.ceil((kernel_size - stride) / 2)
    return padding


# ============================================================================
# Dataset
# ============================================================================
class LegoDataset(Dataset):
    def __init__(self, dir_path, transform=None):
        dir_path = Path(dir_path)
        self.png_files = list(dir_path.rglob('*.png'))
        self.transform = transform

    def __len__(self):
        return len(self.png_files)

    def __getitem__(self, idx):
        filepath = self.png_files[idx]
        img = imageio.imread(filepath)
        img_t = self.transform(img)
        return img_t


# ============================================================================
# Discriminator Components
# ============================================================================
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, 
                bn_momentum=0.9, leaky_relu_slope=0.2, drop_prob=0.3,
                keep_bn=True):
        
        super().__init__()
        self.keep_bn = keep_bn
        padding = calculate_same_padding(kernel_size, stride)
        self.conv2d = nn.Conv2d(in_channels, out_channels, 
                    kernel_size=kernel_size, stride=stride, 
                    padding=padding, bias=False)
        if self.keep_bn:
            self.batchnorm = nn.BatchNorm2d(out_channels, momentum=bn_momentum)
        self.lrelu = nn.LeakyReLU(leaky_relu_slope)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.conv2d(x)
        if self.keep_bn:
            x = self.batchnorm(x)
        x = self.lrelu(x)
        x = self.dropout(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = ConvBlock(1, 64, keep_bn=False)
        self.block2 = ConvBlock(64, 128)
        self.block3 = ConvBlock(128, 256)
        self.block4 = ConvBlock(256, 512)
        self.final = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.final(x)
        x = x.view(-1, 1)
        return x


# ============================================================================
# Generator Components
# ============================================================================
class ConvTBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, padding=1):
        super().__init__()
        self.convt = nn.ConvTranspose2d(in_channels, out_channels, 
                        kernel_size=4, stride=stride, 
                        padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.convt(x)
        x = self.bn(x)
        x = self.lrelu(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = ConvTBlock(100, 512, stride=1, padding=0)
        self.block2 = ConvTBlock(512, 256)
        self.block3 = ConvTBlock(256, 128)
        self.block4 = ConvTBlock(128, 64)
        self.convt = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, 
                        padding=1, bias=False)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.convt(x)
        x = F.tanh(x)
        return x


# ============================================================================
# Main Training Function
# ============================================================================
def main():
    print("=" * 70)
    print("DCGAN Training - Lego Brick Images")
    print("=" * 70)
    
    # Device setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
    elif torch.mps.is_available():
        device = torch.device('mps')
        print("Using device: MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        print("Using device: CPU")
    
    # Data transforms
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=1),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Dataset and dataloader
    print(f"\nLoading dataset from: {DATASET_PATH}")
    dataset_path = Path(DATASET_PATH)
    lego_dataset = LegoDataset(dir_path=dataset_path, transform=img_transform)
    print(f"Total images: {len(lego_dataset)}")
    
    lego_dataloader = DataLoader(lego_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Batches per epoch: {len(lego_dataloader)}")
    
    # Initialize models
    print("\nInitializing models...")
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    # Count parameters
    gen_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    disc_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    print(f"Generator parameters: {gen_params:,}")
    print(f"Discriminator parameters: {disc_params:,}")
    
    # Optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=LR, betas=(BETA_1, BETA_2))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=LR, betas=(BETA_1, BETA_2))
    
    # Loss functions
    loss_fn_g = nn.BCEWithLogitsLoss()
    loss_fn_d = nn.BCEWithLogitsLoss()
    
    # TensorBoard and checkpoints
    writer = SummaryWriter(log_dir="runs/lego_gan")
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(exist_ok=True)
    print(f"\nTensorBoard logs: runs/lego_gan")
    print(f"Checkpoints directory: {checkpoints_dir}")
    
    # Training setup
    generator.train()
    discriminator.train()
    global_step = 0
    
    print(f"\nStarting training for {EPOCHS} epochs...")
    print("=" * 70)
    
    # Training loop
    for epoch in tqdm(range(EPOCHS), desc="Epochs", position=0):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        num_batches = 0
        
        for real_batch in lego_dataloader:
            real_batch = real_batch.to(device)
            batch_size = real_batch.size(0)
            latent_vecs = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
            
            # ================================================================
            # Train Discriminator
            # ================================================================
            with torch.no_grad():
                fake_batch = generator(latent_vecs)
            
            real_predictions = discriminator(real_batch)
            fake_predictions = discriminator(fake_batch)
            
            real_labels = torch.ones_like(real_predictions)
            fake_labels = torch.zeros_like(fake_predictions)
            
            real_loss = loss_fn_d(real_predictions, real_labels)
            fake_loss = loss_fn_d(fake_predictions, fake_labels)
            loss_d = (real_loss + fake_loss) / 2.0
            
            optimizer_d.zero_grad()
            for param in generator.parameters():
                param.requires_grad = False
            loss_d.backward()
            optimizer_d.step()
            for param in generator.parameters():
                param.requires_grad = True
            
            # ================================================================
            # Train Generator
            # ================================================================
            for param in discriminator.parameters():
                param.requires_grad = False
            
            optimizer_g.zero_grad()
            fake_batch = generator(latent_vecs)
            fake_predictions = discriminator(fake_batch)
            real_labels = torch.ones_like(fake_predictions)
            loss_g = loss_fn_g(fake_predictions, real_labels)
            loss_g.backward()
            optimizer_g.step()
            
            for param in discriminator.parameters():
                param.requires_grad = True
            
            # ================================================================
            # Logging
            # ================================================================
            writer.add_scalar("Loss/Discriminator", loss_d.item(), global_step)
            writer.add_scalar("Loss/Generator", loss_g.item(), global_step)
            global_step += 1
            
            epoch_d_loss += loss_d.item()
            epoch_g_loss += loss_g.item()
            num_batches += 1
        
        # Epoch summary
        avg_d_loss = epoch_d_loss / num_batches
        avg_g_loss = epoch_g_loss / num_batches
        print(f"Epoch [{epoch + 1}/{EPOCHS}] - D_loss: {avg_d_loss:.4f}, G_loss: {avg_g_loss:.4f}")
        
        # Generate sample images every 10 epochs
        if (epoch + 1) % 10 == 0:
            generator.eval()
            with torch.no_grad():
                sample_noise = torch.randn(5, LATENT_DIM, 1, 1, device=device)
                generated_samples = generator(sample_noise)
                generated_samples = (generated_samples + 1) / 2
            writer.add_images("Generated", generated_samples, global_step=epoch + 1)
            generator.train()
        
        # Save checkpoints every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(generator.state_dict(), checkpoints_dir / f"generator_epoch_{epoch + 1}.pt")
            torch.save(discriminator.state_dict(), checkpoints_dir / f"discriminator_epoch_{epoch + 1}.pt")
            print(f"  → Checkpoint saved at epoch {epoch + 1}")
    
    # Save final models
    print("\n" + "=" * 70)
    print("Training complete!")
    writer.close()
    torch.save(generator.state_dict(), checkpoints_dir / "generator_final.pt")
    torch.save(discriminator.state_dict(), checkpoints_dir / "discriminator_final.pt")
    print(f"Final models saved to {checkpoints_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
