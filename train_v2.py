import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from tqdm.auto import tqdm
import json
import gc
from typing import Dict, List, Tuple, Optional

# Diffusion imports
from diffusers import UNet2DModel, DDIMScheduler, DDIMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator

# Evaluation imports
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
import torchvision.utils as vutils

# Set up device and mixed precision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
class Config:
    # Data settings
    data_path = "./data"
    image_size = 128
    num_channels = 1

    # Model settings - IMPROVED
    num_train_timesteps = 1000
    num_inference_steps = 100  # Increased for better quality
    beta_start = 0.0001
    beta_end = 0.02
    beta_schedule = "scaled_linear"  # Better than linear
    
    # Prediction type - IMPORTANT FIX
    prediction_type = "epsilon"  # or "v_prediction"

    # Training settings - IMPROVED
    batch_size = 8  # Increased
    gradient_accumulation_steps = 1  # Simplified
    learning_rate = 1e-4
    num_epochs = 260  # More reasonable number
    warmup_steps = 500

    # Model architecture - SIMPLIFIED for better training
    block_out_channels = (128, 128, 256, 256, 512)  # Reduced complexity
    down_block_types = (
        "DownBlock2D", "DownBlock2D", "DownBlock2D",
        "DownBlock2D", "AttnDownBlock2D"
    )
    up_block_types = (
        "UpBlock2D", "AttnUpBlock2D", "UpBlock2D",
        "UpBlock2D", "UpBlock2D"
    )
    layers_per_block = 2
    attention_head_dim = 8

    # EMA settings
    ema_decay = 0.9999
    ema_update_every = 1  # Update every step

    # Evaluation settings
    eval_every = 10
    num_eval_samples = 16
    save_every = 20

    # Output settings
    output_dir = "/workspace/v2/pcos_ddim_outputs"
    checkpoint_dir = "/workspace/v2/pcos_ddim_checkpoints"  # Fixed typo

    # Class settings
    num_classes = 2
    class_names = ["notinfected", "infected"]

config = Config()

# Create output directories
os.makedirs(config.output_dir, exist_ok=True)
os.makedirs(config.checkpoint_dir, exist_ok=True)

print("Configuration loaded successfully!")
print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")

class PCOSDataset(Dataset):
    def __init__(self, data_path: str, split: str = "train", transform=None):
        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform

        # Load image paths and labels
        self.samples = []
        split_path = self.data_path / split

        for class_idx, class_name in enumerate(config.class_names):
            class_path = split_path / class_name
            if class_path.exists():
                for img_path in class_path.glob("*.jpg"):
                    self.samples.append((str(img_path), class_idx))
                for img_path in class_path.glob("*.png"):
                    self.samples.append((str(img_path), class_idx))

        print(f"Found {len(self.samples)} images in {split} set")

        # Print class distribution
        class_counts = {}
        for _, label in self.samples:
            class_counts[label] = class_counts.get(label, 0) + 1

        for class_idx, count in class_counts.items():
            print(f"Class {config.class_names[class_idx]}: {count} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert('L')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('L', (config.image_size, config.image_size), 0)

        if self.transform:
            image = self.transform(image)

        return image, label

# IMPROVED transforms with better normalization
train_transform = transforms.Compose([
    transforms.Resize((config.image_size, config.image_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),  # Added rotation
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # [-1, 1] range
])

eval_transform = transforms.Compose([
    transforms.Resize((config.image_size, config.image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Create datasets and loaders
train_dataset = PCOSDataset(config.data_path, "train", train_transform)
test_dataset = PCOSDataset(config.data_path, "test", eval_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True  # Important for consistent batch sizes
)

test_loader = DataLoader(
    test_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

print("Datasets created successfully!")

class ConditionalUNet2D(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.unet = UNet2DModel(
            sample_size=config.image_size,
            in_channels=config.num_channels,
            out_channels=config.num_channels,
            layers_per_block=config.layers_per_block,
            block_out_channels=config.block_out_channels,
            down_block_types=config.down_block_types,
            up_block_types=config.up_block_types,
            attention_head_dim=config.attention_head_dim,
            num_class_embeds=num_classes,
        )
        self.num_classes = num_classes

    def forward(self, sample, timestep, class_labels=None):
        return self.unet(sample, timestep, class_labels=class_labels).sample

# Initialize model with IMPROVED scheduler settings
model = ConditionalUNet2D(config.num_classes).to(device)

scheduler = DDIMScheduler(
    num_train_timesteps=config.num_train_timesteps,
    beta_start=config.beta_start,
    beta_end=config.beta_end,
    beta_schedule=config.beta_schedule,
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
    prediction_type=config.prediction_type,  # Explicit prediction type
)

# IMPROVED optimizer settings
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate,
    betas=(0.9, 0.999),  # Standard betas
    weight_decay=0.01,   # Increased weight decay
    eps=1e-08,
)

lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.warmup_steps,
    num_training_steps=len(train_loader) * config.num_epochs,
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

class EMA:
    def __init__(self, model, decay=0.9999, update_every=1):
        self.model = model
        self.decay = decay
        self.update_every = update_every
        self.step = 0

        self.ema_model = ConditionalUNet2D(config.num_classes).to(device)
        self.ema_model.load_state_dict(model.state_dict())
        self.ema_model.eval()

        for param in self.ema_model.parameters():
            param.requires_grad = False

    def update(self):
        self.step += 1
        if self.step % self.update_every != 0:
            return

        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1 - self.decay)

# IMPROVED loss computation with better noise handling
def compute_loss(model, scheduler, batch, device):
    images, class_labels = batch
    images = images.to(device)
    class_labels = class_labels.to(device)

    # Sample random timesteps
    timesteps = torch.randint(
        0, scheduler.config.num_train_timesteps,
        (images.shape[0],), device=device
    ).long()

    # Add noise to images
    noise = torch.randn_like(images)
    noisy_images = scheduler.add_noise(images, noise, timesteps)

    # Predict noise
    noise_pred = model(noisy_images, timesteps, class_labels)

    # Compute MSE loss
    loss = F.mse_loss(noise_pred, noise, reduction='mean')
    
    return loss

# IMPROVED generation function with better denoising
@torch.no_grad()
def generate_samples(model, scheduler, num_samples=16, class_label=None, num_inference_steps=None):
    model.eval()

    if num_inference_steps is None:
        num_inference_steps = config.num_inference_steps

    # Set scheduler for inference
    scheduler.set_timesteps(num_inference_steps, device=device)

    # Initialize random noise
    shape = (num_samples, config.num_channels, config.image_size, config.image_size)
    sample = torch.randn(shape, device=device)

    # Class labels
    if class_label is not None:
        class_labels = torch.full((num_samples,), class_label, device=device, dtype=torch.long)
    else:
        # Generate equal numbers of each class
        class_labels = []
        for i in range(num_samples):
            class_labels.append(i % config.num_classes)
        class_labels = torch.tensor(class_labels, device=device, dtype=torch.long)

    # IMPROVED denoising loop
    for i, t in enumerate(tqdm(scheduler.timesteps, desc="Generating", leave=False)):
        # Expand timestep for batch
        t_batch = t.expand(sample.shape[0])
        
        # Predict noise
        with torch.no_grad():
            noise_pred = model(sample, t_batch, class_labels)

        # Compute previous sample
        sample = scheduler.step(noise_pred, t, sample).prev_sample

        # Optional: Add some noise back for better quality (only for early steps)
        if i < len(scheduler.timesteps) * 0.8:  # First 80% of steps
            sample = sample + torch.randn_like(sample) * 0.01

    # Clamp and normalize for visualization
    sample = torch.clamp(sample, -1.0, 1.0)
    sample = (sample + 1.0) / 2.0  # Convert from [-1,1] to [0,1]

    return sample, class_labels

def generate_and_save_samples(model, scheduler, epoch, num_samples=16):
    """Generate samples and save them with improved visualization"""
    samples, class_labels = generate_samples(model, scheduler, num_samples)
    
    # Create directory for this epoch
    epoch_dir = Path(config.output_dir) / f"epoch_{epoch}"
    epoch_dir.mkdir(exist_ok=True)
    
    # Save individual samples
    for i, (sample, label) in enumerate(zip(samples, class_labels)):
        sample_path = epoch_dir / f"sample_{i}_class_{config.class_names[label]}.png"
        vutils.save_image(sample, sample_path)
    
    # Create and save grid
    grid = vutils.make_grid(samples, nrow=4, normalize=False, pad_value=1.0)
    grid_path = epoch_dir / "samples_grid.png"
    vutils.save_image(grid, grid_path)

    # Display results
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy().squeeze(), cmap='gray')
    plt.axis('off')
    plt.title(f"Generated Samples - Epoch {epoch}")
    plt.tight_layout()
    plt.savefig(epoch_dir / "samples_display.png", dpi=150, bbox_inches='tight')
    plt.show()

    return samples

# Initialize EMA and metrics
ema = EMA(model, config.ema_decay, config.ema_update_every)

class MetricsTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.losses = []
        self.epoch_losses = []

    def update(self, loss):
        self.losses.append(loss)

    def end_epoch(self):
        if self.losses:
            epoch_loss = np.mean(self.losses)
            self.epoch_losses.append(epoch_loss)
            self.losses = []
            return epoch_loss
        return 0.0

    def plot_losses(self):
        if len(self.epoch_losses) > 0:
            plt.figure(figsize=(12, 6))
            plt.plot(self.epoch_losses)
            plt.title('Training Loss Over Time')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
            plt.savefig(f"{config.output_dir}/training_loss.png", dpi=150, bbox_inches='tight')
            plt.show()

metrics_tracker = MetricsTracker()

def save_checkpoint(model, optimizer, lr_scheduler, ema, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'ema_model_state_dict': ema.ema_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'loss': loss,
        'config': config.__dict__,
    }, path)

# DEBUGGING: Test data loading and model forward pass
print("\n=== DEBUGGING ===")
print("Testing data loading...")
test_batch = next(iter(train_loader))
test_images, test_labels = test_batch
print(f"Batch shape: {test_images.shape}")
print(f"Image range: [{test_images.min():.3f}, {test_images.max():.3f}]")
print(f"Labels: {test_labels}")

print("\nTesting model forward pass...")
test_timesteps = torch.randint(0, 1000, (test_images.shape[0],), device=device)
with torch.no_grad():
    test_images = test_images.to(device)
    test_labels = test_labels.to(device)
    output = model(test_images, test_timesteps, test_labels)
    print(f"Model output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

# Test generation before training
print("\nTesting generation (before training)...")
test_samples, test_class_labels = generate_samples(model, scheduler, num_samples=4)
print(f"Generated samples shape: {test_samples.shape}")
print(f"Generated range: [{test_samples.min():.3f}, {test_samples.max():.3f}]")

# Display test samples
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for i in range(4):
    axes[i].imshow(test_samples[i].squeeze().cpu().numpy(), cmap='gray')
    axes[i].set_title(f"Class: {config.class_names[test_class_labels[i]]}")
    axes[i].axis('off')
plt.suptitle("Generated Samples (Before Training)")
plt.tight_layout()
plt.savefig(f"{config.output_dir}/test_generation_before_training.png", dpi=150)
plt.show()

print("=== DEBUG COMPLETE ===\n")

# IMPROVED training loop with better monitoring
def train_epoch(model, train_loader, optimizer, lr_scheduler, scheduler, ema, metrics_tracker, epoch):
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
    
    for step, batch in enumerate(progress_bar):
        # Compute loss
        loss = compute_loss(model, scheduler, batch, device)
        
        # Scale loss for gradient accumulation
        loss = loss / config.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Gradient accumulation step
        if (step + 1) % config.gradient_accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # Update EMA
            ema.update()
        
        # Track metrics
        actual_loss = loss.item() * config.gradient_accumulation_steps
        metrics_tracker.update(actual_loss)
        total_loss += actual_loss
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{actual_loss:.4f}",
            'avg_loss': f"{total_loss / (step + 1):.4f}",
            'lr': f"{lr_scheduler.get_last_lr()[0]:.2e}"
        })
        
        # Memory cleanup
        if step % 50 == 0:
            torch.cuda.empty_cache()
    
    return total_loss / num_batches

# Training loop with better checkpointing
print("Starting improved training...")
best_loss = float('inf')

for epoch in range(config.num_epochs):
    # Training
    avg_loss = train_epoch(
        model, train_loader, optimizer, lr_scheduler,
        scheduler, ema, metrics_tracker, epoch
    )
    
    epoch_loss = metrics_tracker.end_epoch()
    print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")
    
    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        save_checkpoint(
            model, optimizer, lr_scheduler, ema, epoch, avg_loss,
            f"{config.checkpoint_dir}/best_model.pth"
        )
        print(f"New best model saved! Loss: {best_loss:.4f}")
    
    # Regular evaluation and sample generation
    if (epoch + 1) % config.eval_every == 0:
        print(f"Generating samples at epoch {epoch+1}...")
        generate_and_save_samples(ema.ema_model, scheduler, epoch+1)
    
    # Regular checkpoint saving
    if (epoch + 1) % config.save_every == 0:
        save_checkpoint(
            model, optimizer, lr_scheduler, ema, epoch, avg_loss,
            f"{config.checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth"
        )
    
    # Plot training progress
    if (epoch + 1) % 20 == 0:
        metrics_tracker.plot_losses()

print("Training completed!")

# Final checkpoint
save_checkpoint(
    model, optimizer, lr_scheduler, ema, config.num_epochs-1, avg_loss,
    f"{config.checkpoint_dir}/final_model.pth"
)

# Final sample generation
print("Generating final samples...")
final_samples = generate_and_save_samples(ema.ema_model, scheduler, "final", num_samples=32)

print("All done! Check the output directory for results.")