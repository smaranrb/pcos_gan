"""
PCOS Diffusion Model Evaluation Script
=====================================
This script loads a trained PCOS diffusion model and performs comprehensive evaluation,
including FID scores, Inception scores, and creates a Hugging Face pipeline.

Usage:
    python evaluate_pcos_model.py --checkpoint_path /path/to/model.pth --data_path /path/to/data
"""

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
import argparse
import json
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Optional

# Diffusion imports
from diffusers import UNet2DModel, DDIMScheduler, DDIMPipeline
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
import torchvision.utils as vutils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class Config:
    """Configuration class - will be loaded from checkpoint"""
    def __init__(self):
        # Default values - these will be overridden when loading checkpoint
        self.data_path = "./data"
        self.image_size = 128
        self.num_channels = 1
        self.num_train_timesteps = 1000
        self.num_inference_steps = 100
        self.beta_start = 0.0001
        self.beta_end = 0.02
        self.beta_schedule = "scaled_linear"
        self.prediction_type = "epsilon"
        self.batch_size = 8
        self.block_out_channels = (128, 128, 256, 256, 512)
        self.down_block_types = (
            "DownBlock2D", "DownBlock2D", "DownBlock2D",
            "DownBlock2D", "AttnDownBlock2D"
        )
        self.up_block_types = (
            "UpBlock2D", "AttnUpBlock2D", "UpBlock2D",
            "UpBlock2D", "UpBlock2D"
        )
        self.layers_per_block = 2
        self.attention_head_dim = 8
        self.num_classes = 2
        self.class_names = ["notinfected", "infected"]

class PCOSDataset(Dataset):
    def __init__(self, data_path: str, split: str = "test", transform=None, class_names=None):
        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform
        self.class_names = class_names or ["notinfected", "infected"]

        # Load image paths and labels
        self.samples = []
        split_path = self.data_path / split

        for class_idx, class_name in enumerate(self.class_names):
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
            print(f"Class {self.class_names[class_idx]}: {count} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert('L')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a default image if loading fails
            image = Image.new('L', (128, 128), 0)

        if self.transform:
            image = self.transform(image)

        return image, label

class ConditionalUNet2D(nn.Module):
    def __init__(self, config):
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
            num_class_embeds=config.num_classes,
        )
        self.num_classes = config.num_classes

    def forward(self, sample, timestep, class_labels=None):
        return self.unet(sample, timestep, class_labels=class_labels).sample

def load_model_and_config(checkpoint_path):
    """Load model and configuration from checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load configuration
    config = Config()
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # Initialize model
    model = ConditionalUNet2D(config).to(device)
    
    # Load model weights - try both regular model and EMA model
    if 'ema_model_state_dict' in checkpoint:
        print("Loading EMA model weights...")
        model.load_state_dict(checkpoint['ema_model_state_dict'])
    elif 'model_state_dict' in checkpoint:
        print("Loading regular model weights...")
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise KeyError("No model state dict found in checkpoint")
    
    # Initialize scheduler
    scheduler = DDIMScheduler(
        num_train_timesteps=config.num_train_timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        beta_schedule=config.beta_schedule,
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
        prediction_type=getattr(config, 'prediction_type', 'epsilon'),
    )
    
    print(f"Model loaded successfully!")
    print(f"Epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"Loss: {checkpoint.get('loss', 'Unknown')}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, scheduler, config

@torch.no_grad()
def generate_samples(model, scheduler, config, num_samples=16, class_label=None, num_inference_steps=None):
    """Generate samples using the trained model."""
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

    # Denoising loop
    for i, t in enumerate(tqdm(scheduler.timesteps, desc="Generating", leave=False)):
        # Expand timestep for batch
        t_batch = t.expand(sample.shape[0])
        
        # Predict noise
        with torch.no_grad():
            noise_pred = model(sample, t_batch, class_labels)

        # Compute previous sample
        sample = scheduler.step(noise_pred, t, sample).prev_sample

    # Clamp and normalize for visualization
    sample = torch.clamp(sample, -1.0, 1.0)
    sample = (sample + 1.0) / 2.0  # Convert from [-1,1] to [0,1]

    return sample, class_labels

@torch.no_grad()
def compute_fid_score(model, scheduler, config, real_loader, num_generated=1000):
    """Compute FID score between real and generated images."""
    model.eval()

    print(f"Computing FID score with {num_generated} samples...")
    
    # Initialize FID metric
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    # Process real images
    real_count = 0
    print("Processing real images...")
    for batch_images, _ in tqdm(real_loader, desc="Real images"):
        if real_count >= num_generated:
            break

        batch_images = batch_images.to(device)
        # Convert grayscale to RGB for InceptionV3
        if batch_images.shape[1] == 1:
            batch_images = batch_images.repeat(1, 3, 1, 1)

        # Denormalize to [0, 1]
        batch_images = (batch_images + 1) / 2
        batch_images = torch.clamp(batch_images, 0, 1)

        fid.update(batch_images, real=True)
        real_count += batch_images.shape[0]

    # Generate fake images
    generated_count = 0
    batch_size = 8  # Smaller batch for generation

    print("Generating synthetic images...")
    while generated_count < num_generated:
        current_batch_size = min(batch_size, num_generated - generated_count)
        generated_images, _ = generate_samples(model, scheduler, config, current_batch_size)

        # Convert grayscale to RGB
        if generated_images.shape[1] == 1:
            generated_images = generated_images.repeat(1, 3, 1, 1)

        generated_images = torch.clamp(generated_images, 0, 1)
        fid.update(generated_images, real=False)
        generated_count += current_batch_size

    # Compute FID
    try:
        fid_score = fid.compute()
        return fid_score.item()
    except Exception as e:
        print(f"Error computing FID: {e}")
        return None

@torch.no_grad()
def compute_inception_score(model, scheduler, config, num_samples=1000):
    """Compute Inception Score for generated images."""
    model.eval()

    print(f"Computing Inception Score with {num_samples} samples...")
    
    # Initialize IS metric
    inception_score = InceptionScore(normalize=True).to(device)

    generated_count = 0
    batch_size = 8

    while generated_count < num_samples:
        current_batch_size = min(batch_size, num_samples - generated_count)
        generated_images, _ = generate_samples(model, scheduler, config, current_batch_size)

        # Convert grayscale to RGB
        if generated_images.shape[1] == 1:
            generated_images = generated_images.repeat(1, 3, 1, 1)

        generated_images = torch.clamp(generated_images, 0, 1)
        inception_score.update(generated_images)
        generated_count += current_batch_size

    # Compute IS
    try:
        is_mean, is_std = inception_score.compute()
        return is_mean.item(), is_std.item()
    except Exception as e:
        print(f"Error computing Inception Score: {e}")
        return None, None

def evaluate_class_conditional_generation(model, scheduler, config, output_dir):
    """Evaluate class-conditional generation quality."""
    print("Evaluating class-conditional generation...")
    
    class_results = {}
    
    for class_idx, class_name in enumerate(config.class_names):
        print(f"Generating samples for class: {class_name}")
        samples, _ = generate_samples(model, scheduler, config, num_samples=16, class_label=class_idx)

        # Save class-specific samples
        grid = vutils.make_grid(samples, nrow=4, normalize=False, pad_value=1)
        grid_path = os.path.join(output_dir, f"class_{class_name}_samples.png")
        vutils.save_image(grid, grid_path)

        # Calculate basic statistics
        sample_mean = samples.mean().item()
        sample_std = samples.std().item()
        sample_min = samples.min().item()
        sample_max = samples.max().item()
        
        class_results[class_name] = {
            'mean': sample_mean,
            'std': sample_std,
            'min': sample_min,
            'max': sample_max,
            'samples_saved': grid_path
        }

        # Display
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy().squeeze(), cmap='gray')
        plt.axis('off')
        plt.title(f"Generated {class_name} samples")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"class_{class_name}_display.png"), 
                   dpi=150, bbox_inches='tight')
        plt.show()

    return class_results

def create_huggingface_pipeline(model, scheduler, config, output_dir):
    """Create and save a Hugging Face pipeline for easy inference."""
    print("Creating Hugging Face pipeline...")
    
    try:
        # Create pipeline
        pipeline = DDIMPipeline(
            unet=model.unet,  # Use the underlying UNet
            scheduler=scheduler
        )
        
        # Save pipeline
        pipeline_path = os.path.join(output_dir, "huggingface_pipeline")
        os.makedirs(pipeline_path, exist_ok=True)
        pipeline.save_pretrained(pipeline_path)
        
        print(f"Pipeline saved to: {pipeline_path}")
        
        # Test the saved pipeline
        print("Testing saved pipeline...")
        loaded_pipeline = DDIMPipeline.from_pretrained(pipeline_path)
        loaded_pipeline = loaded_pipeline.to(device)
        
        # Generate test samples
        test_results = {}
        for class_idx, class_name in enumerate(config.class_names):
            print(f"Testing pipeline with {class_name} samples...")
            
            # Generate samples (note: standard DDIM pipeline doesn't support class conditioning)
            # This is a limitation - you'd need a custom pipeline for class conditioning
            test_images = loaded_pipeline(
                batch_size=4,
                num_inference_steps=config.num_inference_steps // 2,  # Faster for testing
                generator=torch.Generator(device=device).manual_seed(42 + class_idx)
            ).images
            
            # Save test results
            if isinstance(test_images, list):
                # Convert PIL images to tensor for saving
                test_tensor = torch.stack([
                    transforms.ToTensor()(img) for img in test_images
                ])
            else:
                test_tensor = test_images
            
            grid = vutils.make_grid(test_tensor, nrow=2, normalize=False, pad_value=1)
            test_path = os.path.join(output_dir, f"pipeline_test_{class_name}.png")
            vutils.save_image(grid, test_path)
            
            test_results[class_name] = test_path
        
        return pipeline_path, test_results
        
    except Exception as e:
        print(f"Error creating pipeline: {e}")
        return None, None

def comprehensive_evaluation(checkpoint_path, data_path, output_dir, num_eval_samples=500):
    """Run comprehensive evaluation of the trained model."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and configuration
    model, scheduler, config = load_model_and_config(checkpoint_path)
    
    # Update data path in config
    config.data_path = data_path
    
    # Create test dataset and loader
    eval_transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    test_dataset = PCOSDataset(data_path, "test", eval_transform, config.class_names)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"\n{'='*70}")
    print("🔍 STARTING COMPREHENSIVE EVALUATION")
    print(f"{'='*70}")
    
    evaluation_results = {}
    
    # 1. Basic sample generation
    print("\n1. Generating basic samples...")
    basic_samples, sample_labels = generate_samples(model, scheduler, config, num_samples=16)
    grid = vutils.make_grid(basic_samples, nrow=4, normalize=False, pad_value=1)
    vutils.save_image(grid, os.path.join(output_dir, "basic_samples.png"))
    
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy().squeeze(), cmap='gray')
    plt.axis('off')
    plt.title("Generated Samples Overview")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "basic_samples_display.png"), dpi=150, bbox_inches='tight')
    plt.show()
    
    # 2. Class-conditional evaluation
    print("\n2. Evaluating class-conditional generation...")
    class_results = evaluate_class_conditional_generation(model, scheduler, config, output_dir)
    evaluation_results['class_conditional'] = class_results
    
    # 3. FID Score (if enough data)
    if len(test_dataset) >= 50:
        print("\n3. Computing FID score...")
        fid_score = compute_fid_score(model, scheduler, config, test_loader, 
                                     min(num_eval_samples, len(test_dataset)))
        evaluation_results['fid_score'] = fid_score
        if fid_score:
            print(f"FID Score: {fid_score:.4f}")
    else:
        print("\n3. Skipping FID score (insufficient test data)")
        evaluation_results['fid_score'] = None
    
    # 4. Inception Score
    print("\n4. Computing Inception Score...")
    is_mean, is_std = compute_inception_score(model, scheduler, config, num_eval_samples)
    evaluation_results['inception_score'] = {'mean': is_mean, 'std': is_std}
    if is_mean:
        print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")
    
    # 5. Create Hugging Face pipeline
    print("\n5. Creating Hugging Face pipeline...")
    pipeline_path, pipeline_test_results = create_huggingface_pipeline(model, scheduler, config, output_dir)
    evaluation_results['pipeline'] = {
        'path': pipeline_path,
        'test_results': pipeline_test_results
    }
    
    # 6. Save comprehensive results
    print("\n6. Saving evaluation results...")
    
    final_results = {
        'model_info': {
            'checkpoint_path': checkpoint_path,
            'model_parameters': sum(p.numel() for p in model.parameters()),
            'image_size': config.image_size,
            'num_classes': config.num_classes,
            'class_names': config.class_names,
        },
        'evaluation_results': evaluation_results,
        'config': {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
    }
    
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    # 7. Print final summary
    print(f"\n{'='*70}")
    print("✅ EVALUATION COMPLETED!")
    print(f"{'='*70}")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📊 Detailed results: {results_path}")
    if pipeline_path:
        print(f"🤗 HuggingFace pipeline: {pipeline_path}")
    print(f"\n📈 SUMMARY:")
    if evaluation_results.get('fid_score'):
        print(f"• FID Score: {evaluation_results['fid_score']:.4f}")
    if evaluation_results.get('inception_score', {}).get('mean'):
        is_data = evaluation_results['inception_score']
        print(f"• Inception Score: {is_data['mean']:.4f} ± {is_data['std']:.4f}")
    print(f"• Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"• Classes Evaluated: {', '.join(config.class_names)}")
    print(f"\n🎉 Your PCOS diffusion model evaluation is complete!")
    
    return final_results

def main():
    parser = argparse.ArgumentParser(description="Evaluate PCOS Diffusion Model")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to the model checkpoint (.pth file)")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to the dataset directory")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                       help="Directory to save evaluation results")
    parser.add_argument("--num_eval_samples", type=int, default=500,
                       help="Number of samples to generate for evaluation metrics")
    
    args = parser.parse_args()
    
    # Run evaluation
    results = comprehensive_evaluation(
        checkpoint_path=args.checkpoint_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_eval_samples=args.num_eval_samples
    )
    
    return results

if __name__ == "__main__":
    # Example usage if running directly (without command line args)
    # Uncomment and modify these lines to run directly
    
    checkpoint_path = "/workspace/v2/pcos_ddim_checkpoints/best_model.pth"
    data_path = "./data"
    output_dir = "./v2/evaluation_results"
    # 
    results = comprehensive_evaluation(checkpoint_path, data_path, output_dir)
    
    # main()