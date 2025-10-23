"""
Main script for DARTS architecture search on PCOS dataset
"""

import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
from configs.search_config import SearchConfig
from models.network import Network
from utils.data_loader import PCOSDataLoader
from utils.metrics import AverageMeter, accuracy, MetricsCalculator
from search.architect import Architect


class SearchTrainer:
    """DARTS Architecture Search Trainer"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(f'cuda:{config.gpu}' if torch.cuda.is_available() else 'cpu')
        
        # Set random seed
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        print("=" * 70)
        print("DARTS ARCHITECTURE SEARCH FOR PCOS DETECTION")
        print("=" * 70)
        print(f"\nDevice: {self.device}")
        print(f"Seed: {config.seed}")
        
        # Load data
        self._load_data()
        
        # Build model
        self._build_model()
        
        # Setup training
        self._setup_training()
        
    def _load_data(self):
        """Load PCOS dataset"""
        print("\nLoading dataset...")
        
        self.data_loader = PCOSDataLoader(
            data_path=self.config.data_path,
            test_path=self.config.test_path,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            val_split=0.5,
            image_size=224,
            use_augmentation=False  # Minimal augmentation for search
        )
        
        self.train_loader = self.data_loader.get_train_loader()
        self.val_loader = self.data_loader.get_val_loader()
        self.test_loader = self.data_loader.get_test_loader()
        
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"Test batches: {len(self.test_loader)}")
        
    def _build_model(self):
        """Build DARTS network"""
        print("\nBuilding model...")
        
        self.model = Network(
            C=self.config.init_channels,
            num_classes=2,
            layers=self.config.layers,
            steps=self.config.nodes
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        arch_params = sum(p.numel() for p in self.model.arch_parameters())
        weight_params = total_params - arch_params
        
        print(f"Total parameters: {total_params:,}")
        print(f"Weight parameters: {weight_params:,}")
        print(f"Architecture parameters: {arch_params:,}")
        
    def _setup_training(self):
        """Setup training components"""
        print("\nSetting up training...")
        
        # Loss function with class weights
        class_weights = self.data_loader.class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Optimizer for network weights
        self.w_optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.w_optimizer,
            T_max=self.config.epochs,
            eta_min=self.config.learning_rate_min
        )
        
        # Architect (for architecture parameters)
        self.architect = Architect(self.model, self.config)
        
        # Metrics
        self.metrics_calc = MetricsCalculator()
        
        print("✓ Training setup complete")
        print("=" * 70)
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        losses = AverageMeter()
        top1 = AverageMeter()
        
        self.model.train()
        
        # Get validation data iterator
        val_iter = iter(self.val_loader)
        
        for step, (input_train, target_train) in enumerate(self.train_loader):
            input_train = input_train.to(self.device)
            target_train = target_train.to(self.device)
            n = input_train.size(0)
            
            # Get validation batch for architecture update
            try:
                input_val, target_val = next(val_iter)
            except StopIteration:
                val_iter = iter(self.val_loader)
                input_val, target_val = next(val_iter)
            
            input_val = input_val.to(self.device)
            target_val = target_val.to(self.device)
            
            # Update architecture parameters (α)
            self.architect.step(input_val, target_val, self.criterion)
            
            # Update network weights (w)
            self.w_optimizer.zero_grad()
            logits = self.model(input_train)
            loss = self.criterion(logits, target_train)
            
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.w_optimizer.step()
            
            # Measure accuracy
            prec1 = accuracy(logits, target_train, topk=(1,))[0]
            losses.update(loss.item(), n)
            top1.update(prec1.item(), n)
            
            if step % self.config.report_freq == 0 or step == len(self.train_loader) - 1:
                print(f'Train: [{epoch:3d}][{step:3d}/{len(self.train_loader)}]  '
                      f'Loss {losses.avg:.4f}  '
                      f'Acc@1 {top1.avg:.3f}  '
                      f'LR {self.scheduler.get_last_lr()[0]:.6f}')
        
        # Check skip connection ratio
        skip_ratio = self.model.get_skip_connection_ratio()
        print(f'Skip Connection Ratio: {skip_ratio:.3f}')
        
        if skip_ratio > self.config.skip_connect_threshold:
            print(f'WARNING: Skip connections dominating ({skip_ratio:.3f} > {self.config.skip_connect_threshold})')
        
        return losses.avg, top1.avg
    
    def validate(self, epoch):
        """Validate the model"""
        losses = AverageMeter()
        top1 = AverageMeter()
        
        self.model.eval()
        
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for step, (input, target) in enumerate(self.val_loader):
                input = input.to(self.device)
                target = target.to(self.device)
                n = input.size(0)
                
                logits = self.model(input)
                loss = self.criterion(logits, target)
                
                # Get predictions
                probs = torch.softmax(logits, dim=1)
                _, preds = torch.max(logits, 1)
                
                # Measure accuracy
                prec1 = accuracy(logits, target, topk=(1,))[0]
                losses.update(loss.item(), n)
                top1.update(prec1.item(), n)
                
                # Store for metrics
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        # Calculate comprehensive metrics
        metrics = self.metrics_calc.calculate_metrics(
            np.array(all_targets),
            np.array(all_preds),
            np.array(all_probs)
        )
        
        print(f'\nValidation: [{epoch:3d}]  '
              f'Loss {losses.avg:.4f}  '
              f'Acc {metrics["accuracy"]:.4f}  '
              f'Sensitivity {metrics["sensitivity"]:.4f}  '
              f'Specificity {metrics["specificity"]:.4f}  '
              f'AUC {metrics.get("auc", 0):.4f}')
        
        return losses.avg, metrics
    
    def search(self):
        """Main search loop"""
        print("\nStarting architecture search...")
        print("=" * 70)
        
        best_val_acc = 0
        
        for epoch in range(self.config.epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.epochs}")
            print("-" * 70)
            
            start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_metrics = self.validate(epoch)
            
            # Update learning rate
            self.scheduler.step()
            
            epoch_time = time.time() - start_time
            print(f'Epoch time: {epoch_time:.2f}s')
            
            # Save best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                print(f'New best validation accuracy: {best_val_acc:.4f}')
                self.save_checkpoint(epoch, is_best=True)
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config.save_freq == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        print("\n" + "=" * 70)
        print("Architecture search completed!")
        print("=" * 70)
        
        # Extract and save final architecture
        self.save_genotype()
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        os.makedirs('checkpoints', exist_ok=True)
        
        state = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.w_optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'arch_params': [p.data.clone() for p in self.model.arch_parameters()]
        }
        
        filename = 'checkpoints/checkpoint_best.pth' if is_best else f'checkpoints/checkpoint_{epoch}.pth'
        torch.save(state, filename)
        print(f'Saved checkpoint: {filename}')
    
    def save_genotype(self):
        """Extract and save discovered architecture"""
        genotype = self.model.genotype()
        
        os.makedirs('results', exist_ok=True)
        
        # Save as text file
        with open('results/genotype.txt', 'w') as f:
            f.write("DISCOVERED ARCHITECTURE\n")
            f.write("=" * 70 + "\n\n")
            f.write("Normal Cell:\n")
            for i, (op, node) in enumerate(genotype['normal']):
                f.write(f"  {i}: {op} (from node {node})\n")
            f.write(f"  Concat: {genotype['normal_concat']}\n\n")
            
            f.write("Reduction Cell:\n")
            for i, (op, node) in enumerate(genotype['reduce']):
                f.write(f"  {i}: {op} (from node {node})\n")
            f.write(f"  Concat: {genotype['reduce_concat']}\n")
        
        # Save as Python dict
        import json
        with open('results/genotype.json', 'w') as f:
            json.dump(genotype, f, indent=2)
        
        print("\n" + "=" * 70)
        print("DISCOVERED ARCHITECTURE")
        print("=" * 70)
        print("\nNormal Cell:")
        for i, (op, node) in enumerate(genotype['normal']):
            print(f"  {i}: {op} (from node {node})")
        
        print("\nReduction Cell:")
        for i, (op, node) in enumerate(genotype['reduce']):
            print(f"  {i}: {op} (from node {node})")
        
        print("\n✓ Architecture saved to results/genotype.txt and results/genotype.json")


def main():
    config = SearchConfig()
    trainer = SearchTrainer(config)
    trainer.search()


if __name__ == '__main__':
    main()

