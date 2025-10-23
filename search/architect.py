"""
Architect: Handles architecture parameter optimization
"""

import torch
import torch.nn as nn


class Architect:
    """
    Manages architecture parameter updates using validation loss
    """
    
    def __init__(self, model, args):
        """
        Args:
            model: DARTS network
            args: Configuration arguments
        """
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        
        # Optimizer for architecture parameters
        self.optimizer = torch.optim.Adam(
            self.model.arch_parameters(),
            lr=args.arch_learning_rate,
            betas=(0.5, 0.999),
            weight_decay=args.arch_weight_decay
        )
    
    def step(self, input_valid, target_valid, criterion):
        """
        Update architecture parameters using validation data
        
        Args:
            input_valid: Validation input batch
            target_valid: Validation target batch
            criterion: Loss function
        """
        self.optimizer.zero_grad()
        
        # Forward pass on validation data
        logits = self.model(input_valid)
        loss = criterion(logits, target_valid)
        
        # Backward pass
        loss.backward()
        
        # Update architecture parameters
        self.optimizer.step()
        
        return loss.item()

