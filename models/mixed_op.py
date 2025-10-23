"""
Mixed Operation: Core DARTS component that combines all candidate operations
with learnable architecture weights (alpha)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.operations import get_operation


class MixedOp(nn.Module):
    """
    Mixed operation that combines multiple operations with architecture parameters
    
    Key idea: Instead of selecting one operation, we apply all operations
    and weight their outputs using softmax(alpha)
    """
    
    def __init__(self, C, stride, primitives):
        """
        Args:
            C: Number of input/output channels
            stride: Stride for operations (1 or 2)
            primitives: List of operation names (search space)
        """
        super().__init__()
        self._ops = nn.ModuleList()
        
        # Create all candidate operations
        for primitive in primitives:
            op = get_operation(primitive, C, stride, affine=False)
            self._ops.append(op)
    
    def forward(self, x, weights):
        """
        Forward pass with architecture weights
        
        Args:
            x: Input tensor
            weights: Architecture weights (alpha) - softmax applied already
        
        Returns:
            Weighted sum of all operations
        """
        # Apply each operation and weight by alpha
        return sum(w * op(x) for w, op in zip(weights, self._ops))


if __name__ == '__main__':
    # Test MixedOp
    from models.operations import PRIMITIVES_NORMAL
    
    print("Testing MixedOp...")
    C = 16
    mixed_op = MixedOp(C, stride=1, primitives=PRIMITIVES_NORMAL)
    
    x = torch.randn(2, C, 32, 32)
    weights = F.softmax(torch.randn(len(PRIMITIVES_NORMAL)), dim=0)
    
    y = mixed_op(x, weights)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Number of operations: {len(PRIMITIVES_NORMAL)}")
    print(f"Weights sum: {weights.sum().item():.4f}")
    print("✓ MixedOp working correctly!")
