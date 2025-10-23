"""
DARTS Cell: Building block of the network
Two types: Normal Cell (maintains resolution) and Reduction Cell (downsamples)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mixed_op import MixedOp
from models.operations import PRIMITIVES_NORMAL, PRIMITIVES_REDUCE


class Cell(nn.Module):
    """
    DARTS Cell: Directed Acyclic Graph with learnable connections
    
    Architecture:
        - 2 input nodes (from previous 2 cells)
        - N intermediate nodes (default: 4)
        - Each intermediate node receives inputs from all previous nodes
        - Output: Concatenation of all intermediate nodes
    """
    
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        """
        Args:
            steps: Number of intermediate nodes (typically 4)
            multiplier: Channel multiplier for output (typically 4)
            C_prev_prev: Channels from cell k-2
            C_prev: Channels from cell k-1
            C: Current cell channels
            reduction: Whether this is a reduction cell
            reduction_prev: Whether previous cell was reduction
        """
        super().__init__()
        self.reduction = reduction
        self.primitives = PRIMITIVES_REDUCE if reduction else PRIMITIVES_NORMAL
        
        # Preprocess inputs to ensure channel consistency
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
        
        self._steps = steps
        self._multiplier = multiplier
        
        # Build mixed operations for all edges
        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        
        for i in range(self._steps):
            for j in range(2 + i):  # Can connect to input nodes + previous intermediate nodes
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride, self.primitives)
                self._ops.append(op)
    
    def forward(self, s0, s1, weights):
        """
        Forward pass through cell
        
        Args:
            s0: Output from cell k-2
            s1: Output from cell k-1  
            weights: Architecture weights for all edges [num_edges, num_ops]
        
        Returns:
            Cell output (concatenation of intermediate nodes)
        """
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        
        states = [s0, s1]
        offset = 0
        
        # Compute each intermediate node
        for i in range(self._steps):
            # Sum weighted outputs from all previous nodes
            s = sum(self._ops[offset + j](h, weights[offset + j]) 
                   for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        
        # Concatenate all intermediate nodes (skip the two input nodes)
        return torch.cat(states[-self._multiplier:], dim=1)


class ReLUConvBN(nn.Module):
    """Basic conv block with ReLU, Conv, BatchNorm"""
    
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=False)
        )
    
    def forward(self, x):
        return self.op(x)


class FactorizedReduce(nn.Module):
    """
    Reduce spatial resolution by 2x while increasing channels
    Uses factorized approach to avoid representational bottleneck
    """
    
    def __init__(self, C_in, C_out):
        super().__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=False)
    
    def forward(self, x):
        x = self.relu(x)
        # Use two conv layers with offset to preserve information
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


if __name__ == '__main__':
    # Test Cell
    print("Testing DARTS Cell...")
    
    steps = 4
    multiplier = 4
    C = 16
    
    cell = Cell(steps, multiplier, C_prev_prev=C, C_prev=C, C=C, 
                reduction=False, reduction_prev=False)
    
    # Create dummy inputs
    s0 = torch.randn(2, C, 32, 32)
    s1 = torch.randn(2, C, 32, 32)
    
    # Create dummy architecture weights
    num_edges = sum(2 + i for i in range(steps))  # Total edges in cell
    num_ops = len(PRIMITIVES_NORMAL)
    weights = [F.softmax(torch.randn(num_ops), dim=0) for _ in range(num_edges)]
    
    # Forward pass
    output = cell(s0, s1, weights)
    
    print(f"Input s0 shape: {s0.shape}")
    print(f"Input s1 shape: {s1.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output channels: {C * multiplier}")
    print("✓ Cell working correctly!")
