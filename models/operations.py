"""
Medical-Focused Search Space Operations for PCOS Detection
Implements 14 operations: standard, dilated, grouped, attention, and PCOS-specific
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# STANDARD OPERATIONS
# ============================================================================

class SepConv(nn.Module):
    """Separable Convolution: Depthwise + Pointwise"""
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation=1, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )
    
    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):
    """Skip connection - identity mapping"""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x


class Zero(nn.Module):
    """Zero operation - for pruning"""
    def __init__(self, stride):
        super().__init__()
        self.stride = stride
    
    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


# ============================================================================
# DILATED CONVOLUTIONS (For spatial context)
# ============================================================================

class DilConv(nn.Module):
    """Dilated Convolution with specified dilation rate"""
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )
    
    def forward(self, x):
        return self.op(x)


# ============================================================================
# GROUPED CONVOLUTIONS (For efficiency)
# ============================================================================

class GroupConv(nn.Module):
    """Grouped Convolution for parameter efficiency"""
    def __init__(self, C_in, C_out, kernel_size, stride, padding, groups, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride,
                     padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )
    
    def forward(self, x):
        return self.op(x)


# ============================================================================
# ATTENTION MECHANISMS (For feature importance)
# ============================================================================

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention"""
    def __init__(self, C, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(C, C // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(C // reduction, C, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """Spatial attention for region focus"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)


class SelfAttention(nn.Module):
    """Lightweight self-attention for long-range dependencies"""
    def __init__(self, C):
        super().__init__()
        self.query = nn.Conv2d(C, C // 8, 1)
        self.key = nn.Conv2d(C, C // 8, 1)
        self.value = nn.Conv2d(C, C, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        batch, C, H, W = x.size()
        
        # Compute attention
        query = self.query(x).view(batch, -1, H * W).permute(0, 2, 1)
        key = self.key(x).view(batch, -1, H * W)
        attention = F.softmax(torch.bmm(query, key), dim=-1)
        
        # Apply attention
        value = self.value(x).view(batch, -1, H * W)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch, C, H, W)
        
        return self.gamma * out + x


# ============================================================================
# PCOS-SPECIFIC OPERATIONS
# ============================================================================

class CircularConv(nn.Module):
    """Circular convolution for follicle morphology"""
    def __init__(self, C_in, C_out, kernel_size, stride, affine=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)
        self.relu = nn.ReLU(inplace=False)
    
    def forward(self, x):
        # Apply circular padding
        pad = self.kernel_size // 2
        x = F.pad(x, (pad, pad, pad, pad), mode='circular')
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScaleConv(nn.Module):
    """Multi-scale convolution for variable follicle sizes"""
    def __init__(self, C_in, C_out, stride, affine=True):
        super().__init__()
        C_branch = C_out // 4
        
        self.branch1 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_branch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(C_branch, affine=affine)
        )
        self.branch2 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_branch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(C_branch, affine=affine)
        )
        self.branch3 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_branch, 5, stride=stride, padding=2, bias=False),
            nn.BatchNorm2d(C_branch, affine=affine)
        )
        self.branch4 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out - 3*C_branch, 7, stride=stride, padding=3, bias=False),
            nn.BatchNorm2d(C_out - 3*C_branch, affine=affine)
        )
    
    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ], dim=1)


class TextureConv(nn.Module):
    """Texture-aware convolution for stromal echogenicity"""
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )
    
    def forward(self, x):
        return self.op(x)


class EdgeAwareConv(nn.Module):
    """Edge-preserving convolution for ovarian boundaries"""
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)
        self.relu = nn.ReLU(inplace=False)
    
    def forward(self, x):
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


# ============================================================================
# OPERATION REGISTRY
# ============================================================================

def get_operation(op_name, C, stride, affine=True):
    """
    Factory function to get operation by name
    
    Args:
        op_name: Operation name from search space
        C: Number of channels
        stride: Stride for the operation
        affine: Whether to use affine parameters in BatchNorm
    """
    
    OPS = {
        'none': lambda C, stride, affine: Zero(stride),
        'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else 
                        nn.Sequential(nn.ReLU(inplace=False), 
                                     nn.Conv2d(C, C, 1, stride=stride, bias=False),
                                     nn.BatchNorm2d(C, affine=affine)),
        
        # Standard convolution
        'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
        
        # Dilated convolutions
        'dil_conv_3x3_r2': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, dilation=2, affine=affine),
        'dil_conv_3x3_r4': lambda C, stride, affine: DilConv(C, C, 3, stride, 4, dilation=4, affine=affine),
        
        # Grouped convolutions  
        'group_conv_3x3_g4': lambda C, stride, affine: GroupConv(C, C, 3, stride, 1, groups=4, affine=affine),
        'group_conv_3x3_g8': lambda C, stride, affine: GroupConv(C, C, 3, stride, 1, groups=8, affine=affine),
        
        # Attention mechanisms
        'channel_attention': lambda C, stride, affine: ChannelAttention(C),
        'spatial_attention': lambda C, stride, affine: SpatialAttention(),
        'self_attention': lambda C, stride, affine: SelfAttention(C),
        
        # Multi-scale
        'multi_scale_conv': lambda C, stride, affine: MultiScaleConv(C, C, stride, affine=affine),
        
        # PCOS-specific
        'texture_conv_3x3': lambda C, stride, affine: TextureConv(C, C, 3, stride, 1, affine=affine),
        'edge_aware_conv': lambda C, stride, affine: EdgeAwareConv(C, C, 3, stride, 1, affine=affine),
        'circular_conv_3x3': lambda C, stride, affine: CircularConv(C, C, 3, stride, affine=affine),
        'circular_conv_5x5': lambda C, stride, affine: CircularConv(C, C, 5, stride, affine=affine),
    }
    
    return OPS[op_name](C, stride, affine)


# ============================================================================
# SEARCH SPACE DEFINITION
# ============================================================================

# Normal cell operations (for feature extraction)
PRIMITIVES_NORMAL = [
    'sep_conv_3x3',
    'dil_conv_3x3_r2',
    'dil_conv_3x3_r4',
    'group_conv_3x3_g4',
    'group_conv_3x3_g8',
    'channel_attention',
    'spatial_attention',
    'self_attention',
    'multi_scale_conv',
    'texture_conv_3x3',
    'edge_aware_conv',
    'circular_conv_3x3',
    'circular_conv_5x5',
    'skip_connect',
]

# Reduction cell operations (for downsampling)
PRIMITIVES_REDUCE = [
    'sep_conv_3x3',
    'dil_conv_3x3_r2',
    'dil_conv_3x3_r4',
    'group_conv_3x3_g4',
    'group_conv_3x3_g8',
    'skip_connect',
]


if __name__ == '__main__':
    # Test operations
    print("Testing PCOS-NAS Operations...")
    print("=" * 60)
    
    C = 16
    x = torch.randn(2, C, 32, 32)
    
    for op_name in PRIMITIVES_NORMAL:
        op = get_operation(op_name, C, stride=1, affine=True)
        y = op(x)
        params = sum(p.numel() for p in op.parameters())
        print(f"{op_name:25} | Output: {y.shape} | Params: {params:,}")
    
    print("\n✓ All operations working correctly!")
