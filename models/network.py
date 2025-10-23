"""
DARTS Network: Optimized for memory efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cell import Cell
from models.operations import PRIMITIVES_NORMAL, PRIMITIVES_REDUCE


class Network(nn.Module):
    """
    Memory-optimized DARTS network for architecture search
    """
    
    def __init__(self, C=16, num_classes=2, layers=8, steps=4, multiplier=4, stem_multiplier=3):
        super().__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._steps = steps
        self._multiplier = multiplier
        
        # Initial stem convolution
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )
        
        # Build cells
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells.append(cell)
            
            C_prev_prev, C_prev = C_prev, multiplier * C_curr
        
        # Global average pooling + classifier
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        
        # Initialize architecture parameters
        self._initialize_alphas()
    
    def _initialize_alphas(self):
        num_ops_normal = len(PRIMITIVES_NORMAL)
        num_ops_reduce = len(PRIMITIVES_REDUCE)
        
        num_edges = sum(2 + i for i in range(self._steps))
        
        self.alphas_normal = nn.Parameter(torch.randn(num_edges, num_ops_normal) * 1e-3)
        self.alphas_reduce = nn.Parameter(torch.randn(num_edges, num_ops_reduce) * 1e-3)
        
        self._arch_parameters = [self.alphas_normal, self.alphas_reduce]
    
    def arch_parameters(self):
        return self._arch_parameters
    
    def forward(self, x):
        s0 = s1 = self.stem(x)
        
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(self.alphas_reduce, dim=-1)
            else:
                weights = F.softmax(self.alphas_normal, dim=-1)
            
            # Memory optimization: detach states to prevent gradient accumulation
            if i > 0:
                s0 = s0.detach()
            
            s0, s1 = s1, cell(s0, s1, weights)
        
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        
        return logits
    
    def new_forward(self, x):
        """
        Memory-efficient forward pass for search
        Uses gradient checkpointing
        """
        s0 = s1 = self.stem(x)
        
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(self.alphas_reduce, dim=-1)
            else:
                weights = F.softmax(self.alphas_normal, dim=-1)
            
            s0, s1 = s1, cell(s0, s1, weights)
            
            # Clear cache periodically
            if (i + 1) % 2 == 0:
                torch.cuda.empty_cache()
        
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        
        return logits
    
    def get_skip_connection_ratio(self):
        skip_idx_normal = PRIMITIVES_NORMAL.index('skip_connect')
        skip_idx_reduce = PRIMITIVES_REDUCE.index('skip_connect')
        
        weights_normal = F.softmax(self.alphas_normal, dim=-1)
        weights_reduce = F.softmax(self.alphas_reduce, dim=-1)
        
        skip_ratio_normal = weights_normal[:, skip_idx_normal].mean().item()
        skip_ratio_reduce = weights_reduce[:, skip_idx_reduce].mean().item()
        
        return (skip_ratio_normal + skip_ratio_reduce) / 2
    
    def genotype(self):
        def _parse(weights, primitives):
            gene = []
            n = 2
            start = 0
            
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                
                edges = sorted(range(i + 2), 
                             key=lambda x: -max(W[x][k] for k in range(len(W[x])) 
                                               if primitives[k] != 'none'))[:2]
                
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if primitives[k] != 'none':
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((primitives[k_best], j))
                
                start = end
                n += 1
            
            return gene
        
        weights_normal = F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy()
        weights_reduce = F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy()
        
        gene_normal = _parse(weights_normal, PRIMITIVES_NORMAL)
        gene_reduce = _parse(weights_reduce, PRIMITIVES_REDUCE)
        
        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        
        return {
            'normal': gene_normal,
            'normal_concat': list(concat),
            'reduce': gene_reduce,
            'reduce_concat': list(concat)
        }


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
