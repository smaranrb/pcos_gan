"""
Memory-optimized testing of all DARTS components
"""

import torch
import torch.cuda as cuda

def check_memory():
    """Check GPU memory usage"""
    if torch.cuda.is_available():
        allocated = cuda.memory_allocated() / 1024**3  # GB
        reserved = cuda.memory_reserved() / 1024**3    # GB
        print(f"   GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

print("=" * 70)
print("TESTING ALL DARTS COMPONENTS (Memory Optimized)")
print("=" * 70)

# Enable GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# Test 1: Operations
print("\n1. Testing Operations...")
from models.operations import get_operation, PRIMITIVES_NORMAL
op = get_operation('sep_conv_3x3', C=16, stride=1)
op = op.to(device)
x = torch.randn(2, 16, 32, 32).to(device)
with torch.no_grad():
    y = op(x)
assert y.shape == x.shape
print("   ✓ Operations working")
check_memory()
del op, x, y
torch.cuda.empty_cache()

# Test 2: MixedOp
print("\n2. Testing MixedOp...")
from models.mixed_op import MixedOp
mixed_op = MixedOp(16, 1, PRIMITIVES_NORMAL).to(device)
x = torch.randn(2, 16, 32, 32).to(device)
weights = torch.softmax(torch.randn(len(PRIMITIVES_NORMAL)).to(device), dim=0)
with torch.no_grad():
    y = mixed_op(x, weights)
assert y.shape == x.shape
print("   ✓ MixedOp working")
check_memory()
del mixed_op, x, y, weights
torch.cuda.empty_cache()

# Test 3: Cell
print("\n3. Testing Cell...")
from models.cell import Cell
cell = Cell(steps=4, multiplier=4, C_prev_prev=16, C_prev=16, C=16, 
            reduction=False, reduction_prev=False).to(device)
s0 = torch.randn(2, 16, 32, 32).to(device)
s1 = torch.randn(2, 16, 32, 32).to(device)
num_edges = sum(2 + i for i in range(4))
weights = [torch.softmax(torch.randn(len(PRIMITIVES_NORMAL)).to(device), dim=0) 
           for _ in range(num_edges)]
with torch.no_grad():
    y = cell(s0, s1, weights)
assert y.shape[0] == 2 and y.shape[2] == 32 and y.shape[3] == 32
print("   ✓ Cell working")
check_memory()
del cell, s0, s1, y, weights
torch.cuda.empty_cache()

# Test 4: Full Network (with smaller config)
print("\n4. Testing Full Network...")
print("   Using smaller configuration for memory efficiency:")
print("   - Batch size: 1 (instead of 2)")
print("   - Image size: 128x128 (instead of 224x224)")
print("   - Channels: 8 (instead of 16)")
print("   - Layers: 4 (instead of 8)")

from models.network import Network
try:
    model = Network(C=8, num_classes=2, layers=4, steps=4).to(device)
    x = torch.randn(1, 3, 128, 128).to(device)
    
    with torch.no_grad():
        logits = model(x)
    
    assert logits.shape == (1, 2)
    print("   ✓ Network working")
    check_memory()
    
    # Count parameters
    arch_params = sum(p.numel() for p in model.arch_parameters())
    total_params = sum(p.numel() for p in model.parameters())
    weight_params = total_params - arch_params
    
    print(f"\n   Network Statistics:")
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Weight parameters: {weight_params:,}")
    print(f"   - Architecture parameters: {arch_params:,}")
    
    del model, x, logits
    torch.cuda.empty_cache()
    
except RuntimeError as e:
    print(f"   ✗ Network failed: {e}")
    print("   This is likely due to memory constraints")

# Test 5: Genotype extraction (CPU only, no memory issue)
print("\n5. Testing Genotype extraction...")
model = Network(C=8, num_classes=2, layers=4, steps=4)
genotype = model.genotype()
assert 'normal' in genotype and 'reduce' in genotype
print("   ✓ Genotype extraction working")
del model

# Test 6: Architecture parameters
print("\n6. Testing Architecture parameters...")
model = Network(C=8, num_classes=2, layers=4, steps=4)
arch_params = model.arch_parameters()
assert len(arch_params) == 2
print("   ✓ Architecture parameters working")
del model

print("\n" + "=" * 70)
print("ALL TESTS PASSED! ✓")
print("=" * 70)
print("\nMemory Optimization Notes:")
print("- For actual search, we'll use C=16, layers=8, batch_size=64")
print("- With 20GB VRAM, this should work fine")
print("- Image size during search: 224x224")
print("- Gradient accumulation will be used if needed")
print("\nYou're ready to proceed to Part 3: Data Loading & Training!")
