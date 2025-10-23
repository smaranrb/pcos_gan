"""
Check if system has enough memory for DARTS search
"""

import torch
import psutil
import subprocess

def check_system_memory():
    """Check RAM and GPU memory"""
    print("=" * 70)
    print("SYSTEM MEMORY CHECK")
    print("=" * 70)
    
    # CPU RAM
    ram = psutil.virtual_memory()
    print(f"\nCPU RAM:")
    print(f"  Total: {ram.total / 1024**3:.2f} GB")
    print(f"  Available: {ram.available / 1024**3:.2f} GB")
    print(f"  Used: {ram.used / 1024**3:.2f} GB ({ram.percent}%)")
    
    # GPU Memory
    if torch.cuda.is_available():
        print(f"\nGPU Information:")
        for i in range(torch.cuda.device_count()):
            print(f"\n  GPU {i}: {torch.cuda.get_device_name(i)}")
            
            # Total memory
            total_memory = torch.cuda.get_device_properties(i).total_memory
            print(f"    Total Memory: {total_memory / 1024**3:.2f} GB")
            
            # Current usage
            allocated = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            print(f"    Allocated: {allocated / 1024**3:.2f} GB")
            print(f"    Reserved: {reserved / 1024**3:.2f} GB")
            print(f"    Free: {(total_memory - reserved) / 1024**3:.2f} GB")
        
        # Recommendations
        print("\n" + "=" * 70)
        print("RECOMMENDATIONS FOR DARTS SEARCH")
        print("=" * 70)
        
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        if total_vram >= 20:
            print("\n✓ Excellent! 20GB+ VRAM")
            print("  Recommended config:")
            print("    - init_channels: 16")
            print("    - layers: 8")
            print("    - batch_size: 64")
        elif total_vram >= 16:
            print("\n✓ Good! 16GB+ VRAM")
            print("  Recommended config:")
            print("    - init_channels: 12")
            print("    - layers: 8")
            print("    - batch_size: 48")
        elif total_vram >= 11:
            print("\n⚠ Adequate. 11GB+ VRAM")
            print("  Recommended config:")
            print("    - init_channels: 12")
            print("    - layers: 6")
            print("    - batch_size: 32")
        else:
            print("\n⚠ Limited VRAM. Consider:")
            print("    - init_channels: 8")
            print("    - layers: 6")
            print("    - batch_size: 16")
            print("    - gradient_accumulation_steps: 4")
    else:
        print("\n⚠ No GPU detected. CPU training will be very slow.")
    
    print("\n" + "=" * 70)

if __name__ == '__main__':
    check_system_memory()
