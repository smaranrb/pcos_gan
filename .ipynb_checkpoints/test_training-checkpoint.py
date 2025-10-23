"""
Quick test of training pipeline (1 epoch)
"""

import torch
from configs.search_config import SearchConfig
from search_architecture import SearchTrainer

# Modify config for quick test
class TestConfig(SearchConfig):
    epochs = 2  # Just 2 epochs for testing
    batch_size = 16  # Smaller batch
    init_channels = 8  # Smaller network
    layers = 4
    report_freq = 10

print("Running quick training test (2 epochs)...")
print("This will verify the complete pipeline works\n")

config = TestConfig()
trainer = SearchTrainer(config)
trainer.search()

print("\n✓ Training pipeline test successful!")
print("You can now run the full search with: python search_architecture.py")

