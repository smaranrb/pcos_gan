"""
Configuration for DARTS architecture search
"""

class SearchConfig:
    # Dataset
    data_path = './train'  # Your training data directory
    test_path = './test'   # Your test data directory
    batch_size = 64
    num_workers = 4
    
    # Architecture
    init_channels = 16
    layers = 8
    nodes = 4
    
    # Training
    epochs = 50
    learning_rate = 0.025
    learning_rate_min = 0.001
    momentum = 0.9
    weight_decay = 3e-4
    
    # Architecture parameters
    arch_learning_rate = 3e-4
    arch_weight_decay = 1e-3
    
    # Regularization
    grad_clip = 5.0
    
    # Monitoring
    report_freq = 50
    save_freq = 10
    
    # Early stopping
    skip_connect_threshold = 0.7  # Stop if skip connections dominate
    
    # Device
    gpu = 0
    seed = 2
    
    # Class weights (from your dataset analysis)
    class_weights = [1.36, 1.00]  # [negative, positive]
