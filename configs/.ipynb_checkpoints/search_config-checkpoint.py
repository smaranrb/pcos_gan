"""
Memory-optimized configuration for DARTS architecture search
"""

class SearchConfig:
    # Dataset
    data_path = './data/train'
    test_path = './data/test'
    batch_size = 32  # Reduced from 64 for safety
    num_workers = 4
    
    # Architecture (more conservative)
    init_channels = 12  # Reduced from 16
    layers = 8
    nodes = 4
    
    # Image size
    image_size = 224
    
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
    
    # Memory optimization
    gradient_accumulation_steps = 2  # Effectively batch_size=64
    use_amp = True  # Mixed precision training
    
    # Monitoring
    report_freq = 50
    save_freq = 10
    
    # Early stopping
    skip_connect_threshold = 0.7
    
    # Device
    gpu = 0
    seed = 2
    
    # Class weights
    class_weights = [1.36, 1.00]
