import torch
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class ExperimentConfig:
    
    # Device configuration
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 42
    num_workers: int = 8  
    
    # Dataset configuration
    datasets: List[str] = ('mnist', 'cifar10')
    data_root: str = './data'
    
    # Model configuration
    mnist_models: List[str] = ('lenet1', 'lenet4', 'lenet5')
    cifar10_models: List[str] = ('resnet20', 'resnet50')
    
    # Experiment One: Adversarial sample library configuration
    num_samples_mnist: int = 1500        
    num_samples_cifar10: int = 1500     
    attack_methods: List[str] = ('fgsm', 'pgd', 'deepfool', 'autoattack')
    
    # Batch size configuration 
    batch_size_train: int = 256       
    batch_size_inference: int = 512    
    batch_size_attack: int = 128       
    
    # FGSM parameters
    fgsm_eps: List[float] = (0.03, 0.05, 0.1)
    
    # PGD parameters
    pgd_eps: float = 0.03
    pgd_steps: int = 40               
    pgd_alpha: float = 0.01
    
    
    
    # Clustering configuration
    n_clusters_list: List[int] = (5, 10, 15)                
    hdbscan_min_cluster_size: int = 200
    hdbscan_min_samples: int = 15


    
    # Experiment Two: Mutation configuration
    num_seeds_per_cluster: int = 50   
    num_mutations_per_seed: int = 10  
    
    # Mutation parameter ranges
    rotation_range: Tuple[float, float] = (-15, 15)
    translation_range: Tuple[int, int] = (-4, 4)
    scale_range: Tuple[float, float] = (0.9, 1.1)
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    gaussian_noise_range: Tuple[float, float] = (0.01, 0.05)
    gaussian_blur_sigma: Tuple[float, float] = (0.5, 1.5)
    
    # Output paths
    output_dir: str = './results'
    checkpoint_dir: str = './checkpoints'
    figure_dir: str = './figures'
    
    # Mixed precision training
    use_amp: bool = True              
    
    # Memory optimization
    pin_memory: bool = True           
    non_blocking: bool = True        

# Initialize configuration
config = ExperimentConfig()
