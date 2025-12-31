import torch
import torch.nn.functional as F
import numpy as np
import random
from scipy.ndimage import gaussian_filter
from torchvision import transforms
import cv2


class MutationEngine:
    """Semantic-preserving mutation operations"""
    
    def __init__(self, config):
        self.config = config
        
    def rotate(self, image, angle=None):
        """Rotation transformation"""
        if angle is None:
            angle = random.uniform(*self.config.rotation_range)
        
        if isinstance(image, np.ndarray):
            # NumPy implementation
            h, w = image.shape[-2:]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            if len(image.shape) == 2:  # Grayscale image
                rotated = cv2.warpAffine(image, M, (w, h))
            else:  # Color image
                rotated = np.zeros_like(image)
                for c in range(image.shape[0]):
                    rotated[c] = cv2.warpAffine(image[c], M, (w, h))
            return rotated
        else:
            # PyTorch implementation
            return transforms.functional.rotate(image, angle)
    
    def translate(self, image, tx=None, ty=None):
        """Translation transformation"""
        if tx is None:
            tx = random.randint(*self.config.translation_range)
        if ty is None:
            ty = random.randint(*self.config.translation_range)
        
        if isinstance(image, np.ndarray):
            h, w = image.shape[-2:]
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            
            if len(image.shape) == 2:
                translated = cv2.warpAffine(image, M, (w, h))
            else:
                translated = np.zeros_like(image)
                for c in range(image.shape[0]):
                    translated[c] = cv2.warpAffine(image[c], M, (w, h))
            return translated
        else:
            return transforms.functional.affine(image, angle=0, translate=[tx, ty], 
                                               scale=1.0, shear=0)
    
    def scale(self, image, scale_factor=None):
        """Scaling transformation"""
        if scale_factor is None:
            scale_factor = random.uniform(*self.config.scale_range)
        
        if isinstance(image, np.ndarray):
            h, w = image.shape[-2:]
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            
            if len(image.shape) == 2:
                scaled = cv2.resize(image, (new_w, new_h))
                # Crop or pad to original size
                if scale_factor > 1:
                    start_h = (new_h - h) // 2
                    start_w = (new_w - w) // 2
                    scaled = scaled[start_h:start_h+h, start_w:start_w+w]
                else:
                    pad_h = (h - new_h) // 2
                    pad_w = (w - new_w) // 2
                    scaled = np.pad(scaled, ((pad_h, h-new_h-pad_h), (pad_w, w-new_w-pad_w)), 
                                  mode='edge')
            else:
                scaled = np.zeros_like(image)
                for c in range(image.shape[0]):
                    temp = cv2.resize(image[c], (new_w, new_h))
                    if scale_factor > 1:
                        start_h = (new_h - h) // 2
                        start_w = (new_w - w) // 2
                        scaled[c] = temp[start_h:start_h+h, start_w:start_w+w]
                    else:
                        pad_h = (h - new_h) // 2
                        pad_w = (w - new_w) // 2
                        scaled[c] = np.pad(temp, ((pad_h, h-new_h-pad_h), (pad_w, w-new_w-pad_w)), 
                                         mode='edge')
            return scaled
        else:
            return transforms.functional.affine(image, angle=0, translate=[0, 0], 
                                               scale=scale_factor, shear=0)
    
    def adjust_brightness(self, image, factor=None):

        if factor is None:
            factor = random.uniform(*self.config.brightness_range)
        
        adjusted = image * factor
        return np.clip(adjusted, 0, 1)
    
    def adjust_contrast(self, image, factor=None):
    
        if factor is None:
            factor = random.uniform(*self.config.contrast_range)
        
        mean = np.mean(image)
        adjusted = (image - mean) * factor + mean
        return np.clip(adjusted, 0, 1)
    
    def add_gaussian_noise(self, image, sigma=None):
  
        if sigma is None:
            sigma = random.uniform(*self.config.gaussian_noise_range)
        
        noise = np.random.normal(0, sigma, image.shape)
        noisy = image + noise
        return np.clip(noisy, 0, 1)
    
    def gaussian_blur(self, image, sigma=None):
    
        if sigma is None:
            sigma = random.uniform(*self.config.gaussian_blur_sigma)
        
        if len(image.shape) == 2:
            blurred = gaussian_filter(image, sigma=sigma)
        else:
            blurred = np.zeros_like(image)
            for c in range(image.shape[0]):
                blurred[c] = gaussian_filter(image[c], sigma=sigma)
        
        return blurred
    
    def apply_random_mutation(self, image):
        mutations = [
            self.rotate,
            self.translate,
            self.scale,
            self.adjust_brightness,
            self.adjust_contrast,
            self.add_gaussian_noise,
            self.gaussian_blur
        ]
        
        mutation = random.choice(mutations)
        return mutation(image)
    
    def mutate_batch(self, seeds, num_mutations_per_seed):
        """
        Batch execute sklearn on seed samples
        
        Args:
            seeds: numpy array of shape (N, C, H, W)
            num_mutations_per_seed: Number of mutations per seed
        
        Returns:
            mutated_samples: numpy array of shape (N*M, C, H, W)
            seed_indices: Seed index corresponding to each mutated sample
        """
        mutated_samples = []
        seed_indices = []
        
        for i, seed in enumerate(seeds):
            for _ in range(num_mutations_per_seed):
                mutated = self.apply_random_mutation(seed)
                mutated_samples.append(mutated)
                seed_indices.append(i)
        
        return np.array(mutated_samples), np.array(seed_indices)


class ErrorPotentialEvaluator:
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.device
        self.model.eval()
    
    def evaluate_error_potential(self, seeds, seed_labels, mutation_engine, 
                                 num_mutations_per_seed):
        mutated_samples, seed_indices = mutation_engine.mutate_batch(
            seeds, num_mutations_per_seed
        )
        
        print(f" Generated {len(mutated_samples)} mutated samples")

        print("\n Evaluating error potential...")
        predictions = []
        batch_size = 128
        
        with torch.no_grad():
            for i in range(0, len(mutated_samples), batch_size):
                batch = torch.FloatTensor(mutated_samples[i:i+batch_size]).to(self.device)
                outputs = self.model(batch)
                preds = outputs.argmax(dim=1).cpu().numpy()
                predictions.append(preds)
        
        predictions = np.concatenate(predictions)
        
        # Calculate error production rate for each seed
        error_production_rates = []
        
        for seed_idx in range(len(seeds)):
            mask = seed_indices == seed_idx
            seed_mutations = predictions[mask]
            seed_label = seed_labels[seed_idx]
            
            # Error production rate = proportion still misclassified after mutation
            error_rate = np.mean(seed_mutations != seed_label)
            error_production_rates.append(error_rate)
        
        return np.array(error_production_rates), mutated_samples, predictions
