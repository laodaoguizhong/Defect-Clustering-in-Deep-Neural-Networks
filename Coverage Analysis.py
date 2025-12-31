import numpy as np
import torch
import torch.nn as nn
import json
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from experiment_2 import (
    get_parent_seeds_from_clusters,
    sample_clean_baseline_seeds,
    validate_and_collect_seeds,
    filter_noise_points
)
from clustering_analysis import ClusteringAnalyzer
from mutation_engine import MutationEngine
from config import config as exp_config


class NeuronCoverageTracker:
    """Neuron coverage tracker"""
    
    def __init__(self, model):
        self.model = model
        self.activated_neurons = {}  
        self.total_neurons = {}      
        self.hooks = []
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward hooks for all layers"""
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    activations = output.detach()
                    if len(activations.shape) > 2:
                        activations = activations.reshape(activations.size(0), -1)
                    
                    # record the activated neuron indices
                    activated = (activations > 0.6).any(dim=0)  # (num_neurons,)
                    activated_indices = torch.where(activated)[0].cpu().numpy()
                    
                    # accumulate the activated neurons
                    if name not in self.activated_neurons:
                        self.activated_neurons[name] = set()
                        self.total_neurons[name] = activations.size(1)
                    
                    self.activated_neurons[name].update(activated_indices.tolist())
                    
            return hook
        
        # register hooks for all Conv and Linear layers
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hook = module.register_forward_hook(hook_fn(name))
                self.hooks.append(hook)
    
    def get_coverage(self):
        """calculate the current cumulative coverage"""
        if not self.total_neurons:
            return 0.0
        
        total_activated = sum(len(neurons) for neurons in self.activated_neurons.values())
        total_neurons = sum(self.total_neurons.values())
        
        return total_activated / total_neurons if total_neurons > 0 else 0.0
    
    def get_layer_coverage(self):
        """get the coverage details for each layer"""
        layer_coverage = {}
        for name in self.total_neurons:
            activated = len(self.activated_neurons.get(name, set()))
            total = self.total_neurons[name]
            layer_coverage[name] = {
                'activated': activated,
                'total': total,
                'coverage': activated / total if total > 0 else 0.0
            }
        return layer_coverage
    
    def reset(self):
        """reset the coverage statistics"""
        self.activated_neurons.clear()
    
    def remove_hooks(self):
        """remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def run_mutation_with_coverage(seeds, labels, model, device, mutation_engine, 
                               n_mutations=20, tracker=None):
    """
    Synchronize the coverage and defect detection during mutation testing
    
    Returns:
        coverage_curve: coverage growth curve
        error_curve: defect discovery cumulative curve
        final_coverage: final coverage
        total_errors: total defects
    """
    model.eval()
    
    if tracker is None:
        tracker = NeuronCoverageTracker(model)
    
    tracker.reset()
    
    coverage_curve = []
    error_curve = []
    total_errors = 0
    errors_per_seed = []
    
    for seed_idx, (seed, label) in enumerate(zip(seeds, labels)):
        seed_errors = 0
        
        for mut_idx in range(n_mutations):
            mutated = mutation_engine.apply_random_mutation(seed)
            
            with torch.no_grad():
                inputs = torch.FloatTensor(mutated).unsqueeze(0).to(device)
                pred = model(inputs).argmax(dim=1).item()
            if pred != label:
                seed_errors += 1
                total_errors += 1
            current_coverage = tracker.get_coverage()
            coverage_curve.append(current_coverage)
            error_curve.append(total_errors)
        
        errors_per_seed.append(seed_errors)
    
    final_coverage = tracker.get_coverage()
    layer_coverage = tracker.get_layer_coverage()
    
    return {
        'coverage_curve': coverage_curve,
        'error_curve': error_curve,
        'final_coverage': final_coverage,
        'total_errors': total_errors,
        'errors_per_seed': np.array(errors_per_seed),
        'layer_coverage': layer_coverage,
        'ESR': float((np.array(errors_per_seed) > 0).mean()),
        'DDC': float(np.mean(errors_per_seed))
    }


def run_Coverage_Analysis(config):
    
    dataset_name = config['dataset']
    model_name = config['model']
    device = torch.device(config['device'])
    
    print(f"\n{'='*80}")
    print(f"Experiment 3: Neuron Coverage Analysis")
    print(f"Dataset: {dataset_name.upper()} | Model: {model_name.upper()}")
    print(f"{'='*80}\n")
    
    # load model and data
    from models import get_model
    from torchvision import datasets, transforms
    
    model = get_model(model_name, dataset_name)
    checkpoint = torch.load(f"./checkpoints/{dataset_name}_{model_name}.pth", 
                           map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device).eval()
    
    dataset_class = datasets.MNIST if dataset_name == 'mnist' else datasets.CIFAR10
    test_dataset = dataset_class(root='./data', train=False, download=True, 
                                transform=transforms.ToTensor())
    
    # load adversarial sample library and clustering results
    adv_library = np.load(f"./results/{dataset_name}/{model_name}/adversarial_library.npz")
    
    analyzer = ClusteringAnalyzer(model, exp_config)
    features = analyzer.extract_features(adv_library['samples'])
    clustering_results = analyzer.perform_hdbscan_clustering_highdim(features)
    
    n_seeds_per_group = config.get('n_seeds_per_group', exp_config.num_seeds_per_cluster)
    
    # extract three seed groups (same as experiment2)
    print(" Extract Near seed group...")
    core_seeds, core_labels, _ = get_parent_seeds_from_clusters(
        features, clustering_results['labels'], adv_library, 
        test_dataset, model, device, 'core', n_seeds_per_group)
    
    print(" Extract Far seed group...")
    noise_seeds, noise_labels, _ = get_parent_seeds_from_clusters(
        features, clustering_results['labels'], adv_library, 
        test_dataset, model, device, 'noise', n_seeds_per_group)
    
    print(" Extract Random baseline group...")
    clean_indices = sample_clean_baseline_seeds(test_dataset, model, device, n_seeds_per_group)
    clean_seeds, clean_labels, _ = validate_and_collect_seeds(
        clean_indices, test_dataset, model, device, n_seeds_per_group)
    
    min_n = min(len(core_seeds), len(noise_seeds), len(clean_seeds))
    core_seeds, core_labels = core_seeds[:min_n], core_labels[:min_n]
    noise_seeds, noise_labels = noise_seeds[:min_n], noise_labels[:min_n]
    clean_seeds, clean_labels = clean_seeds[:min_n], clean_labels[:min_n]
    
    
    mutation_engine = MutationEngine(exp_config)
    tracker = NeuronCoverageTracker(model)
    n_mutations = config.get('n_mutations_per_seed', exp_config.num_mutations_per_seed)
    
    print(f"\n Start coverage synchronization statistics (each group {min_n} seeds, {n_mutations} mutations)\n")
    
    
    results_near = run_mutation_with_coverage(
        core_seeds, core_labels, model, device, mutation_engine, n_mutations, tracker)
    print(f" Near group completed | coverage: {results_near['final_coverage']:.4f} | defects: {results_near['total_errors']}")
    
    tracker.reset()
    results_far = run_mutation_with_coverage(
        noise_seeds, noise_labels, model, device, mutation_engine, n_mutations, tracker)
    print(f" Far group completed  | coverage: {results_far['final_coverage']:.4f} | defects: {results_far['total_errors']}")
    
    tracker.reset()
    results_random = run_mutation_with_coverage(
        clean_seeds, clean_labels, model, device, mutation_engine, n_mutations, tracker)
    print(f" Random group completed | coverage: {results_random['final_coverage']:.4f} | defects: {results_random['total_errors']}")
    
    tracker.remove_hooks()
    
    results = {
        'dataset': dataset_name,
        'model': model_name,
        'config': {
            'n_seeds_per_group': min_n,
            'n_mutations_per_seed': n_mutations,
            'total_tests_per_group': min_n * n_mutations
        },
        'coverage_comparison': {
            'near': {
                'final_coverage': float(results_near['final_coverage']),
                'total_errors': int(results_near['total_errors']),
                'ESR': float(results_near['ESR']),
                'DDC': float(results_near['DDC'])
            },
            'far': {
                'final_coverage': float(results_far['final_coverage']),
                'total_errors': int(results_far['total_errors']),
                'ESR': float(results_far['ESR']),
                'DDC': float(results_far['DDC'])
            },
            'random': {
                'final_coverage': float(results_random['final_coverage']),
                'total_errors': int(results_random['total_errors']),
                'ESR': float(results_random['ESR']),
                'DDC': float(results_random['DDC'])
            }
        },
        'analysis': {
            'coverage_similarity': {
                'near_vs_far_diff': abs(results_near['final_coverage'] - results_far['final_coverage']),
                'near_vs_random_diff': abs(results_near['final_coverage'] - results_random['final_coverage'])
            },
            'defect_detection_disparity': {
                'near_vs_far_ratio': results_near['total_errors'] / max(results_far['total_errors'], 1),
                'near_vs_random_ratio': results_near['total_errors'] / max(results_random['total_errors'], 1)
            }
        }
    }
    
     
    save_dir = f"./results/{dataset_name}/{model_name}/experiment_3"
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, 'coverage_analysis.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    
    visualize_coverage_vs_defects(
        results_near, results_far, results_random, 
        save_dir, model_name, min_n, n_mutations
    )
    
    print(f"\n{'='*80}")
    print("Experiment 3 Results Summary")
    print(f"{'='*80}")
    print(f" Coverage comparison:")
    print(f"  Near:   {results_near['final_coverage']:.4f}")
    print(f"  Far:    {results_far['final_coverage']:.4f}")
    print(f"  Random: {results_random['final_coverage']:.4f}")
    print(f"\n Defect discovery comparison:")
    print(f"  Near:   {results_near['total_errors']} defects")
    print(f"  Far:    {results_far['total_errors']} defects")   
    print(f"  Random: {results_random['total_errors']} defects")
    print(f"{'='*80}\n")
    
    return results


def visualize_coverage_vs_defects(results_near, results_far, results_random, 
                                  save_dir, model_name, n_seeds, n_mutations):
    """visualize the relationship between coverage and defect detection"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    

    ax1 = axes[0]
    total_tests = n_seeds * n_mutations
    x = np.arange(total_tests)
    
    ax1.plot(x, results_near['coverage_curve'], label='Near', color='#e74c3c', linewidth=2)
    ax1.plot(x, results_far['coverage_curve'], label='Far', color='#f39c12', linewidth=2)
    ax1.plot(x, results_random['coverage_curve'], label='Random', color='#3498db', linewidth=2)
    
    ax1.set_xlabel('Number of Mutation Tests', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Cumulative Coverage', fontsize=12, fontweight='bold')
    ax1.set_title('Neuron Coverage Growth', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.plot(x, results_near['error_curve'], label='Near', color='#e74c3c', linewidth=2)
    ax2.plot(x, results_far['error_curve'], label='Far', color='#f39c12', linewidth=2)
    ax2.plot(x, results_random['error_curve'], label='Random', color='#3498db', linewidth=2)
    
    ax2.set_xlabel('Number of Mutation Tests', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Defects Found', fontsize=12, fontweight='bold')
    ax2.set_title('Defect Discovery Curve', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[2]
    groups = ['Near', 'Far', 'Random']
    coverages = [results_near['final_coverage'], 
                results_far['final_coverage'], 
                results_random['final_coverage']]
    defects = [results_near['total_errors'], 
              results_far['total_errors'], 
              results_random['total_errors']]
    
    x_pos = np.arange(len(groups))
    width = 0.35
    bars1 = ax3.bar(x_pos - width/2, coverages, width, label='Coverage', 
                   color='#3498db', alpha=0.8, edgecolor='#2874a6', linewidth=1.5)
    
    ax3_twin = ax3.twinx()
    bars2 = ax3_twin.bar(x_pos + width/2, defects, width, label='Defects', 
                        color='#e74c3c', alpha=0.8, edgecolor='#c0392b', linewidth=1.5)
    ax3.set_xlim(-0.5, len(groups) + 0.5)  
    ax3.set_xlabel('Seed Groups', fontsize=12, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(groups, fontsize=11)
    ax3.set_ylabel('Final Coverage', fontsize=12, fontweight='bold', color='#2874a6')
    ax3.tick_params(axis='y', labelcolor='#2874a6', labelsize=10)
    ax3.set_ylim(0, 1.0)
    ax3_twin.set_ylabel('Total Defects', fontsize=12, fontweight='bold', color='#c0392b')
    ax3_twin.tick_params(axis='y', labelcolor='#c0392b', labelsize=10)
    ax3.set_title('Coverage vs Defects Comparison', fontsize=14, fontweight='bold')
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, 
              loc='lower right',  
              fontsize=11, 
              framealpha=0.95,    
              edgecolor='gray')
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    plt.suptitle(f'{model_name.upper()} - Coverage & Defect Analysis', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'coverage_analysis.png'), 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Experiment Three: Neuron Coverage Analysis')
    parser.add_argument('--dataset', type=str, default=None, choices=['mnist', 'cifar10'])
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--n_seeds_per_group', type=int, default=exp_config.num_seeds_per_cluster)
    parser.add_argument('--n_mutations_per_seed', type=int, default=exp_config.num_mutations_per_seed)
    
    args = parser.parse_args()
    base_config = vars(args)
    
    model_map = {"mnist": ["lenet1", "lenet4", "lenet5"], 
                "cifar10": ["resnet20", "resnet50"]}
    all_results = []
    
    if args.dataset and args.model:
        try:
            all_results.append(run_Coverage_Analysis(base_config))
        except Exception as e:
            print(f" Wrong input parameters: {e}")
            import traceback
            traceback.print_exc()
    else:
        for ds, models in model_map.items():
            for m in models:
                cfg = base_config.copy()
                cfg["dataset"], cfg["model"] = ds, m
                try:
                    all_results.append(run_Coverage_Analysis(cfg))
                except Exception as e:
                    print(f" Wrong input parameters: {ds}/{m} {e}")
        
        if all_results:
            consolidated = {
                'experiment': 'Neuron Coverage Analysis (RQ Supplement)',
                'total_models': len(all_results),
                'results': all_results,
                'summary': {
                    'datasets': list(set(r['dataset'] for r in all_results)),
                    'models': list(set(r['model'] for r in all_results)),
                }
            }
            
            os.makedirs('./results', exist_ok=True)
            with open('./results/experiment_3_consolidated.json', 'w') as f:
                json.dump(consolidated, f, indent=2, ensure_ascii=False)
            
            print(f"\n All results have been saved to ./results/experiment_3_consolidated.json")
