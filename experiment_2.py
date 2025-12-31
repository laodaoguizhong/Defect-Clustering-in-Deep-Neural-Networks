"""
Experiment Two: Validation of Defect Detection Potential (RQ2)
"""

import numpy as np
import torch
import json
import os
import time
from scipy.stats import ttest_ind
from clustering_analysis import ClusteringAnalyzer
from mutation_engine import MutationEngine
from config import config as exp_config


def filter_noise_points(cluster_labels):
    """Filter HDBSCAN noise points"""
    valid_mask = (cluster_labels != -1)
    return valid_mask, ~valid_mask


def validate_and_collect_seeds(indices, test_dataset, model, device, n_samples):
    """Validate and collect correctly classified seeds"""
    seeds, labels, valid_indices = [], [], []
    model.eval()
    
    with torch.no_grad():
        for idx in indices:
            if len(seeds) >= n_samples:
                break
            img, label = test_dataset[int(idx)]
            pred = model(img.unsqueeze(0).to(device)).argmax(dim=1).item()
            
            if pred == label:
                seeds.append(img.numpy())
                labels.append(label)
                valid_indices.append(idx)
    
    return np.array(seeds), np.array(labels), np.array(valid_indices)


def get_parent_seeds_from_clusters(
    features, cluster_labels, adv_library, test_dataset, 
    model, device, strategy='core', n_samples=300
):
    """Select seeds based on clustering results"""
    
    initial_n_samples = n_samples * 5
    valid_mask, noise_mask = filter_noise_points(cluster_labels)
    
    if strategy == 'core':
        # Core cluster region
        valid_features = features[valid_mask]
        valid_labels = cluster_labels[valid_mask]
        valid_adv_indices = np.where(valid_mask)[0]
        
        selected_adv_indices = []
        samples_per_cluster = initial_n_samples // len(np.unique(valid_labels))
        
        for cluster_id in np.unique(valid_labels):
            cluster_mask = (valid_labels == cluster_id)
            cluster_features = valid_features[cluster_mask]
            cluster_adv_indices = valid_adv_indices[cluster_mask]
            
            # Select nearest samples
            centroid = cluster_features.mean(axis=0)
            distances = np.linalg.norm(cluster_features - centroid, axis=1)
            closest = np.argsort(distances)[:min(samples_per_cluster, len(cluster_adv_indices))]
            selected_adv_indices.extend(cluster_adv_indices[closest])
        
        if len(selected_adv_indices) < initial_n_samples:
            available = list(set(valid_adv_indices) - set(selected_adv_indices))
            if available:
                extra = np.random.choice(available, 
                    min(initial_n_samples - len(selected_adv_indices), len(available)), 
                    replace=False)
                selected_adv_indices.extend(extra)
        
        selected_adv_indices = np.array(selected_adv_indices[:initial_n_samples])
        
    elif strategy == 'noise':
        # Outlier noise points
        noise_indices = np.where(noise_mask)[0]
        if len(noise_indices) == 0:
            raise ValueError("No noise points available")
        
        # Calculate distance to cluster center
        valid_features = features[valid_mask]
        cluster_centroids = np.array([
            valid_features[cluster_labels[valid_mask] == cid].mean(axis=0)
            for cid in np.unique(cluster_labels[valid_mask])
        ])
        
        noise_features = features[noise_indices]
        distances = np.min(np.linalg.norm(
            noise_features[:, np.newaxis, :] - cluster_centroids[np.newaxis, :, :], 
            axis=2), axis=1)
        selected_adv_indices = noise_indices[np.argsort(distances)[::-1][:min(initial_n_samples, len(noise_indices))]]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    parent_indices = adv_library['original_indices'][selected_adv_indices]
    seeds, labels, valid_indices = validate_and_collect_seeds(
        parent_indices, test_dataset, model, device, n_samples)
    if len(seeds) < n_samples:
        available_mask = valid_mask if strategy == 'core' else noise_mask
        remaining = list(set(np.where(available_mask)[0]) - set(selected_adv_indices))
        np.random.shuffle(remaining)
        
        extra_parent_indices = adv_library['original_indices'][remaining]
        extra_seeds, extra_labels, extra_indices = validate_and_collect_seeds(
            extra_parent_indices, test_dataset, model, device, n_samples - len(seeds))
        
        seeds = np.concatenate([seeds, extra_seeds])
        labels = np.concatenate([labels, extra_labels])
        valid_indices = np.concatenate([valid_indices, extra_indices])
    
    return seeds, labels, valid_indices


def sample_clean_baseline_seeds(test_dataset, model, device, n_samples=300):
    """Sample baseline group"""
    from torch.utils.data import DataLoader
    
    correct_indices = []
    model.eval()
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(DataLoader(test_dataset, batch_size=128)):
            inputs, labels = inputs.to(device), labels.to(device)
            preds = model(inputs).argmax(dim=1)
            correct_indices.extend((i * 128 + np.where((preds == labels).cpu().numpy())[0]).tolist())
            
            if len(correct_indices) >= n_samples:
                break
    
    return np.random.choice(correct_indices, min(n_samples, len(correct_indices)), replace=False)


def run_mutation_testing(seeds, labels, model, device, mutation_engine, n_mutations=20):
    """Test seeds by mutation"""
    model.eval()
    errors_per_seed = []
    total_time = 0.0
    
    for seed, label in zip(seeds, labels):
        seed_errors = 0
        start_time = time.time()
        first_error_time = None
        
        for _ in range(n_mutations):
            mutated = mutation_engine.apply_random_mutation(seed)
            
            with torch.no_grad():
                pred = model(torch.FloatTensor(mutated).unsqueeze(0).to(device)).argmax(dim=1).item()
            
            if pred != label:
                seed_errors += 1
                if first_error_time is None:
                    first_error_time = time.time()
        
        # Cumulative time: First error or completion of all variations
        total_time += (first_error_time or time.time()) - start_time
        errors_per_seed.append(seed_errors)
    
    errors_per_seed = np.array(errors_per_seed)
    
    return {
        'ESR': float((errors_per_seed > 0).mean()),
        'DDC': float(errors_per_seed.mean()),
        'total_errors': int(errors_per_seed.sum()),
        'mean_errors_per_seed': float(errors_per_seed.mean()),
        'std_errors_per_seed': float(errors_per_seed.std()),
        'median_errors_per_seed': float(np.median(errors_per_seed))
    }, errors_per_seed, total_time


def compute_stats(errors1, errors2, metric_name):
    """Calculate statistical tests"""
    t_stat, p_value = ttest_ind(errors1, errors2)
    pooled_std = np.sqrt((errors1.std()**2 + errors2.std()**2) / 2)
    cohens_d = (errors1.mean() - errors2.mean()) / pooled_std if pooled_std > 0 else 0
    
    sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
    
    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'cohens_d': float(cohens_d),
        'significance': sig
    }


def run_experiment_two(config):
    """Run experiment two"""
    
    dataset_name = config['dataset']
    model_name = config['model']
    device = torch.device(config['device'])
    
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name.upper()} | Model: {model_name.upper()}")
    print(f"{'='*80}\n")
    
    # Load model
    from models import get_model
    from torchvision import datasets, transforms
    
    model = get_model(model_name, dataset_name)
    checkpoint = torch.load(f"./checkpoints/{dataset_name}_{model_name}.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device).eval()
    
    # Load dataset
    dataset_class = datasets.MNIST if dataset_name == 'mnist' else datasets.CIFAR10
    test_dataset = dataset_class(root='./data', train=False, download=True, transform=transforms.ToTensor())
    
    # Load adversarial library and cluster
    adv_library = np.load(f"./results/{dataset_name}/{model_name}/adversarial_library.npz")
    
    analyzer = ClusteringAnalyzer(model, exp_config)
    features = analyzer.extract_features(adv_library['samples'])
    clustering_results = analyzer.perform_hdbscan_clustering_highdim(features)
    
    n_seeds_per_group = config.get('n_seeds_per_group', exp_config.num_seeds_per_cluster)
    
    
    # Extract three groups of seeds
    print("Extract near seeds...")
    core_seeds, core_labels, _ = get_parent_seeds_from_clusters(
        features, clustering_results['labels'], adv_library, 
        test_dataset, model, device, 'core', n_seeds_per_group)
    
    print("\n Extract far seeds...")
    noise_seeds, noise_labels, _ = get_parent_seeds_from_clusters(
        features, clustering_results['labels'], adv_library, 
        test_dataset, model, device, 'noise', n_seeds_per_group)
    
    print("\n Randomly select baseline group...")
    clean_indices = sample_clean_baseline_seeds(test_dataset, model, device, n_seeds_per_group)
    clean_seeds, clean_labels, _ = validate_and_collect_seeds(
        clean_indices, test_dataset, model, device, n_seeds_per_group)
    
    # Uniformly crop to the same number of seeds
    min_n = min(len(core_seeds), len(noise_seeds), len(clean_seeds))    
    core_seeds, core_labels = core_seeds[:min_n], core_labels[:min_n]
    noise_seeds, noise_labels = noise_seeds[:min_n], noise_labels[:min_n]
    clean_seeds, clean_labels = clean_seeds[:min_n], clean_labels[:min_n]
    
    # Mutation testing
    mutation_engine = MutationEngine(exp_config)
    n_mutations = config.get('n_mutations_per_seed', exp_config.num_mutations_per_seed)
    
    print(f"\n Start mutation testing(each seed {n_mutations} times)\n")
    
    stats_core, errors_core, time_core = run_mutation_testing(
        core_seeds, core_labels, model, device, mutation_engine, n_mutations)
    
    
    stats_noise, errors_noise, time_noise = run_mutation_testing(
        noise_seeds, noise_labels, model, device, mutation_engine, n_mutations)
   
    stats_clean, errors_clean, time_clean = run_mutation_testing(
        clean_seeds, clean_labels, model, device, mutation_engine, n_mutations)
   
    
    # Calculate DDE
    dde_core = (errors_core > 0).sum() / time_core if time_core > 0 else 0
    dde_noise = (errors_noise > 0).sum() / time_noise if time_noise > 0 else 0
    dde_clean = (errors_clean > 0).sum() / time_clean if time_clean > 0 else 0
    
    
    esr_core, esr_noise, esr_clean = (errors_core > 0).astype(float), (errors_noise > 0).astype(float), (errors_clean > 0).astype(float)
    
    
    results = {
        'dataset': dataset_name,
        'model': model_name,
        'config': {
            'n_seeds_per_group': min_n,
            'n_mutations_per_seed': n_mutations,
            'total_tests_per_group': min_n * n_mutations
        },
        'group_comparison': {
            'near': {
                'n_seeds': min_n,
                'ESR': stats_core['ESR'],
                'DDC': stats_core['DDC'],
                'DDE': float(dde_core),
                'n_effective_seeds': int((errors_core > 0).sum()),
                'total_time': float(time_core),
                'total_errors': stats_core['total_errors']
            },
            'far': {
                'n_seeds': min_n,
                'ESR': stats_noise['ESR'],
                'DDC': stats_noise['DDC'],
                'DDE': float(dde_noise),
                'n_effective_seeds': int((errors_noise > 0).sum()),
                'total_time': float(time_noise),
                'total_errors': stats_noise['total_errors']
            },
            'random_baseline': {
                'n_seeds': min_n,
                'ESR': stats_clean['ESR'],
                'DDC': stats_clean['DDC'],
                'DDE': float(dde_clean),
                'n_effective_seeds': int((errors_clean > 0).sum()),
                'total_time': float(time_clean),
                'total_errors': stats_clean['total_errors']
            }
        },
        'statistical_tests': {
            'DDC': {
                'near_vs_baseline': compute_stats(errors_core, errors_clean, 'DDC'),
                'near_vs_far': compute_stats(errors_core, errors_noise, 'DDC')
            },
            'ESR': {
                'near_vs_baseline': compute_stats(esr_core, esr_clean, 'ESR'),
                'near_vs_far': compute_stats(esr_core, esr_noise, 'ESR')
            },
            'DDE': {
                'near_vs_baseline': compute_stats(errors_core, errors_clean, 'DDE'),
                'near_vs_far': compute_stats(errors_core, errors_noise, 'DDE')
            }
        }
        
    }
    
    # Save results
    save_dir = f"./results/{dataset_name}/{model_name}/experiment_2"
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, 'experiment_two.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # visualisation
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 1, figsize=(3, 8))
    groups, colors = ['Near', 'Far', 'Random'], ['#e74c3c', '#f39c12', '#3498db']
    x = np.arange(3) * 0.5
    
    for ax, values, ylabel, ylim, yticks in zip(
        axes,
        [[stats_core['ESR'], stats_noise['ESR'], stats_clean['ESR']],
         [stats_core['DDC'], stats_noise['DDC'], stats_clean['DDC']],
         [dde_core, dde_noise, dde_clean]],
        ['ESR', 'DDC', 'DDE (Defects per Second)'],
        [(0, 1), (0, 4), None],
        [np.arange(0, 1.01, 0.2), np.arange(0, 4.01, 0.5), None]
    ):
        ax.bar(x, values, color=colors, alpha=0.7, width=0.2)
        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.set_ylabel(ylabel, fontweight='bold')
        if ylim: ax.set_ylim(ylim)
        if yticks is not None: ax.set_yticks(yticks)
        ax.grid(True, alpha=0.3, axis='y')
    
    axes[2].set_xlabel(model_name.capitalize(), fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'experiment_two.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f" Results：")
    print(f"Near (n={min_n}): ESR={stats_core['ESR']:.3f} | DDC={stats_core['DDC']:.3f} | DDE={dde_core:.2f}")
    print(f"Far (n={min_n}): ESR={stats_noise['ESR']:.3f} | DDC={stats_noise['DDC']:.3f} | DDE={dde_noise:.2f}")
    print(f"Random (n={min_n}): ESR={stats_clean['ESR']:.3f} | DDC={stats_clean['DDC']:.3f} | DDE={dde_clean:.2f}")
    print(f"{'='*80}\n")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Experiment Two: Validation of Defect Detection Potential')
    parser.add_argument('--dataset', type=str, default=None, choices=['mnist', 'cifar10'])
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--n_seeds_per_group', type=int, default=exp_config.num_seeds_per_cluster)
    parser.add_argument('--n_mutations_per_seed', type=int, default=exp_config.num_mutations_per_seed)

    args = parser.parse_args()
    base_config = vars(args)

    model_map = {"mnist": ["lenet1", "lenet4", "lenet5"], "cifar10": ["resnet20", "resnet50"]}
    all_results = []

    if args.dataset and args.model:
        try:
            all_results.append(run_experiment_two(base_config))
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
                    all_results.append(run_experiment_two(cfg))
                except Exception as e:
                    print(f" Wrong input parameters: {ds}/{m} {e}")
        
        if all_results:
            consolidated = {
                'experiment': 'Defect Detection Potential Validation(RQ2)',
                'total_models': len(all_results),
                'results': all_results,
                'summary': {
                    'datasets': list(set(r['dataset'] for r in all_results)),
                    'models': list(set(r['model'] for r in all_results))
                }
            }
            
            os.makedirs('./results', exist_ok=True)
            with open('./results/experiment_2.json', 'w') as f:
                json.dump(consolidated, f, indent=2, ensure_ascii=False)
            
            print(f" Results saved in ./results/experiment_2.json ")

