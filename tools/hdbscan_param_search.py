import itertools
import os
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import torch

from sklearn.metrics import silhouette_score

from config import config as exp_config
from clustering_analysis import ClusteringAnalyzer
from experiment_1 import _load_model, _load_or_generate_adv


def evaluate_hdbscan_configuration(
    features: np.ndarray,
    analyzer: ClusteringAnalyzer,
    min_cluster_size: int,
    min_samples: int,
) -> Dict[str, Any]:
    """
    Run HDBSCAN once for the given combination of (min_cluster_size, min_samples),
    returning clustering quality metrics consistent with experiment_1 to facilitate grid search.  
    """
    res = analyzer.perform_hdbscan_clustering_highdim(
        features,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )

    labels = res["labels"]
    n_noise = int(np.sum(labels == -1))
    noise_ratio = n_noise / len(labels)

    return {
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "n_clusters": int(res["n_clusters"]),
        "noise_ratio": float(noise_ratio),
        "silhouette_score": float(res["silhouette_score"]),
        "davies_bouldin_score": float(res["davies_bouldin_score"]),
        "calinski_harabasz_score": float(res["calinski_harabasz_score"]),
    }


def grid_search_hdbscan_params(
    dataset: str,
    model_name: str,
    device: str = None,
    min_cluster_size_list: Optional[List[int]] = None,
    min_samples_list: Optional[List[int]] = None,
    noise_ratio_threshold: float = 1.0,  
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    For a given dataset and model, employing an adversarial sample library alongside high-dimensional features,
    search for HDBSCAN parameters across a grid of (min_cluster_size, min_samples).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if min_cluster_size_list is None:
        min_cluster_size_list = [5,10,15,20,30,50,80,100,150,170,200,250]
    if min_samples_list is None:
        min_samples_list = [3, 5, 8, 10, 15, 20, 30, 50, 80, 100]

    torch_device = torch.device(device)

    print(f"=== Grid search HDBSCAN params on {dataset}-{model_name} ({torch_device}) ===")
    print(f"min_cluster_size_list = {min_cluster_size_list}")
    print(f"min_samples_list      = {min_samples_list}")

    # Load the model and adversarial sample library
    dummy_cfg: Dict[str, Any] = {
        "dataset": dataset,
        "model": model_name,
        "device": device,
        "hdbscan_min_cluster_size": None,
        "hdbscan_min_samples": None,
    }

    model = _load_model(dataset, model_name, torch_device)
    adv_lib = _load_or_generate_adv(model, dataset, model_name, dummy_cfg, torch_device)

    #Extracting high-dimensional features
    analyzer = ClusteringAnalyzer(model, exp_config)
    features = analyzer.extract_features(adv_lib["samples"])

    #  Grid search
    all_results: List[Dict[str, Any]] = []
    best_cfg: Dict[str, Any] = None

    for mcs, ms in itertools.product(min_cluster_size_list, min_samples_list):
        stats = evaluate_hdbscan_configuration(features, analyzer, mcs, ms)
        all_results.append(stats)
        print(
            f"[mcs={mcs:4d}, ms={ms:4d}] "
            f"clusters={stats['n_clusters']}, "
            f"noise={stats['noise_ratio']:.3f}, "
            f"silhouette={stats['silhouette_score']:.4f}"
        )

        if best_cfg is None:
            best_cfg = stats
        else:
            if stats["silhouette_score"] > best_cfg["silhouette_score"]:
                best_cfg = stats
            elif (
                np.isclose(stats["silhouette_score"], best_cfg["silhouette_score"], atol=1e-3)
                and stats["noise_ratio"] < best_cfg["noise_ratio"]
            ):
                best_cfg = stats

    print("\n=== Grid search finished ===")
    if best_cfg is None:
        print("No configuration satisfied the constraints (at least 2 clusters and low noise).")
    else:
        print(
            "Best configuration:\n"
            f"  min_cluster_size = {best_cfg['min_cluster_size']}\n"
            f"  min_samples      = {best_cfg['min_samples']}\n"
            f"  n_clusters       = {best_cfg['n_clusters']}\n"
            f"  noise_ratio      = {best_cfg['noise_ratio']:.4f}\n"
            f"  silhouette_score = {best_cfg['silhouette_score']:.4f}"
        )

    return best_cfg, all_results


def _parse_int_list(arg: str) -> List[int]:
    return [int(x.strip()) for x in arg.split(",") if x.strip()]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Grid search HDBSCAN hyperparameters.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["mnist", "cifar10"],
        help="Dataset name",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., lenet1, lenet4, lenet5, resnet20, resnet50)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--noise_ratio_threshold",
        type=float,
        default=1.0,
        help="Deprecated: kept for backward compatibility; no longer used for filtering.",
    )
    parser.add_argument(
        "--mcs_list",
        type=str,
        default=None,
        help="Comma-separated min_cluster_size list, e.g., '50,100,150,200'.",
    )
    parser.add_argument(
        "--ms_list",
        type=str,
        default=None,
        help="Comma-separated min_samples list, e.g., '5,10,15,30,50'.",
    )

    args = parser.parse_args()

    mcs_list = _parse_int_list(args.mcs_list) if args.mcs_list else None
    ms_list = _parse_int_list(args.ms_list) if args.ms_list else None

    best, _ = grid_search_hdbscan_params(
        dataset=args.dataset,
        model_name=args.model,
        device=args.device,
        min_cluster_size_list=mcs_list,
        min_samples_list=ms_list,
        noise_ratio_threshold=args.noise_ratio_threshold, 
    )

    if best is not None:
        print(
            f"\nSuggested defaults for {args.dataset}-{args.model}: "
            f"({best['min_cluster_size']}, {best['min_samples']})"
        )


