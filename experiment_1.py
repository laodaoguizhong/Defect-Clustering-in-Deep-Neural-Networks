import os
import json
from typing import Dict, Any, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from sklearn.decomposition import PCA

from clustering_analysis import ClusteringAnalyzer
from adversarial_generator import AdversarialSampleGenerator
from models import get_model
from config import config as exp_config


class ClusteringIntensityCalculator:
    """
    Based on the definition of clustering intensity λ in the paper, implement a numerically more robust version:
        - Calculate volume only over the effective PCA subspace of the cluster;
        - Apply mild regularisation to the covariance matrix;
        - Use `slogdet` to compute `log(det)`, avoiding numerical overflow;
        - Return detailed information for each cluster alongside the global λ (∑|C_k| / ∑ vol(C_k)).

    """

    @staticmethod
    def _cluster_log_volume(
        subset: np.ndarray,
        pca_dim: int = 30,
        eps: float = 1e-5,
        min_cluster_size: int = 5,
    ) -> Tuple[float, float]:
        """
        Calculate log(volume) in the PCA subspace of the cluster.

        Args:
            subset: (n_k, D) features in the cluster
            pca_dim: PCA target dimension upper limit
            eps: covariance matrix regularization strength
            min_cluster_size: minimum cluster size allowed to calculate volume

        Returns:
            (log_vol, vol) if the volume cannot be stably calculated, return (None, None)
        """
        n_k, D = subset.shape
        if n_k < min_cluster_size:
            return None, None
        dim = min(pca_dim, D, n_k - 1)
        if dim < 1:
            return None, None

        # PCA dimensionality reduction to the effective subspace
        pca = PCA(n_components=dim)
        Xr = pca.fit_transform(subset)  # (n_k, dim)

        # covariance + regularization
        cov = np.cov(Xr, rowvar=False)  # (dim, dim)
        cov = cov + eps * np.eye(cov.shape[0], dtype=cov.dtype)

        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            # the covariance matrix is not positive definite, so skip this cluster
            return None, None

        # volume ~ sqrt(det(cov)) → log_vol = 0.5 * logdet
        log_vol = 0.5 * float(logdet)
        vol = float(np.exp(log_vol))
        return log_vol, vol

    @staticmethod
    def calculate_clustering_intensity(
        features: np.ndarray,
        labels: np.ndarray,
        pca_dim: int = 30,
        eps: float = 1e-5,
        min_cluster_size: int = 5,
    ) -> Tuple[Dict[int, Dict[str, Any]], float, float]:
        """
        Calculate the clustering intensity λ for each cluster and the global λ.

        Args:
            features: (N, D) L2 normalized high-dimensional features
            labels: (N,) HDBSCAN cluster labels (-1 for noise)
            pca_dim: PCA target dimension upper limit
            eps: covariance matrix regularization strength
            min_cluster_size: minimum cluster size allowed to calculate volume

        Returns:
            per_cluster: {cid: {'size', 'volume', 'log_volume', 'lambda', 'log_lambda'}}
            global_lambda: float, the global λ defined in the paper = ∑|C_k| / ∑ vol(C_k)
            avg_log_lambda: float, the average log λ of the effective clusters (optional for analysis)
        """
        clusters = [c for c in np.unique(labels) if c != -1]
        per_cluster: Dict[int, Dict[str, Any]] = {}
        total_size, total_vol = 0.0, 0.0
        log_lambdas = []

        for cid in clusters:
            mask = (labels == cid)
            size = int(mask.sum())
            if size < min_cluster_size:
                # filter out small clusters to avoid unstable volume estimation
                continue

            subset = features[mask]
            log_vol, vol = ClusteringIntensityCalculator._cluster_log_volume(
                subset,
                pca_dim=pca_dim,
                eps=eps,
                min_cluster_size=min_cluster_size,
            )
            if log_vol is None or vol <= 0:
                continue

            log_lambda = float(np.log(size) - log_vol)
            lam = float(np.exp(log_lambda))

            per_cluster[int(cid)] = {
                "size": size,
                "volume": vol,
                "log_volume": log_vol,
                "lambda": lam,
                "log_lambda": log_lambda,
            }

            total_size += size
            total_vol += vol
            log_lambdas.append(log_lambda)

        global_lambda = float(total_size / total_vol) if total_vol > 0 else 0.0
        avg_log_lambda = float(np.mean(log_lambdas)) if log_lambdas else 0.0
        return per_cluster, global_lambda, avg_log_lambda


def _load_model(dataset: str, model: str, device: torch.device):
    model_obj = get_model(model, dataset)
    ckpt = f"./checkpoints/{dataset}_{model}.pth"
    ckpt_data = torch.load(ckpt, map_location=device)
    model_obj.load_state_dict(ckpt_data["model_state_dict"])
    return model_obj.to(device).eval()


def _load_dataset(dataset: str) -> DataLoader:
    if dataset == "mnist":
        ds = datasets.MNIST(
            exp_config.data_root,
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
    else:
        ds = datasets.CIFAR10(
            exp_config.data_root,
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
    return DataLoader(ds, batch_size=128, shuffle=False, num_workers=exp_config.num_workers)


def _load_or_generate_adv(
    model: torch.nn.Module,
    dataset: str,
    model_name: str,
    cfg: Dict[str, Any],
    device: torch.device,
):
    """
    generate the adversarial library.
    """
    exp_config.device = str(device)
    if cfg.get("hdbscan_min_cluster_size"):
        exp_config.hdbscan_min_cluster_size = cfg["hdbscan_min_cluster_size"]
    if cfg.get("hdbscan_min_samples"):
        exp_config.hdbscan_min_samples = cfg["hdbscan_min_samples"]

    num_samples = cfg.get("num_adversarial_samples")
    if num_samples is None:
        num_samples = (
            exp_config.num_samples_mnist
            if dataset == "mnist"
            else exp_config.num_samples_cifar10
        )

    # the path to save the adversarial library
    path = os.path.join(
        exp_config.output_dir,
        dataset,
        model_name,
        "adversarial_library.npz",
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # if the adversarial library already exists and is not forced to be regenerated, then load it
    if os.path.exists(path) and not cfg.get("force_regenerate", False):
        print(f" Loading cached adversarial library from: {path}")
        return dict(np.load(path, allow_pickle=True))

    print(f" Generating adversarial library for {dataset}-{model_name} ...")

    loader = _load_dataset(dataset)
    gen = AdversarialSampleGenerator(model, exp_config, dataset)
    lib = gen.build_adversarial_library(loader, num_samples=num_samples)

    np.savez(path, **lib)
    print(f" Adversarial library saved to: {path}")
    return lib


def run_experiment_one(cfg: Dict[str, Any]):
    dataset = cfg["dataset"]
    model_name = cfg["model"]
    device = torch.device(cfg["device"])

    print(f"===== Running Experiment 1: RQ1 on {dataset}-{model_name} ({device}) =====")
    #Default parameters for HDBSCAN after hdbscan_param_search.py grid search
    hdbscan_defaults = {"mnist": {"lenet1": (100, 100),"lenet4": (200, 30),"lenet5": (200, 100),},"cifar10": {"resnet20": (200, 30),"resnet50": (200, 20),},}
    if dataset in hdbscan_defaults and model_name in hdbscan_defaults[dataset]:
        default_min_cluster_size, default_min_samples = hdbscan_defaults[dataset][model_name]
        if cfg.get("hdbscan_min_cluster_size") is None:
            exp_config.hdbscan_min_cluster_size = default_min_cluster_size
        if cfg.get("hdbscan_min_samples") is None:
            exp_config.hdbscan_min_samples = default_min_samples

    # 1. load the model & the adversarial library
    model = _load_model(dataset, model_name, device)
    adv_lib = _load_or_generate_adv(model, dataset, model_name, cfg, device)

    # 2. feature extraction + high-dimensional clustering
    analyzer = ClusteringAnalyzer(model, exp_config)
    features = analyzer.extract_features(adv_lib["samples"])
    cluster_res = analyzer.perform_hdbscan_clustering_highdim(features)

    labels = cluster_res["labels"]
    cluster_res["features"] = features
    cluster_res["cluster_labels"] = labels
    cluster_res["noise_ratio"] = cluster_res["n_noise"] / len(labels)

    # 3. calculate the clustering intensity λ
    per_cluster, global_lambda, avg_log_lambda = ClusteringIntensityCalculator.calculate_clustering_intensity(
        features,
        labels,
        pca_dim=getattr(exp_config, "lambda_pca_dim", 30),
        eps=getattr(exp_config, "lambda_cov_eps", 1e-5),
        min_cluster_size=getattr(exp_config, "lambda_min_cluster_size", 5),
    )

    # 4. output the results
    save_dir = f"./results/{dataset}/{model_name}/experiment_1"
    os.makedirs(save_dir, exist_ok=True)

    # 5. construct the result dictionary
    results: Dict[str, Any] = {
        "dataset": dataset,
        "model": model_name,
        "adversarial_library_size": int(len(adv_lib["samples"])),
        "clustering_metrics": {
            "n_clusters": int(cluster_res["n_clusters"]),
            "noise_ratio_eta": float(cluster_res["noise_ratio"]),
            "silhouette_score_delta": float(cluster_res["silhouette_score"]),
            "davies_bouldin_index": float(
                cluster_res.get("davies_bouldin_score", -1.0)
            ),
            "calinski_harabasz_score": float(
                cluster_res.get("calinski_harabasz_score", -1.0)
            ),
        },
        "clustering_intensity": {
            "global_lambda": float(global_lambda),
            "average_log_lambda": float(avg_log_lambda),
            "per_cluster_lambda": {
                int(cid): float(info["lambda"])
                for cid, info in per_cluster.items()
            },
        },
        "cluster_details": [],
    }

    # detailed information for each cluster
    for cid, info in sorted(per_cluster.items(), key=lambda x: int(x[0])):
        size = int(info["size"])
        results["cluster_details"].append(
            {
                "cluster_id": int(cid),
                "size": size,
                "lambda": float(info["lambda"]),
                "log_lambda": float(info["log_lambda"]),
                "volume": float(info["volume"]),
                "percentage": size / len(adv_lib["samples"]),
            }
        )
    def _json_friendly(obj):
        if isinstance(obj, dict):
            return {k: _json_friendly(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_json_friendly(i) for i in obj]
        if isinstance(obj, (float, np.floating)):
            if np.isinf(obj):
                return "Infinity" if obj > 0 else "-Infinity"
            if np.isnan(obj):
                return "NaN"
            return float(obj)
        if isinstance(obj, (int, np.integer)):
            return int(obj)
        return obj

    out_path = os.path.join(save_dir, "experiment_one_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_json_friendly(results), f, indent=2, ensure_ascii=False)

    print(f" Experiment 1 results saved to: {out_path}")
    return results


def consolidate_experiment_1(output_path: str = "./results/experiment_1.json"):
    """
    collect the results of five models and save them to the root directory of results.
    """
    combos = [
        ("mnist", "lenet1"),
        ("mnist", "lenet4"),
        ("mnist", "lenet5"),
        ("cifar10", "resnet20"),
        ("cifar10", "resnet50"),
    ]
    merged = []
    for ds, model in combos:
        path = os.path.join("results", ds, model, "experiment_1", "experiment_1.json")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            try:
                merged.append(json.load(f))
            except Exception as e:
                print(f" Failed to load {path}: {e}")
                continue

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f" Consolidated results saved to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["mnist", "cifar10"],
        help="Dataset name",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (e.g., lenet1, lenet4, lenet5, resnet20, resnet50)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--num_adversarial_samples",
        type=int,
        default=None,
        help="Number of adversarial samples to generate (per dataset-model)",
    )
    parser.add_argument(
        "--hdbscan_min_cluster_size",
        type=int,
        default=None,
        help="Override HDBSCAN min_cluster_size in config",
    )
    parser.add_argument(
        "--hdbscan_min_samples",
        type=int,
        default=None,
        help="Override HDBSCAN min_samples in config",
    )
    parser.add_argument(
        "--force_regenerate",
        action="store_true",
        help="Force regenerate adversarial library even if cached",
    )

    args = vars(parser.parse_args())

    # default model combinations (if not explicitly specified dataset and model)
    model_map = {
        "mnist": ["lenet1", "lenet4", "lenet5"],
        "cifar10": ["resnet20", "resnet50"],
    }

    # If user explicitly provides model and dataset → run once
    if args["dataset"] and args["model"]:
        run_experiment_one(args)
    else:
        # Traverse all dataset-model combinations
        for ds, models in model_map.items():
            for m in models:
                cfg = args.copy()
                cfg["dataset"] = ds
                cfg["model"] = m
                run_experiment_one(cfg)

    # consolidate the results of five models and save them to the root directory of results.
    consolidate_experiment_1()
