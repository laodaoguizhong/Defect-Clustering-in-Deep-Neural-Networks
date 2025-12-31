import os
import json
import argparse
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from config import config as exp_config
from adversarial_generator import AdversarialSampleGenerator
from clustering_analysis import ClusteringAnalyzer
from models import get_model

class ClusteringIntensityCalculator:
    """
    The same definition of λ as used in the experiment:
      - Calculate the covariance volume in the PCA subspace of the cluster;
      - Add regularization eps * I;
      - Use slogdet to calculate log(det);
      - Return the clustering intensity for each cluster and the global intensity.
    """

    @staticmethod
    def _cluster_log_volume(
        subset: np.ndarray,
        pca_dim: int = 30,
        eps: float = 1e-5,
        min_cluster_size: int = 5,
    ) -> Tuple[float, float]:
        n_k, D = subset.shape
        if n_k < min_cluster_size:
            return None, None

        dim = min(pca_dim, D, n_k - 1)
        if dim < 1:
            return None, None

        pca = PCA(n_components=dim)
        Xr = pca.fit_transform(subset)  # (n_k, dim)

        cov = np.cov(Xr, rowvar=False)
        cov = cov + eps * np.eye(cov.shape[0], dtype=cov.dtype)

        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            return None, None

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
        Calculate the clustering intensity for each cluster and the global intensity.

        Returns:
            per_cluster: {cid: {'size', 'volume', 'log_volume', 'lambda', 'log_lambda'}}
            global_lambda: float = sum|C_k| / sum vol(C_k)
            avg_log_lambda: float = the average log lambda of each cluster
        """
        clusters = [c for c in np.unique(labels) if c != -1]
        per_cluster: Dict[int, Dict[str, Any]] = {}
        total_size, total_vol = 0.0, 0.0
        log_lambdas: List[float] = []

        for cid in clusters:
            mask = (labels == cid)
            size = int(mask.sum())
            if size < min_cluster_size:
                continue

            subset = features[mask]
            log_vol, vol = ClusteringIntensityCalculator._cluster_log_volume(
                subset, pca_dim=pca_dim, eps=eps, min_cluster_size=min_cluster_size
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




def _load_model(dataset: str, model_name: str, device: torch.device):
    model_obj = get_model(model_name, dataset)
    ckpt = f"./checkpoints/{dataset}_{model_name}.pth"
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
    device: torch.device,
    num_samples: int = None,
    force_regenerate: bool = False,
):
    exp_config.device = str(device)

    if num_samples is None:
        num_samples = (
            exp_config.num_samples_mnist
            if dataset == "mnist"
            else exp_config.num_samples_cifar10
        )

    path = os.path.join(
        exp_config.output_dir,
        dataset,
        model_name,
        "adversarial_library.npz",
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path) and not force_regenerate:
        print(f"[INFO] Loading cached adversarial library from: {path}")
        return dict(np.load(path, allow_pickle=True))

    print(f"[INFO] Generating adversarial library for {dataset}-{model_name} ...")
    loader = _load_dataset(dataset)
    gen = AdversarialSampleGenerator(model, exp_config, dataset)
    lib = gen.build_adversarial_library(loader, num_samples=num_samples)

    np.savez(path, **lib)
    print(f"[INFO] Adversarial library saved to: {path}")
    return lib


def extract_features_for_baseline(
    dataset: str,
    model_name: str,
    device: torch.device,
    num_samples: int = None,
    force_regenerate_adv: bool = False,
):
    model = _load_model(dataset, model_name, device)
    adv_lib = _load_or_generate_adv(
        model, dataset, model_name, device,
        num_samples=num_samples,
        force_regenerate=force_regenerate_adv,
    )

    analyzer = ClusteringAnalyzer(model, exp_config)
    features = analyzer.extract_features(adv_lib["samples"])
    return features




def build_null_baseline(
    features: np.ndarray,
    cluster_sizes: List[int],
    n_runs: int = 1000,
    pca_dim: int = 30,
    eps: float = 1e-5,
    min_cluster_size: int = 5,
    seed: int = 42,
):
    """
     Use "random pseudo-clusters" to construct the baseline:
      - Keep the number of samples in each cluster consistent with the real cluster;
      - Randomly sample from all features to construct pseudo-clusters;
      - Calculate global_log_lambda for each run.

    Returns: 
        null_global_logs: (n_runs,) 
    """
    N = features.shape[0]
    total_size = int(sum(cluster_sizes))
    if total_size > N:
        raise ValueError(
            f"Total cluster size {total_size} exceeds total samples {N}."
        )

    rng = np.random.default_rng(seed)

    null_global_logs = []

    for b in range(n_runs):
        perm = rng.permutation(N)[:total_size]
        start = 0

        vols = []
        valid = True

        for size in cluster_sizes:
            idx = perm[start:start + size]
            subset = features[idx]
            start += size

            log_vol, vol = ClusteringIntensityCalculator._cluster_log_volume(
                subset,
                pca_dim=pca_dim,
                eps=eps,
                min_cluster_size=min_cluster_size,
            )
            if log_vol is None or vol <= 0:
                valid = False
                break

            vols.append(vol)

        if not valid:
            continue

        vols = np.array(vols, dtype=float)
        total_vol = float(np.sum(vols))
        log_global = float(np.log(total_size) - np.log(total_vol))

        null_global_logs.append(log_global)

    null_global_logs = np.array(null_global_logs, dtype=float)

    return null_global_logs


def compute_z_and_p(real_value: float, null_samples: np.ndarray) -> Tuple[float, float]:
    """
    One-tailed test: P(null ≥ real) is regarded as the p-value.
    The z-score uses the mean and standard deviation of the null distribution.
    """
    mu = float(np.mean(null_samples))
    sigma = float(np.std(null_samples, ddof=1)) if len(null_samples) > 1 else 0.0

    if sigma > 0:
        z = (real_value - mu) / sigma
    else:
        z = float("inf") if real_value > mu else float("-inf")

    greater_equal = np.sum(null_samples >= real_value)
    p = (greater_equal + 1.0) / (len(null_samples) + 1.0)

    return float(z), float(p)



def plot_significance(
    null_samples: np.ndarray,
    real_value: float,
    save_path: str,
    title: str,
    xlabel: str,
):
    plt.figure()
    plt.hist(null_samples, bins=30, alpha=0.7)
    plt.axvline(real_value, linestyle="--")  
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Plot saved to {save_path}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["mnist", "cifar10"],
        help="Dataset name (if not specified, run all)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (if not specified, run all)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--results_json",
        type=str,
        default=None,
        help="Path to experiment_one_results.json; "
             "if not provided, use default ./results/{dataset}/{model}/experiment_1/experiment_one_results.json"
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=1000,
        help="Number of baseline runs"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for baseline sampling"
    )
    parser.add_argument(
        "--force_regenerate_adv",
        action="store_true",
        help="Force regenerate adversarial library even if cached"
    )

    args = parser.parse_args()
    device = torch.device(args.device)

    model_map = {
        "mnist": ["lenet1", "lenet4", "lenet5"],
        "cifar10": ["resnet20", "resnet50"],
    }

    if args.dataset and args.model:
        combinations = [(args.dataset, args.model)]
    else:
        combinations = [(ds, m) for ds, models in model_map.items() for m in models]

    all_results = []

    for dataset, model_name in combinations:
        print(f"\n{'='*60}")
        print(f"🔬 Running baseline analysis: {dataset} / {model_name}")
        print(f"{'='*60}")

        if args.results_json is None:
            results_json = os.path.join(
                "results",
                dataset,
                model_name,
                "experiment_1",
                "experiment_one_results.json",
            )
        else:
            results_json = args.results_json

        if not os.path.exists(results_json):
            print(f"[WARN] Results JSON not found: {results_json}, skipping...")
            continue

        with open(results_json, "r", encoding="utf-8") as f:
            res = json.load(f)

        cluster_details = res["cluster_details"]
        cluster_sizes = [int(c["size"]) for c in cluster_details]

        print(f"[INFO] Cluster sizes: {cluster_sizes}")

        features = extract_features_for_baseline(
            dataset=dataset,
            model_name=model_name,
            device=device,
            num_samples=None,
            force_regenerate_adv=args.force_regenerate_adv,
        )
        print(f"[INFO] Features shape: {features.shape}")

        pca_dim = getattr(exp_config, "lambda_pca_dim", 30)
        eps = getattr(exp_config, "lambda_cov_eps", 1e-5)
        min_cluster_size = getattr(exp_config, "lambda_min_cluster_size", 5)

        print(f"[INFO] Building null baseline with {args.n_runs} runs ...")
        null_global_logs = build_null_baseline(
            features=features,
            cluster_sizes=cluster_sizes,
            n_runs=args.n_runs,
            pca_dim=pca_dim,
            eps=eps,
            min_cluster_size=min_cluster_size,
            seed=args.seed,
        )

        print(f"[INFO] Collected {len(null_global_logs)} valid baseline samples.")

        # 计算真实"全局 log λ"
        real_global_lambda = float(res["clustering_intensity"]["global_lambda"])
        real_global_log_lambda = float(np.log(real_global_lambda)) if real_global_lambda > 0 else float("-inf")

        z_global, p_global = compute_z_and_p(real_global_log_lambda, null_global_logs)
        print(f"[RESULT] Global log lambda:")
        print(f"         real = {real_global_log_lambda:.4f}")
        print(f"         z-score = {z_global:.4f}, p-value = {p_global:.6f}")


        save_dir = os.path.join("results", dataset, model_name, "experiment_1")
        os.makedirs(save_dir, exist_ok=True)

        global_plot_path = os.path.join(save_dir, "baseline_global_log_lambda.png")
        title_global = f"{dataset}/{model_name} - Global log lambda (z={z_global:.2f}, p={p_global:.2e})"
        plot_significance(
            null_samples=null_global_logs,
            real_value=real_global_log_lambda,
            save_path=global_plot_path,
            title=title_global,
            xlabel="Global log lambda",
        )


        json_results = {
            "dataset": dataset,
            "model": model_name,
            "experiment_config": {
                "n_runs": args.n_runs,
                "seed": args.seed,
                "pca_dim": pca_dim,
                "eps": eps,
                "min_cluster_size": min_cluster_size,
            },
            "global_log_lambda": {
                "real_value": float(real_global_log_lambda),
                "z_score": float(z_global),
                "p_value": float(p_global),
                "null_distribution": {
                    "mean": float(np.mean(null_global_logs)),
                    "std": float(np.std(null_global_logs, ddof=1)),
                    "min": float(np.min(null_global_logs)),
                    "max": float(np.max(null_global_logs)),
                    "median": float(np.median(null_global_logs)),
                    "q25": float(np.percentile(null_global_logs, 25)),
                    "q75": float(np.percentile(null_global_logs, 75)),
                    "n_samples": int(len(null_global_logs)),
                },
            },
            "cluster_info": {
                "n_clusters": len(cluster_sizes),
                "cluster_sizes": cluster_sizes,
                "total_samples": int(sum(cluster_sizes)),
            },
        }

        json_path = os.path.join(save_dir, "baseline_lambda_analysis.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        print(f"[INFO] JSON results saved to: {json_path}")


        all_results.append(json_results)


    if len(all_results) > 0:
        first_result = all_results[0]
        consolidated_results = {
            "summary": {
                "total_models": len(all_results),
                "experiment_config": first_result["experiment_config"],
            },
            "results": all_results,
        }


        consolidated_path = os.path.join("results", "baseline_lambda_analysis_consolidated.json")
        os.makedirs(os.path.dirname(consolidated_path), exist_ok=True)
        with open(consolidated_path, "w", encoding="utf-8") as f:
            json.dump(consolidated_results, f, indent=2, ensure_ascii=False)
        print(f"\n[INFO] Consolidated JSON results saved to: {consolidated_path}")

    print(f"\n{'='*60}")
    print("✅ All baseline analyses completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
