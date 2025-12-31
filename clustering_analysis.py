import torch
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import hdbscan
import warnings

# 抑制 sklearn 的 FutureWarning
warnings.filterwarnings('ignore', category=FutureWarning)


class ClusteringAnalyzer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.device
        self.model.eval()
        
    def extract_features(self, samples, batch_size=None):
        """
        Extract feature vectors and perform L2 normalization
        
        Args:
            samples: numpy array of shape (N, C, H, W)
        
        Returns:
            features: numpy array of shape (N, D) - L2 normalized features
        """
        if batch_size is None:
            batch_size = 512
        
        features_list = []
        
        with torch.no_grad():
            for i in range(0, len(samples), batch_size):
                batch = torch.FloatTensor(samples[i:i+batch_size]).to(self.device)
                batch_features = self.model.get_features(batch)
                features_list.append(batch_features.cpu().numpy())
                
        features = np.concatenate(features_list, axis=0)
        
        # L2 normalization
        features_norm = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / (features_norm + 1e-10)
        
        return features
    
    def perform_hdbscan_clustering_highdim(
        self,
        features,
        min_cluster_size=None,
        min_samples=None,
    ):
        """
        Perform HDBSCAN clustering in the high-dimensional feature space
        
        Args:
            features: high-dimensional features (N, D), e.g., the last but one layer features, already L2 normalized
        
        Returns:
            results: dict, containing the clusterer, labels and various clustering quality metrics
        """
        if min_cluster_size is None:
            min_cluster_size = getattr(self.config, "hdbscan_min_cluster_size", 50)
        if min_samples is None:
            min_samples = getattr(self.config, "hdbscan_min_samples", 10)

        # Perform HDBSCAN clustering in the high-dimensional feature space
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=0.0,
            metric='euclidean'   # equivalent to cosine distance on L2 normalized features
        )
        
        labels = clusterer.fit_predict(features)
        
        # filter out noise points
        mask = labels != -1
        valid_labels = labels[mask]
        valid_features = features[mask]
        
        n_clusters = len(np.unique(valid_labels))
        n_noise = np.sum(labels == -1)
        
        # calculate the clustering quality metrics
        if n_clusters > 1 and len(valid_features) > n_clusters:
            silhouette = silhouette_score(valid_features, valid_labels)
            davies_bouldin = davies_bouldin_score(valid_features, valid_labels)
            calinski_harabasz = calinski_harabasz_score(valid_features, valid_labels)
        else:
            silhouette = davies_bouldin = calinski_harabasz = 0.0
        
        
        if n_clusters > 0:
            cluster_sizes = np.bincount(valid_labels)
        else:
            cluster_sizes = np.array([])
        
        results = {
            'clusterer': clusterer,
            'labels': labels,               
            'valid_labels': valid_labels,    
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'silhouette_score': silhouette,
            'davies_bouldin_score': davies_bouldin,
            'calinski_harabasz_score': calinski_harabasz,
            'cluster_sizes': cluster_sizes
        }
        
        return results
    
