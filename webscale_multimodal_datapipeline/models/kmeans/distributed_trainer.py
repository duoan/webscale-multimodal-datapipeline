"""
Distributed KMeans Trainer using Ray

Implements distributed KMeans training with deterministic sharding:
- Driver initializes centroids and coordinates iterations
- Workers process fixed data shards (by file URLs)
- Each iteration: workers compute distances and assignments
- Driver merges results and updates centroids
"""

import os
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import ray


@ray.remote
class DataPrepWorker:
    """Ray worker for data preprocessing (parquet to numpy conversion)."""

    def __init__(self, worker_id: int, parquet_urls: list[str], output_dir: str):
        """Initialize data prep worker.

        Args:
            worker_id: Unique worker identifier
            parquet_urls: List of parquet file URLs to process
            output_dir: Directory to save preprocessed numpy files
        """
        self.worker_id = worker_id
        self.parquet_urls = parquet_urls
        self.output_dir = output_dir

    def process_shard(self) -> dict[str, Any]:
        """Process parquet files and convert to numpy format.

        Returns:
            Dictionary with worker_id, output paths, and stats
        """
        all_features = []
        all_ids = []

        for url in self.parquet_urls:
            # Load parquet file
            try:
                # Try using pyarrow first (faster for parquet)
                parquet_file = pq.ParquetFile(url)
                table = parquet_file.read()
                df = table.to_pandas()
            except Exception:
                # Fallback to pandas
                df = pd.read_parquet(url)

            # Extract id and feature columns
            if "id" not in df.columns or "feature" not in df.columns:
                raise ValueError(
                    f"Parquet file {url} must have 'id' and 'feature' columns. Found columns: {df.columns.tolist()}"
                )

            # Get IDs (strings)
            ids = df["id"].values
            all_ids.append(ids)

            # Convert feature to numpy array
            features = df["feature"].values
            feature_list = []
            for feat in features:
                if isinstance(feat, (list, tuple)):
                    feature_list.append(np.array(feat, dtype=np.float32))
                elif isinstance(feat, np.ndarray):
                    feature_list.append(feat.astype(np.float32))
                else:
                    raise ValueError(f"Cannot convert feature to numpy array: type {type(feat)}")

            if feature_list:
                features_array = np.vstack(feature_list)
            else:
                features_array = np.array([]).reshape(0, 0)

            if features_array.ndim == 1:
                features_array = features_array.reshape(1, -1)

            all_features.append(features_array)

        # Concatenate all features and IDs
        if all_features:
            features = np.vstack(all_features).astype(np.float32)
            ids = np.concatenate(all_ids) if all_ids else np.array([])
        else:
            features = np.array([]).reshape(0, 0)
            ids = np.array([])

        # Save preprocessed data
        os.makedirs(self.output_dir, exist_ok=True)
        features_file = os.path.join(self.output_dir, f"shard_{self.worker_id}_features.npy")
        ids_file = os.path.join(self.output_dir, f"shard_{self.worker_id}_ids.npy")

        np.save(features_file, features)
        np.save(ids_file, ids)

        return {
            "worker_id": self.worker_id,
            "num_samples": len(features),
            "features_file": features_file,
            "ids_file": ids_file,
        }


@ray.remote
class KMeansWorker:
    """Ray worker for distributed KMeans computation."""

    def __init__(self, worker_id: int, features_file: str, ids_file: str):
        """Initialize worker with preprocessed numpy data.

        Args:
            worker_id: Unique worker identifier
            features_file: Path to preprocessed features numpy file
            ids_file: Path to preprocessed IDs numpy file
        """
        self.worker_id = worker_id
        self.features_file = features_file
        self.ids_file = ids_file
        self.features: np.ndarray | None = None
        self.ids: np.ndarray | None = None

    def load_shard(self):
        """Load preprocessed features and IDs from numpy files.

        This loads the data once, then reuses it across iterations.
        """
        # Load preprocessed numpy files
        self.features = np.load(self.features_file).astype(np.float32)
        self.ids = np.load(self.ids_file)

    def compute_distances_and_assignments(
        self, centroids: np.ndarray, iteration: int, output_dir: str
    ) -> dict[str, Any]:
        """Compute distances and assignments for this shard.

        Args:
            centroids: Current centroid matrix of shape (n_clusters, n_features)
            iteration: Current iteration number
            output_dir: Directory to write output files

        Returns:
            Dictionary with worker_id, num_samples, and output file path
        """
        if self.features is None:
            self.load_shard()

        if len(self.features) == 0:
            # Empty shard
            output_file = os.path.join(output_dir, f"iter_{iteration}_shard_{self.worker_id}_assignments.npy")
            ids_file = os.path.join(output_dir, f"iter_{iteration}_shard_{self.worker_id}_ids.npy")
            np.save(output_file, np.array([]))
            np.save(ids_file, np.array([]))
            return {"worker_id": self.worker_id, "num_samples": 0, "output_file": output_file, "ids_file": ids_file}

        # Ensure features match centroid dimensions
        if self.features.shape[1] != centroids.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch: features have {self.features.shape[1]} dims, "
                f"centroids have {centroids.shape[1]} dims"
            )

        # Compute distances: (n_samples, n_clusters)
        # Use broadcasting: features (n, d) vs centroids (k, d)
        distances = np.sqrt(((self.features[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2))

        # Find closest centroid (assignment)
        assignments = np.argmin(distances, axis=1)
        min_distances = np.min(distances, axis=1)

        # Write assignments and IDs to files
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, f"iter_{iteration}_shard_{self.worker_id}_assignments.npy")
        np.save(output_file, assignments)

        # Save IDs for reference (mapping id -> assignment)
        ids_file = os.path.join(output_dir, f"iter_{iteration}_shard_{self.worker_id}_ids.npy")
        np.save(ids_file, self.ids)

        # Also save distances for potential use
        distances_file = os.path.join(output_dir, f"iter_{iteration}_shard_{self.worker_id}_distances.npy")
        np.save(distances_file, min_distances)

        return {
            "worker_id": self.worker_id,
            "num_samples": len(self.features),
            "output_file": output_file,
            "ids_file": ids_file,
            "distances_file": distances_file,
        }

    def compute_cluster_sums_and_counts(self, centroids: np.ndarray, iteration: int, output_dir: str) -> dict[str, Any]:
        """Compute cluster sums and counts for centroid update.

        This is called after assignments are computed.

        Args:
            centroids: Current centroid matrix
            iteration: Current iteration number
            output_dir: Directory containing assignment files

        Returns:
            Dictionary with cluster sums and counts
        """
        if self.features is None:
            self.load_shard()

        if len(self.features) == 0:
            n_features = centroids.shape[1]
            return {
                "worker_id": self.worker_id,
                "cluster_sums": np.zeros((centroids.shape[0], n_features)),
                "cluster_counts": np.zeros(centroids.shape[0]),
            }

        # Load assignments for this shard
        assignments_file = os.path.join(output_dir, f"iter_{iteration}_shard_{self.worker_id}_assignments.npy")
        assignments = np.load(assignments_file)

        n_clusters = centroids.shape[0]
        n_features = self.features.shape[1]

        # Compute sums and counts per cluster
        cluster_sums = np.zeros((n_clusters, n_features))
        cluster_counts = np.zeros(n_clusters)

        for cluster_id in range(n_clusters):
            mask = assignments == cluster_id
            if np.any(mask):
                cluster_sums[cluster_id] = self.features[mask].sum(axis=0)
                cluster_counts[cluster_id] = np.sum(mask)

        return {"worker_id": self.worker_id, "cluster_sums": cluster_sums, "cluster_counts": cluster_counts}


class DistributedKMeansTrainer:
    """Distributed KMeans trainer using Ray."""

    def __init__(
        self,
        n_clusters: int,
        data_urls: list[str],
        n_workers: int = 4,
        max_iterations: int = 100,
        tolerance: float = 1e-4,
        random_state: int = 42,
        output_dir: str = "./kmeans_training",
    ):
        """Initialize distributed KMeans trainer.

        Args:
            n_clusters: Number of clusters
            data_urls: List of data file URLs (features)
            n_workers: Number of Ray workers
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance (centroid change)
            random_state: Random seed for centroid initialization
            output_dir: Directory to write intermediate results
        """
        self.n_clusters = n_clusters
        self.data_urls = data_urls
        self.n_workers = n_workers
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_state = random_state
        self.output_dir = output_dir

        self.centroids: np.ndarray | None = None
        self.workers: list[Any] = []
        self._shard_urls: list[list[str]] = []
        self._preprocessed_dir = os.path.join(output_dir, "preprocessed")
        self._preprocessed_paths: list[dict[str, str]] = []  # [{features_file, ids_file}, ...]

    def _create_shards(self) -> list[list[str]]:
        """Create deterministic shards from data URLs.

        Returns:
            List of shard URL lists (one per worker)
        """
        # Deterministic sharding: assign each URL to a worker by hash
        shards: list[list[str]] = [[] for _ in range(self.n_workers)]

        for url in self.data_urls:
            # Deterministic assignment based on URL hash
            worker_id = hash(url) % self.n_workers
            shards[worker_id].append(url)

        return shards

    def _preprocess_data(self) -> list[dict[str, str]]:
        """Preprocess parquet files to numpy format using DataPrepWorkers.

        Returns:
            List of dictionaries with features_file and ids_file paths
        """
        print("Starting data preprocessing (parquet -> numpy)...")

        # Create shards for preprocessing
        shard_urls = self._create_shards()

        # Create DataPrepWorkers
        prep_workers = []
        for worker_id in range(self.n_workers):
            worker = DataPrepWorker.remote(
                worker_id=worker_id, parquet_urls=shard_urls[worker_id], output_dir=self._preprocessed_dir
            )
            prep_workers.append(worker)

        # Process all shards in parallel
        print(f"Processing {self.n_workers} shards in parallel...")
        futures = [worker.process_shard.remote() for worker in prep_workers]
        results = ray.get(futures)

        # Sort by worker_id to ensure consistent ordering
        results.sort(key=lambda x: x["worker_id"])

        # Extract file paths
        preprocessed_paths = []
        total_samples = 0
        for result in results:
            preprocessed_paths.append({"features_file": result["features_file"], "ids_file": result["ids_file"]})
            total_samples += result["num_samples"]
            print(f"  Shard {result['worker_id']}: {result['num_samples']} samples -> {result['features_file']}")

        print(f"Data preprocessing completed. Total: {total_samples} samples")
        return preprocessed_paths

    def _initialize_centroids(
        self, preprocessed_paths: list[dict[str, str]] | None = None, n_samples: int = 1000
    ) -> np.ndarray:
        """Initialize centroids by sampling from preprocessed numpy data.

        Args:
            preprocessed_paths: Optional list of preprocessed file paths (if None, uses first few shards)
            n_samples: Number of samples to use for initialization

        Returns:
            Initial centroid matrix of shape (n_clusters, n_features)
        """
        # Use preprocessed data if available
        if preprocessed_paths is None:
            preprocessed_paths = self._preprocessed_paths[: min(len(self._preprocessed_paths), 5)]

        # Load first shard to determine feature dimension
        if not preprocessed_paths:
            raise ValueError("No preprocessed data available for centroid initialization")

        sample_features = np.load(preprocessed_paths[0]["features_file"])
        n_features = sample_features.shape[1]

        # Initialize centroids by sampling from data (KMeans++ style)
        np.random.seed(self.random_state)

        # Sample n_samples features randomly from first few shards
        sampled_features = []
        for path_dict in preprocessed_paths:
            features = np.load(path_dict["features_file"])
            sampled_features.append(features)

            total_loaded = sum(len(f) for f in sampled_features)
            if total_loaded >= n_samples:
                break

        if sampled_features:
            sampled_features_array = np.vstack(sampled_features)
            # Use random sampling from data for initialization
            if len(sampled_features_array) >= self.n_clusters:
                indices = np.random.choice(len(sampled_features_array), self.n_clusters, replace=False)
                centroids = sampled_features_array[indices].astype(np.float32)
            else:
                # Not enough samples, use random initialization
                centroids = np.random.randn(self.n_clusters, n_features).astype(np.float32)
        else:
            # No samples found, use random initialization
            centroids = np.random.randn(self.n_clusters, n_features).astype(np.float32)

        # Save initial centroids
        centroids_dir = os.path.join(self.output_dir, "iter_0")
        os.makedirs(centroids_dir, exist_ok=True)
        centroids_file = os.path.join(centroids_dir, "centroids.npy")
        np.save(centroids_file, centroids)

        return centroids

    def _check_convergence(self, old_centroids: np.ndarray, new_centroids: np.ndarray) -> bool:
        """Check if centroids have converged.

        Args:
            old_centroids: Previous iteration centroids
            new_centroids: Current iteration centroids

        Returns:
            True if converged, False otherwise
        """
        centroid_shift = np.linalg.norm(new_centroids - old_centroids)
        return centroid_shift < self.tolerance

    def _update_centroids(self, iteration: int) -> tuple[np.ndarray, bool]:
        """Update centroids based on cluster assignments from all workers.

        Args:
            iteration: Current iteration number

        Returns:
            Tuple of (new_centroids, converged)
        """
        # Load current centroids
        old_centroids_file = os.path.join(self.output_dir, f"iter_{iteration}", "centroids.npy")
        old_centroids = np.load(old_centroids_file)

        # Ask workers to compute cluster sums and counts
        futures = []
        for worker in self.workers:
            future = worker.compute_cluster_sums_and_counts.remote(old_centroids, iteration, self.output_dir)
            futures.append(future)

        results = ray.get(futures)

        # Merge results from all workers
        n_clusters, n_features = old_centroids.shape
        total_sums = np.zeros((n_clusters, n_features))
        total_counts = np.zeros(n_clusters)

        for result in results:
            total_sums += result["cluster_sums"]
            total_counts += result["cluster_counts"]

        # Update centroids: new_centroid = sum / count
        new_centroids = old_centroids.copy()
        for cluster_id in range(n_clusters):
            if total_counts[cluster_id] > 0:
                new_centroids[cluster_id] = total_sums[cluster_id] / total_counts[cluster_id]

        # Save new centroids
        next_iter = iteration + 1
        next_centroids_dir = os.path.join(self.output_dir, f"iter_{next_iter}")
        os.makedirs(next_centroids_dir, exist_ok=True)
        next_centroids_file = os.path.join(next_centroids_dir, "centroids.npy")
        np.save(next_centroids_file, new_centroids)

        # Check convergence
        converged = self._check_convergence(old_centroids, new_centroids)

        return new_centroids, converged

    def train(self) -> np.ndarray:
        """Train distributed KMeans model.

        Returns:
            Final centroid matrix
        """
        # Step 1: Preprocess data (parquet -> numpy)
        self._preprocessed_paths = self._preprocess_data()

        # Step 2: Initialize KMeans workers with preprocessed data
        self.workers = []
        for worker_id, path_dict in enumerate(self._preprocessed_paths):
            worker = KMeansWorker.remote(
                worker_id=worker_id, features_file=path_dict["features_file"], ids_file=path_dict["ids_file"]
            )
            self.workers.append(worker)

        # Step 3: Initialize centroids
        print("Initializing centroids...")
        self.centroids = self._initialize_centroids(preprocessed_paths=self._preprocessed_paths)
        print(f"Initialized {self.n_clusters} centroids with {self.centroids.shape[1]} features")

        # Training loop
        for iteration in range(self.max_iterations):
            iter_num = iteration  # Current iteration number (0-indexed)
            print(f"\nIteration {iter_num + 1}/{self.max_iterations}")

            # Load current centroids for this iteration
            if iteration == 0:
                centroids_file = os.path.join(self.output_dir, "iter_0", "centroids.npy")
            else:
                centroids_file = os.path.join(self.output_dir, f"iter_{iter_num}", "centroids.npy")
            self.centroids = np.load(centroids_file)

            # Step 1: Workers compute distances and assignments
            print("Computing distances and assignments...")
            futures = []
            for worker in self.workers:
                future = worker.compute_distances_and_assignments.remote(self.centroids, iter_num, self.output_dir)
                futures.append(future)

            assignment_results = ray.get(futures)
            total_samples = sum(r["num_samples"] for r in assignment_results)
            print(f"Processed {total_samples} samples across {len(assignment_results)} workers")

            # Step 2: Update centroids based on assignments
            print("Updating centroids...")
            new_centroids, converged = self._update_centroids(iter_num)

            # Check convergence
            if converged:
                print(f"Converged at iteration {iter_num + 1}")
                break

            self.centroids = new_centroids

            # Compute inertia for monitoring
            inertia = self._compute_inertia(iter_num)
            print(f"Inertia: {inertia:.4f}")

        print(f"\nTraining completed. Final centroids saved to {self.output_dir}")
        return self.centroids

    def _compute_inertia(self, iteration: int) -> float:
        """Compute inertia (sum of squared distances) for monitoring.

        Args:
            iteration: Current iteration number

        Returns:
            Total inertia
        """
        # Load min distances from all shards
        total_inertia = 0.0
        for worker_id in range(self.n_workers):
            distances_file = os.path.join(self.output_dir, f"iter_{iteration}_shard_{worker_id}_distances.npy")
            if os.path.exists(distances_file):
                distances = np.load(distances_file)
                total_inertia += (distances**2).sum()
        return total_inertia

    def save(self, model_path: str):
        """Save trained model.

        Args:
            model_path: Path to save the model
        """
        import joblib

        os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else ".", exist_ok=True)

        model_data = {"centroids": self.centroids, "n_clusters": self.n_clusters, "output_dir": self.output_dir}

        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
