"""
neurolinkdb/index/kmeans.py
───────────────────────────
Pure-Python spherical K-Means clustering algorithm for unit-normalized vectors.

No external libraries (NumPy, SciPy, scikit-learn) required, fulfilling
the standalone database systems educational requirements.
"""

from __future__ import annotations

import random
from typing import List, Sequence, Tuple

from neurochat.embeddings.similarity import (
    cosine_similarity,
    euclidean_distance,
    l2_norm,
    normalize,
    validate_equal_dimensions,
)


class KMeans:
    """
    Spherical K-Means clustering implementation.

    Clusters vectors on the unit hypersphere using cosine similarity.

    Parameters
    ----------
    n_clusters:
        Number of clusters (centroids) to generate.
    max_iter:
        Maximum number of Lloyd's iteration cycles.
    tol:
        Convergence tolerance for centroid movement.
    random_seed:
        Seed for deterministic centroid initialization.
    """

    def __init__(
        self,
        n_clusters: int = 4,
        max_iter: int = 30,
        tol: float = 1e-4,
        random_seed: int = 42,
    ) -> None:
        if n_clusters <= 0:
            raise ValueError(f"n_clusters must be positive, got {n_clusters}")
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_seed = random_seed
        self.centroids: List[List[float]] = []

    def fit(self, vectors: Sequence[Sequence[float]]) -> KMeans:
        """
        Compute cluster centroids from input vectors.

        Parameters
        ----------
        vectors:
            Sequence of numerical vectors of identical dimensionality.

        Returns
        -------
        KMeans
            Self with populated centroids.
        """
        n_samples = len(vectors)
        if n_samples == 0:
            raise ValueError("Cannot fit KMeans on empty vectors.")

        dim = len(vectors[0])
        for v in vectors:
            if len(v) != dim:
                raise ValueError("All vectors must have identical dimensionality.")

        k = min(self.n_clusters, n_samples)
        # Ensure all training vectors are normalized
        norm_vectors = [normalize(v) for v in vectors]

        # 1. Deterministic K-Means++ style initialization
        self.centroids = self._init_centroids(norm_vectors, k, dim)

        # 2. Iteration loop
        for _ in range(self.max_iter):
            # Step A: Assignment
            assignments = self.predict(norm_vectors)

            # Step B: Centroid update
            new_centroids: List[List[float]] = []
            max_shift = 0.0

            for cluster_idx in range(k):
                cluster_pts = [
                    norm_vectors[i]
                    for i, a in enumerate(assignments)
                    if a == cluster_idx
                ]

                if not cluster_pts:
                    # Empty cluster: keep existing centroid or re-seed
                    new_centroids.append(self.centroids[cluster_idx])
                    continue

                # Compute mean vector
                sum_vec = [0.0] * dim
                for pt in cluster_pts:
                    for d in range(dim):
                        sum_vec[d] += pt[d]

                new_centroid = normalize(sum_vec)
                shift = euclidean_distance(self.centroids[cluster_idx], new_centroid)
                max_shift = max(max_shift, shift)
                new_centroids.append(new_centroid)

            self.centroids = new_centroids

            if max_shift < self.tol:
                break

        return self

    def predict(self, vectors: Sequence[Sequence[float]]) -> List[int]:
        """
        Assign each vector to its closest centroid by cosine similarity.
        """
        if not self.centroids:
            raise ValueError("KMeans model is not fitted yet.")

        assignments: List[int] = []
        for v in vectors:
            best_idx = 0
            best_sim = -float("inf")
            for idx, c in enumerate(self.centroids):
                sim = cosine_similarity(v, c)
                if sim > best_sim:
                    best_sim = sim
                    best_idx = idx
            assignments.append(best_idx)
        return assignments

    def _init_centroids(
        self, vectors: List[List[float]], k: int, dim: int
    ) -> List[List[float]]:
        """Select initial centroids using deterministic K-Means++ heuristic."""
        rng = random.Random(self.random_seed)
        first_idx = rng.randint(0, len(vectors) - 1)
        centroids: List[List[float]] = [list(vectors[first_idx])]

        while len(centroids) < k:
            # Compute distance of each point to its closest centroid
            distances: List[float] = []
            for v in vectors:
                max_sim = max(cosine_similarity(v, c) for c in centroids)
                # Cosine distance: 1 - cos(theta) >= 0
                dist = max(0.0, 1.0 - max_sim)
                distances.append(dist * dist)

            total_dist = sum(distances)
            if total_dist == 0.0:
                # All remaining vectors are identical to centroids; pick randomly
                rem = [i for i in range(len(vectors)) if vectors[i] not in centroids]
                if rem:
                    centroids.append(list(vectors[rem[0]]))
                else:
                    break
            else:
                target = rng.uniform(0.0, total_dist)
                cum = 0.0
                chosen = False
                for idx, d in enumerate(distances):
                    cum += d
                    if cum >= target:
                        centroids.append(list(vectors[idx]))
                        chosen = True
                        break
                if not chosen:
                    centroids.append(list(vectors[-1]))

        return centroids

