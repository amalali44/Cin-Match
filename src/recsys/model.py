"""Small-data mean-centered SVD baseline, portable artifacts, and inference."""

import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from recsys import __version__
from recsys.metrics import ranking_metrics
from recsys.ml_data import Ratings, read_split, read_titles, sha256, source_signature

MAX_K = 100
MAX_FACTOR_BYTES = 256 * 1024 * 1024


def validate_k(k):
    if type(k) is not int or not 1 <= k <= MAX_K:
        raise ValueError(f"k must be between 1 and {MAX_K}")


@dataclass
class RecommendationModel:
    users: np.ndarray
    movies: np.ndarray
    titles: np.ndarray
    user_factors: np.ndarray
    movie_factors: np.ndarray
    seen_indices: np.ndarray
    seen_indptr: np.ndarray
    item_counts: np.ndarray
    item_means: np.ndarray
    global_mean: float
    metadata: dict

    @classmethod
    def fit(cls, ratings: Ratings, titles: dict, rank=20, seed=42):
        ratings.validate()
        if type(rank) is not int or not 1 <= rank <= 128:
            raise ValueError("rank must be between 1 and 128")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an unsigned 32-bit integer")
        users, u = np.unique(ratings.users, return_inverse=True)
        movies, m = np.unique(ratings.movies, return_inverse=True)
        if min(len(users), len(movies)) < 2:
            raise ValueError("SVD requires at least two training users and two training movies")
        if any(int(movie) not in titles for movie in movies):
            raise ValueError("Training movie is missing from the catalog")
        actual_rank = min(rank, min(len(users), len(movies)) - 1)
        if (len(users) + len(movies)) * actual_rank * 8 > MAX_FACTOR_BYTES:
            raise ValueError("Dense factor memory would exceed 256 MiB; reduce rank or use a smaller ETL run")
        order = np.lexsort((m, u))
        u, m, values = u[order], m[order], ratings.values[order]
        matrix = csr_matrix((values, (u, m)), shape=(len(users), len(movies)))
        matrix.sort_indices()
        mean = float(values.mean())
        residuals = matrix.copy()
        residuals.data -= mean
        if np.all(residuals.data == 0):
            # A constant-rating fixture has no residual signal to factorize.
            user_factors = np.zeros((len(users), actual_rank))
            movie_factors = np.zeros((len(movies), actual_rank))
        else:
            estimator = TruncatedSVD(n_components=actual_rank, n_iter=7, random_state=seed)
            user_factors = estimator.fit_transform(residuals)
            movie_factors = estimator.components_.T
        counts = np.bincount(m, minlength=len(movies))
        means = np.bincount(m, weights=values, minlength=len(movies)) / counts
        metadata = {"engine": "mean_centered_svd", "requested_rank": rank,
                    "effective_rank": actual_rank, "seed": seed,
                    "training_rows": len(values), "training_users": len(users),
                    "training_movies": len(movies)}
        return cls(users, movies, np.array([titles[int(i)] for i in movies]),
                   user_factors, movie_factors, matrix.indices, matrix.indptr,
                   counts, means, mean, metadata)

    def _user_index(self, user_id):
        index = int(np.searchsorted(self.users, user_id))
        return index if index < len(self.users) and self.users[index] == user_id else None

    def seen(self, user_id):
        index = self._user_index(user_id)
        if index is None:
            return np.array([], dtype=np.int64)
        return self.seen_indices[self.seen_indptr[index]:self.seen_indptr[index + 1]]

    def recommend(self, user_id: int, k=10, mode="svd") -> dict:
        validate_k(k)
        if type(user_id) is not int or not 0 < user_id <= 2**31 - 1:
            raise ValueError("user-id must be a positive signed 32-bit integer")
        if mode not in ("svd", "popularity"):
            raise ValueError("Unknown recommendation mode")
        index = self._user_index(user_id)
        scores = (self.user_factors[index] @ self.movie_factors.T + self.global_mean
                  if index is not None and mode == "svd" else self.item_means)
        if index is None or mode == "popularity":
            order = np.lexsort((self.movies, -self.item_means, -self.item_counts))
            strategy = "popularity_fallback" if index is None else "popularity"
        else:
            order = np.lexsort((self.movies, -scores))
            strategy = "svd"
        seen = set(self.seen(user_id).tolist())
        chosen = [int(i) for i in order if int(i) not in seen][:k]
        return {"user_id": user_id, "strategy": strategy, "requested_k": k,
                "recommendations": [{"movie_id": int(self.movies[i]),
                                     "title": str(self.titles[i]), "score": float(scores[i])}
                                    for i in chosen]}

    def predict(self, user_id, movie_id, mode="svd") -> tuple[float, bool]:
        user = self._user_index(user_id)
        movie = int(np.searchsorted(self.movies, movie_id))
        known_movie = movie < len(self.movies) and self.movies[movie] == movie_id
        warm = user is not None and known_movie
        if warm and mode == "svd":
            value = self.global_mean + self.user_factors[user] @ self.movie_factors[movie]
        elif known_movie:
            value = self.item_means[movie]
        else:
            value = self.global_mean
        return float(np.clip(value, 0.5, 5.0)), bool(warm)

    def save(self, output: Path, source: dict) -> dict:
        if output.exists():
            raise ValueError(f"Model output already exists: {output}")
        output.mkdir(parents=True)
        marker = output / "INCOMPLETE"
        marker.write_text("Model has not completed.\n", encoding="utf-8")
        arrays = {key: value for key, value in vars(self).items() if isinstance(value, np.ndarray)}
        np.savez_compressed(output / "model.npz", **arrays)
        report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                  "parameters": self.metadata, "global_mean": self.global_mean,
                  "source": source, "model_sha256": sha256(output / "model.npz"),
                  "versions": {"package": __version__, "python": platform.python_version(),
                               "numpy": np.__version__, "sklearn": sklearn.__version__}}
        temporary = output / "model.json.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(output / "model.json")
        marker.unlink()
        return report

    @classmethod
    def load(cls, directory: Path):
        if (directory / "INCOMPLETE").exists() or not (directory / "model.json").is_file():
            raise ValueError("Require a completed model directory containing model.json")
        report = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        if report.get("schema_version") != 1 or report.get("parameters", {}).get("engine") != "mean_centered_svd":
            raise ValueError("Unsupported model format")
        if sha256(directory / "model.npz") != report["model_sha256"]:
            raise ValueError("Model checksum mismatch")
        with np.load(directory / "model.npz", allow_pickle=False) as arrays:
            model = cls(**{key: arrays[key] for key in arrays.files},
                        global_mean=report["global_mean"], metadata=report["parameters"])
        return model, report


def train(data: Path, output: Path, rank=20, seed=42) -> dict:
    if output.exists():
        raise ValueError(f"Model output already exists: {output}")
    if output.resolve().is_relative_to(data.resolve()):
        raise ValueError("Model output must be outside the ETL input directory")
    before = source_signature(data)
    model = RecommendationModel.fit(read_split(data, "train"), read_titles(data), rank, seed)
    if source_signature(data) != before:
        raise ValueError("Training inputs changed during fitting")
    return model.save(output, before)


def evaluate(data: Path, directory: Path, split="validation", k=10, threshold=4.0) -> dict:
    validate_k(k)
    if split not in ("validation", "test"):
        raise ValueError("Evaluation split must be validation or test")
    if not np.isfinite(threshold) or not 0.5 <= threshold <= 5:
        raise ValueError("Relevance threshold must be finite and between 0.5 and 5")
    model, metadata = RecommendationModel.load(directory)
    if source_signature(data) != metadata["source"]:
        raise ValueError("ETL training source differs from this model's source")
    before = source_signature(data, split)
    holdout = read_split(data, split)
    if not len(holdout.values):
        raise ValueError(f"{split} split is empty")
    relevant = {}
    seen_pairs = {(int(user), int(model.movies[item]))
                  for user in model.users for item in model.seen(int(user))}
    for user, movie, rating in zip(holdout.users, holdout.movies, holdout.values):
        user, movie = int(user), int(movie)
        if (user, movie) in seen_pairs:
            raise ValueError("Holdout overlaps training user/movie pairs")
        if rating >= threshold:
            relevant.setdefault(user, set()).add(movie)
    if len(relevant) > 2000 or len(relevant) * len(model.movies) > 20_000_000:
        raise ValueError("Local ranking evaluation exceeds 2,000 users or 20M candidate scores; use a smaller ETL run")
    metrics = {}
    for mode in ("svd", "popularity"):
        rankings = {user: [item["movie_id"] for item in model.recommend(user, k, mode)["recommendations"]]
                    for user in relevant}
        predictions = [model.predict(int(u), int(m), mode) for u, m in zip(holdout.users, holdout.movies)]
        errors = np.array([prediction for prediction, _ in predictions]) - holdout.values
        warm = np.array([covered for _, covered in predictions], dtype=bool)
        metrics[mode] = {**ranking_metrics(rankings, relevant, k),
                         "rmse_with_fallback": float(np.sqrt(np.mean(errors**2))),
                         "warm_rmse": float(np.sqrt(np.mean(errors[warm]**2))) if warm.any() else None,
                         "warm_prediction_coverage": float(warm.mean())}
    if source_signature(data, split) != before:
        raise ValueError("Evaluation inputs changed during scoring")
    return {"schema_version": 1, "split": split, "k": k, "relevance_threshold": threshold,
            "holdout_rows": len(holdout.values), "model_sha256": metadata["model_sha256"],
            "holdout_source": before, "metrics": metrics,
            "protocol": {"candidates": "all training movies, excluding each user's training history",
                         "ranking_users": "holdout users with at least one relevant rating, including cold users",
                         "precision_denominator": "fixed k even when fewer items are returned",
                         "recall_denominator": "all relevant holdout movies, including unseen movies",
                         "fallback": "known movie mean; otherwise training global mean"}}
