"""Bounded, Spark-free readers for completed CinMatch ETL runs."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds

MAX_ROWS = 200_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(root: Path) -> dict:
    if (root / "INCOMPLETE").exists() or not (root / "manifest.json").is_file():
        raise ValueError("Require a completed ETL run with manifest.json and no INCOMPLETE marker")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Unsupported ETL manifest schema")
    summary = manifest.get("counts")
    counts = summary.get("splits") if isinstance(summary, dict) else None
    if not isinstance(counts, dict):
        raise ValueError("ETL manifest must declare split counts")
    if any(type(counts.get(name)) is not int or counts[name] < 0
           for name in ("train", "validation", "test")):
        raise ValueError("ETL manifest must declare nonnegative train/validation/test counts")
    return manifest


def _dataset(root: Path, name: str):
    return ds.dataset(str((root / name).resolve()), format="parquet", partitioning="hive")


@dataclass(frozen=True)
class Ratings:
    users: np.ndarray
    movies: np.ndarray
    values: np.ndarray

    def validate(self):
        if not (self.users.ndim == self.movies.ndim == self.values.ndim == 1 and
                len(self.users) == len(self.movies) == len(self.values)):
            raise ValueError("Ratings must contain equally sized one-dimensional arrays")
        if len(self.users) > MAX_ROWS:
            raise ValueError(f"Local baseline supports at most {MAX_ROWS:,} rows per split; use a smaller ETL run")
        for ids in (self.users, self.movies):
            if ids.dtype.kind not in "iu" or np.any(ids <= 0) or np.any(ids > 2**31 - 1):
                raise ValueError("User and movie IDs must be positive signed 32-bit integers")
        values = self.values
        if (not np.isfinite(values).all() or np.any(values < 0.5) or
                np.any(values > 5) or np.any(values * 2 != np.floor(values * 2))):
            raise ValueError("Ratings must be finite half-star values between 0.5 and 5")
        if len(set(zip(self.users.tolist(), self.movies.tolist()))) != len(values):
            raise ValueError("Duplicate user/movie pairs found; use deduplicated ETL output")


def read_split(root: Path, split: str) -> Ratings:
    manifest = read_manifest(root)
    if split not in ("train", "validation", "test"):
        raise ValueError("Unknown split")
    expected = manifest["counts"]["splits"][split]
    if expected > MAX_ROWS:
        raise ValueError(f"Local baseline supports at most {MAX_ROWS:,} rows per split; use a smaller ETL run")
    dataset = _dataset(root, "interactions")
    if not {"userId", "movieId", "rating", "split"}.issubset(dataset.schema.names):
        raise ValueError("Interaction Parquet must contain userId, movieId, rating, and split")
    for name in ("userId", "movieId"):
        if not pa.types.is_integer(dataset.schema.field(name).type):
            raise ValueError(f"{name} must have an integer Parquet type")
    chunks = []
    count = 0
    for batch in dataset.to_batches(columns=["userId", "movieId", "rating"],
                                    filter=ds.field("split") == split,
                                    batch_size=8192, use_threads=False):
        count += batch.num_rows
        if count > MAX_ROWS:
            raise ValueError(f"Actual split exceeds the {MAX_ROWS:,}-row local baseline limit")
        if any(column.null_count for column in batch.columns):
            raise ValueError("Null user/movie IDs or ratings in ETL output")
        chunks.append(tuple(np.asarray(batch.column(name)) for name in ("userId", "movieId", "rating")))
    if count != expected:
        raise ValueError(f"{split} row count differs from manifest: expected {expected}, found {count}")
    arrays = [np.concatenate([chunk[i] for chunk in chunks]) if chunks else np.array([], dtype=dtype)
              for i, dtype in enumerate((np.int64, np.int64, np.float64))]
    order = np.lexsort((arrays[1], arrays[0]))
    result = Ratings(arrays[0][order], arrays[1][order], arrays[2][order].astype(np.float64))
    result.validate()
    return result


def read_titles(root: Path) -> dict[int, str]:
    catalog = _dataset(root, "movies")
    if not {"movieId", "title"}.issubset(catalog.schema.names):
        raise ValueError("Movie Parquet must contain movieId and title")
    if catalog.count_rows() > MAX_ROWS:
        raise ValueError("Movie catalog exceeds local baseline limit")
    rows = catalog.to_table(columns=["movieId", "title"]).to_pylist()
    if any(type(row["movieId"]) is not int or not 0 < row["movieId"] <= 2**31 - 1
           or not isinstance(row["title"], str) or not row["title"].strip() for row in rows):
        raise ValueError("Movie catalog contains invalid IDs or titles")
    titles = {row["movieId"]: row["title"] for row in rows}
    if len(titles) != len(rows):
        raise ValueError("Duplicate movie IDs in catalog")
    return titles


def source_signature(root: Path, split: str = "train") -> dict:
    manifest = read_manifest(root)
    if manifest["counts"]["splits"][split] > MAX_ROWS:
        raise ValueError(f"Local baseline supports at most {MAX_ROWS:,} rows per split; use a smaller ETL run")
    signature = {"manifest_sha256": sha256(root / "manifest.json"), "files": {}}
    for name, predicate in (("interactions", ds.field("split") == split), ("movies", None)):
        for fragment in sorted(_dataset(root, name).get_fragments(filter=predicate), key=lambda f: f.path):
            path = Path(fragment.path)
            signature["files"][path.relative_to(root.resolve()).as_posix()] = sha256(path)
    return signature
