"""ETL configuration and input provenance without a Spark dependency."""

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

from recsys.data import MAX_TIMESTAMP


@dataclass(frozen=True)
class ETLConfig:
    ratings: Path
    movies: Path
    output: Path
    train_end: int
    validation_end: int
    buckets: int = 16

    def validate(self) -> None:
        if not 0 <= self.train_end < self.validation_end <= MAX_TIMESTAMP:
            raise ValueError("Require 0 <= train-end < validation-end <= signed 64-bit maximum")
        if not 1 <= self.buckets <= 1024:
            raise ValueError("buckets must be between 1 and 1024")
        for source in (self.ratings, self.movies):
            if not source.is_file():
                raise ValueError(f"Input must be an existing local CSV file: {source}")
        if self.ratings.resolve() == self.movies.resolve():
            raise ValueError("ratings and movies must be different files")
        if self.output.exists():
            raise ValueError(f"Output already exists; choose a new run directory: {self.output}")

    def manifest_dict(self) -> dict:
        return {key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(self).items()}


def fingerprint(path: Path) -> dict:
    """Hash bytes in bounded chunks and reject a file changed during hashing."""
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Input changed during fingerprinting: {path}")
    return {"path": str(path.resolve()), "bytes": after.st_size,
            "mtime_ns": after.st_mtime_ns, "sha256": digest.hexdigest()}
