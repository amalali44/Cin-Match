# Implementation roadmap

Land each milestone in a separate commit when its implementation and checks are
complete. Commit dates reflect when the work actually occurs.
Use a separate `codex/` feature branch for each milestone and review its diff
before committing and pushing. The ETL milestone uses `codex/pyspark-sql-etl`.

## 1. Project foundation — implemented

- Installable Python package, CLI, explicit data contract, streaming profile.
- Synthetic fixture, validation tests, CI, and dataset attribution.

Acceptance: installation, sample command, invalid-input checks, and tests pass.

## 2. PySpark + SQL ETL — implemented and reviewed

- Explicit schemas, invalid-row quarantine, deterministic deduplication.
- Metadata joins and SQL user/movie feature and genre/popularity aggregations.
- Partitioned Parquet and data-quality counts with dataset fingerprints.
- Global temporal train/validation/test boundaries; training-only features.

Acceptance: integration tests verify deduplication, split isolation, schema,
partition outputs, and feature statistics without holdout leakage.

## 3. Collaborative filtering

- Small-data scikit-learn baseline and distributed Spark ALS.
- Fixed seeds, validation tuning, saved models, parameters, and runtime versions.
- Exclude rated items and define cold-start behavior for new users/items.

Acceptance: train/save/reload/infer test and unseen-item checks pass.

## 4. Evaluation and analytics

- Precision@K and Recall@K with relevance threshold and fixed K.
- Document candidate universe, macro averaging, and empty-user behavior.
- RMSE plus prediction coverage to expose cold-start omissions.
- Record baseline comparisons, split counts, and experiment manifests.

Acceptance: hand-calculated metric fixtures and held-out experiment artifacts.

## 5. Docker training and inference

- Pin Python, Java, Spark, and ML dependencies in a container runtime.
- Mount data/artifacts and expose training, evaluation, and inference commands.
- Container smoke test in CI and full MovieLens execution/resource guidance.

Acceptance: a clean container trains, evaluates, reloads, and recommends.

## 6. Full-scale evidence

Run the actual MovieLens 25M data on recorded hardware and configuration.
Publish measured row counts, runtime, metrics, coverage, and reproducibility
instructions before describing 25M interactions as a completed result.
