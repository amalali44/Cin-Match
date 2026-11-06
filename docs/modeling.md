# CinMatch recommendation baseline

This milestone connects ETL output to model training, inference, and evaluation.
The model is a small-data scikit-learn baseline. Distributed Spark ALS, model
tuning, Docker, and a full MovieLens 25M experiment remain future work.

## Run locally

Use Python 3.11+ from a directory containing `pyproject.toml`:

```sh
python -m pip install -e ".[ml]"
python -m recsys demo --output artifacts/model-demo
python -m recsys recommend --model artifacts/model-demo/model --user-id 1 --k 3
```

This path requires neither Java nor Spark. It uses only bundled synthetic data.
The sample has six training, three validation, and three test interactions.
The generated demo data implements the input contract consumed by the model;
it does not include the full SQL feature tables or replace Spark ETL validation.

`demo` writes `data/`, `model/`, `recommendations.json`, and `evaluation.json`.
Existing output directories are refused. Run it under a new directory to repeat.

## Train from a completed ETL run

```sh
recsys train --data artifacts/sample-etl --output artifacts/svd-model --rank 20 --seed 42
recsys recommend --model artifacts/svd-model --user-id 1 --k 10
recsys evaluate --data artifacts/sample-etl --model artifacts/svd-model --split validation --k 10 --threshold 4.0 --output artifacts/validation-metrics.json
```

Training reads `manifest.json`, the `train` partition under `interactions/`, and
the `movies/` catalog. Parquet is scanned in batches using PyArrow with split
filtering; it does not load validation or test ratings into the model. Source
files and the ETL manifest are hashed before and after fitting. A completed ETL
run must have accurate split counts and no `INCOMPLETE` marker.

Output model directories must be new and outside the ETL input directory.
Recommendation and evaluation commands emit JSON to stdout. Evaluation can also
save a new JSON report file outside the model/data directories.

## What the model learns

The training ratings form a sparse user-by-movie matrix. CinMatch subtracts the
training global mean from observed ratings and applies scikit-learn
`TruncatedSVD` to learn user and movie factors. Missing residuals are zero, which
corresponds to global-mean imputation. This simple baseline does not optimize an
observed-only ALS objective and does not imply the performance of Spark ALS.

Requested rank is reduced to `min(number_of_users, number_of_movies) - 1` when
necessary. At least two training users and two training movies are required.
The default rank is 20; the default random seed is 42. Constant-rating data uses
zero residual factors without attempting a degenerate decomposition.

Known-user recommendations rank training movies by reconstructed score and
exclude all movies in that user's training history. Equal scores break ties by
movie ID. Only movies with training interactions are candidates; catalog-only
movies are not silently treated as learned candidates. Recommendations include
movie titles, scores, and the strategy used. Scores are unbounded ranking
values, not calibrated probabilities; rating predictions used for RMSE are
clipped to the MovieLens 0.5–5 range.

New users receive a training-popularity fallback, ordered by interaction count,
then mean rating, then movie ID. Its displayed score is the item mean, even
though popularity ordering uses count first. Known users also have seen movies
excluded when evaluating the popularity comparator.

## Model artifacts and reproducibility

- `model.npz`: numeric factors, ID mappings, titles, seen-item indices, and item
  statistics. Loading uses `allow_pickle=False`; no pickled estimator is used.
- `model.json`: algorithm parameters, effective rank, seed, training counts,
  training global mean, package/runtime versions, source signatures, and the
  NPZ checksum. It is the completion record alongside the absence of `INCOMPLETE`.

Reloading uses saved factors and training history, so inference does not require
the original ratings. Source row order is normalized before fitting. A fixed
seed supports repeatability in the same numerical environment; floating-point
results may differ across BLAS libraries, hardware, or dependency versions.
This milestone pins scikit-learn and PyArrow, while recording the NumPy version;
full environment locking/containerization remains later work.

## Evaluation protocol

Validation is the default split. Use it for comparing settings; evaluate the
selected model on `--split test` only after selection. There is no automatic
hyperparameter search or test-driven tuning in this milestone.

The report compares **SVD** with **training popularity** using the same candidates:

| Metric | Definition |
| --- | --- |
| Precision@K | Per-user hits divided by fixed K, then macro-averaged |
| Recall@K | Per-user hits divided by all relevant held-out movies, then macro-averaged |
| RMSE with fallback | Clipped prediction error over every held-out interaction |
| Warm RMSE | Error only where both user and movie appeared in training |
| Warm prediction coverage | Fraction of held-out interactions with a trained user and movie |

A relevant movie has a held-out rating at least `--threshold` (default 4.0).
Ranking includes users with at least one relevant held-out movie, including
new users. Missing/cold movies remain in recall denominators even though the
training-only candidate universe cannot retrieve them. Short recommendation
lists retain the fixed K precision denominator. When no user has a relevant
rating, ranking metrics are `null` and `evaluated_users` is zero.

For rating predictions, new users with a known movie receive its training mean;
unknown movies receive the training global mean. Coverage is reported separately
so successful fallback predictions do not hide cold-start limitations. Empty
holdouts and user/movie overlap with training are rejected. Evaluation requires
the exact training source signature used for that model, and records/checks the
holdout file hashes for the evaluation run.

## Deliberate local limits

Each training or evaluation split is limited to **200,000 interactions**. Both
declared and actual row counts are checked. Dense factor arrays are limited to
256 MiB. Rank must be 1–128; K must be 1–100. Catalogs are limited to 200,000 rows.
Ranking evaluation is limited to 2,000 relevant users and 20 million candidate
scores. Limits fail explicitly; there is no silent sampling or row truncation.
These are guardrails, not guarantees about total process memory.

For a larger dataset, create an explicit smaller ETL input/run for this baseline.
The full 25M training path will use distributed ALS rather than raising these
limits and collecting all interactions into local memory.

## Verification

```sh
python -m unittest discover -s tests -v
```

ML tests run when the ML dependencies are installed; base installations skip
them. Spark tests remain opt-in. CI runs the ML suite on Python 3.11 and 3.13,
and exercises Spark ETL output through model training in a separate job. Tests
cover save/reload equivalence, seen-item exclusion, cold starts, training-only
fitting, source integrity, hand-calculated metrics, and CLI flows. Synthetic
metrics validate plumbing; they are not evidence of recommendation quality.
