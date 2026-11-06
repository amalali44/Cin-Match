# Review notes: recommendation baseline

Prepared November 5, 2026. Branch: `feature/collaborative-filtering`, based on the
merged ETL milestone in `origin/main`. Reviewed and approved before publication.

Proposed commit message:

```text
Add SVD recommendations and held-out evaluation
```

## What this changes

CinMatch previously prepared ratings for machine learning. It can now train a
small collaborative-filtering model from those outputs, recommend unseen movies,
save/reload its factors, and evaluate it against a popularity baseline.

- `train`: read training-only Parquet from a completed ETL run and fit a
  mean-centered scikit-learn TruncatedSVD model with a fixed seed.
- `recommend`: load saved factors and return movie IDs, titles, scores, and the
  strategy used. Rated movies are excluded; new users get training popularity.
- `evaluate`: compare SVD with popularity on validation or test data using
  Precision@K, Recall@K, RMSE with fallback, warm RMSE, and coverage.
- `demo`: generate a clearly labeled synthetic sample and exercise the complete
  model workflow without requiring Java, Spark, or a MovieLens download.

Model artifacts use NumPy arrays and JSON metadata rather than pickled estimators.
Metadata records parameters, versions, training counts, and data/model checksums.
Existing outputs are refused, incomplete runs are rejected, and changed training
inputs prevent evaluation with an old model.

## Validation completed

- All 45 tests passed on Python 3.13, including the two Spark integration tests.
- The Spark tests used Java 17 and the existing Windows Java filesystem adapter.
- The base installation also passed its 23 applicable tests; the 22 optional
  ML/Spark tests were correctly skipped without their runtime dependencies.
- Editable installation, the Java-free demo, and installed `train`, `recommend`,
  and `evaluate` commands succeeded.
- Training and evaluation were also exercised against the actual Spark-generated
  synthetic ETL outputs, including their Hive partition directories.
- CI now includes model checks on Python 3.11 and 3.13 and a Spark ETL-to-model
  smoke flow. These updated CI jobs have not run remotely because this branch
  has not been pushed.

Tests cover hand-calculated metrics, training-only fitting, seen-item exclusion,
save/reload consistency, cold starts, checksums, split overlap, and input limits.

## Review these decisions

1. This is a mean-imputed SVD baseline. Distributed ALS remains a separate next
   step; this model does not optimize the same objective as Spark ALS.
2. Candidates are movies seen in training. Relevant unseen holdout movies stay
   in recall denominators, making cold-item limitations visible.
3. Precision always divides by K, even when fewer candidates remain. Ranking
   metrics include users with at least one relevant holdout rating and are null
   when there are no such users. Validation is the default evaluation split.
4. New-user rating predictions use an item's training mean; unknown movies use
   the training global mean. Coverage separates these fallbacks from warm cases.
5. The local model has explicit limits: 200,000 rows per split, 256 MiB for dense
   factors, rank at most 128, and bounded ranking evaluation. No silent sampling.
6. All executed examples use synthetic data. Real-data quality, full MovieLens
   25M execution, distributed training, tuning, and Docker remain unverified or
   unimplemented as documented in the roadmap.

## Try it

From this branch's directory containing `pyproject.toml`, with Python 3.11+:

```sh
python -m pip install -e ".[ml]"
python -m recsys demo --output artifacts/review-demo
python -m recsys recommend --model artifacts/review-demo/model --user-id 1 --k 3
python -m unittest discover -s tests -v
```

The demo returns two unseen training movies for user 1 and writes inspectable
JSON reports. Its 12 interactions produce six training, three validation, and
three test rows. Metrics are wiring checks rather than performance benchmarks.
Choose another output directory when repeating the demo.

Start with `docs/modeling.md` for the model and metric semantics, `src/recsys/model.py`
for training/inference, and `src/recsys/ml_data.py` for the input contract and
provenance checks.
