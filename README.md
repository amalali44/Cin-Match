# CinMatch

Personalized movie recommendations and analytics, built on MovieLens.
The project uses Python, SQL, and PySpark, with scikit-learn models and Docker
training/inference planned in later milestones.

**Current stage: PySpark + SQL ETL.** The package validates and profiles ratings,
quarantines invalid interactions, builds chronological splits and training-only
features, and writes Parquet datasets with a provenance manifest. Models,
recommendation evaluation, and Docker remain planned work. No full MovieLens
25M run or recommendation performance is claimed yet.

## Quick start

Requires Python 3.11+. Runtime code uses only the standard library.

```sh
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e .
recsys profile --sample --output artifacts/sample-profile.json
python -m unittest discover -s tests -v
```

You can also use `python -m recsys profile --sample` after installation.
The bundled fixture contains 12 **synthetic** interactions, for smoke testing.

## Profile real data

Download and extract MovieLens separately, then supply its ratings file:

```sh
recsys profile --ratings data/raw/ml-25m/ratings.csv --output artifacts/ml25m-profile.json
```

The command validates `userId`, `movieId`, `rating`, and `timestamp`, then emits
JSON containing interaction count, rating statistics, and timestamp range.
Processing uses constant working memory, one CSV row at a time. Invalid rows
produce an error with a line number and exit code 2. Extra named columns are
accepted. Deduplication and metadata joins belong to the upcoming ETL stage.

MovieLens 25M contains 25 million ratings. See the official
[dataset page](https://grouplens.org/datasets/movielens/25m/) and the README and
usage terms included in its download. Data and generated artifacts stay out of Git.

## Structure

```text
src/recsys/       CLI, data contracts, Spark ETL, SQL, and synthetic fixtures
tests/           Unit tests and opt-in Spark integration tests
docs/roadmap.md   Implementation stages and acceptance checks
data/raw/        Local downloads (ignored)
artifacts/       Generated profiles and future models (ignored)
```

## Build ML-ready features

Install Java 17+ and set `JAVA_HOME`, then install the pinned Spark extra:

```sh
python -m pip install ".[spark]"
recsys etl --ratings src/recsys/fixtures/ratings.csv --movies src/recsys/fixtures/movies.csv --output artifacts/sample-etl --train-end 1000000060 --validation-end 1000000120 --buckets 2 --master "local[2]"
```

The sample command is runnable from the repository root. Each run requires a
**new output directory**. For real data, use `ratings.csv` and `movies.csv` from
the same MovieLens release and choose explicit Unix timestamp boundaries.
See [ETL usage and data semantics](docs/etl.md) for output schemas, deduplication,
cold-start behavior, resource guidance, and integration test commands.

## Pipeline roadmap

```text
MovieLens CSV → PySpark + SQL ETL → partitioned Parquet
             → temporal splits → collaborative filtering
             → Precision@K, Recall@K, RMSE → model artifacts + inference
```

Spark ALS will provide distributed collaborative filtering; scikit-learn will
provide a small-data baseline. See the [roadmap](docs/roadmap.md) for staged
implementation and evidence required before reporting full-scale results.
