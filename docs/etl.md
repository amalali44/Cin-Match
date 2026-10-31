# CinMatch ETL

## Runtime and execution

Python 3.11+, Java 17+, and `python -m pip install ".[spark]"` are required.
The Spark extra pins PySpark 4.0.1. The lightweight `profile` command still
requires no Spark or Java installation.

```sh
recsys etl --ratings data/raw/ml-25m/ratings.csv --movies data/raw/ml-25m/movies.csv --output artifacts/ml25m-run-001 --train-end 1451606399 --validation-end 1514764799 --buckets 16 --master "local[4]"
```

The example boundaries mean training through 2015-12-31 23:59:59 UTC,
validation through 2017-12-31 23:59:59 UTC, and test after that. They are example
experiment choices, not tuned benchmark settings. Adjust them for your study.
Validation or test splits may be empty; the manifest reports exact counts.
Training must contain at least one interaction.

Inputs must be local files, or files on a shared filesystem mounted at identical
paths on every executor and the driver. This CLI does not currently accept
object-store URIs. Omit `--master` to use the Spark deployment's configured master.
For 25M data, allocate driver/executor memory and shuffle partitions through
Spark configuration for the available hardware; the two-core synthetic fixture
is a correctness check and does not demonstrate full-scale performance.

The driver streams each input through SHA-256 before and after execution.
Spark reads and transforms the rating data without collecting it into Python;
only the three split counts return to the driver. Cached frames use
`MEMORY_AND_DISK`. Hashing adds two sequential input reads and detects source
changes before the run is published. Keep source files immutable during a run.

## Data contract and cleaning

- Ratings header: `userId,movieId,rating,timestamp`, in that order. Movie metadata
  header: `movieId,title,genres`. Spark enforces these headers and column counts.
- IDs must be positive signed 32-bit integers. Ratings must be finite, within
  0.5–5.0, in half-star increments. Timestamps must be nonnegative signed 64-bit
  Unix seconds. Parsing uses explicit string schemas and safe numeric casts.
- Malformed rating rows and invalid values are retained in `quarantine/` with
  a JSON payload and a rejection reason. Ratings without movie metadata are
  quarantined as `missing_movie_metadata` before deduplication.
- Invalid or duplicate movie metadata IDs fail the run instead of silently
  multiplying joined interactions. Movie titles may contain quoted commas;
  genres are pipe-separated and become an array.
- Each user/movie pair retains the latest timestamp. At equal timestamps the
  higher rating wins. Exact duplicates collapse. This occurs before splitting,
  so a pair never appears in multiple splits. A later re-rating can remove an
  earlier training interaction; this is an explicit snapshot policy.

## Split and feature semantics

Training contains `timestamp <= train_end`. Validation contains
`train_end < timestamp <= validation_end`; test contains later interactions.
Boundaries are global and fixed, not randomly sampled or per-user quantiles.

SQL aggregates are computed exclusively from the training split:

| Dataset | Columns |
| --- | --- |
| `user_features/` | userId, user_rating_count, user_mean_rating, user_last_timestamp |
| `movie_features/` | movieId, movie_rating_count, movie_mean_rating |
| `genre_analytics/` | genre, rating_count, user_count, mean_rating |
| `movies/` | movieId, title, genres |
| `quarantine/` | raw_payload, rejection_reason |

`interactions/` includes typed IDs, rating, timestamp, title, genres, the user and
movie aggregates, and `user_cold_start` / `movie_cold_start` flags. Missing
training aggregates remain null. Consumers must choose a model fallback; ETL
does not fill missing means using held-out ratings. Catalog metadata can include
movies absent from training, but their rating-derived features stay null.

The interaction dataset is partitioned by `split` and a deterministic
`user_bucket` hash (default 16 buckets). Read the **root** interaction directory
to recover these partition columns. User/movie aggregates are history summaries
for analytics and future collaborative filtering; using rating means as
supervised training targets/features would require out-of-fold or per-event
historical encoding to avoid self-contribution.

Genre statistics count each rating once per listed genre, so multi-genre movies
contribute to more than one genre. `(no genres listed)` is excluded.

## Run completeness and provenance

Existing output directories are refused. An incomplete run leaves `INCOMPLETE`
and potentially partial Parquet files, without `manifest.json`. Choose a fresh
directory after fixing the cause; no automatic deletion or overwrite occurs.
Consumers must require `manifest.json` before reading a run.

The manifest records input paths, byte sizes, SHA-256 hashes, modification times,
split boundaries, partition settings, package/Python/Spark versions, cleaning
policies, and quality counts. The accounting identity is:

```text
input = invalid + missing_movie_metadata + duplicates_removed + accepted
accepted = train + validation + test
```

`manifest.json` is published only after every dataset is written and the input
hashes are rechecked. Separate runs have different timestamps and physical
Parquet file layouts; the cleaning, split, and feature semantics are deterministic.

## Integration tests

```sh
# macOS/Linux
RECSYS_RUN_SPARK_TESTS=1 python -m unittest discover -s tests -v
```

```powershell
# Windows PowerShell
$env:RECSYS_RUN_SPARK_TESTS = "1"
python -m unittest discover -s tests -v
```

The Spark tests exercise rejected rows, unknown movie IDs, deduplication,
boundary timestamps, train-only statistics, cold-start flags, partitioned
Parquet round-trips, provenance, and output collision protection. CI runs these
on Ubuntu with Java 17. Native Windows Hadoop filesystem support may require
additional Hadoop native libraries; Linux is the supported CI runtime.

Local verification also passed all 19 tests on Windows with Python 3.13, Java 17,
and the [GlobalMentor Java filesystem adapter](https://github.com/globalmentor/hadoop-bare-naked-local-fs)
version 0.1.0 on Spark's driver classpath. That adapter was used only for the
local test runtime and is not a production dependency. The Ubuntu CI job uses
Spark's standard Hadoop filesystem; its result will be available after pushing.
