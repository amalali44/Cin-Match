"""CinMatch: personalized movie recommendation and analytics pipeline."""

import argparse
import csv
import json
import sys
from importlib.resources import files
from pathlib import Path

from recsys.data import profile_ratings


def _run_etl(args) -> int:
    from recsys.etl_config import ETLConfig

    config = ETLConfig(args.ratings, args.movies, args.output,
                       args.train_end, args.validation_end, args.buckets)
    config.validate()
    try:
        from pyspark.errors import PySparkException
        from py4j.protocol import Py4JJavaError
        from pyspark.sql import SparkSession
        from recsys.etl import run_etl
    except ImportError as exc:
        raise RuntimeError('ETL requires the Spark extra: python -m pip install ".[spark]"') from exc
    spark = None
    try:
        builder = SparkSession.builder.appName("CinMatchETL")
        if args.master:
            builder = builder.master(args.master)
        spark = builder.getOrCreate()
        report = run_etl(spark, config)
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    except (PySparkException, Py4JJavaError) as exc:
        raise RuntimeError(f"Spark ETL failed: {exc}") from exc
    finally:
        if spark is not None:
            spark.stop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="recsys", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    profile = commands.add_parser("profile", help="Validate and summarize ratings CSV")
    source = profile.add_mutually_exclusive_group(required=True)
    source.add_argument("--ratings", type=Path, help="Path to MovieLens ratings.csv")
    source.add_argument("--sample", action="store_true", help="Use synthetic bundled data")
    profile.add_argument("--output", type=Path, help="Also save the JSON report")
    etl = commands.add_parser("etl", help="Build split Parquet features with PySpark and SQL")
    etl.add_argument("--ratings", type=Path, required=True)
    etl.add_argument("--movies", type=Path, required=True)
    etl.add_argument("--output", type=Path, required=True, help="New run directory; never overwritten")
    etl.add_argument("--train-end", type=int, required=True, help="Inclusive Unix timestamp")
    etl.add_argument("--validation-end", type=int, required=True, help="Inclusive Unix timestamp")
    etl.add_argument("--buckets", type=int, default=16, help="User hash partitions per split (1-1024)")
    etl.add_argument("--master", help="Spark master, e.g. local[2]; otherwise use Spark configuration")
    args = parser.parse_args(argv)
    try:
        if args.command == "etl":
            return _run_etl(args)
        if args.sample:
            resource = files("recsys").joinpath("fixtures/ratings.csv")
            with resource.open("r", encoding="utf-8", newline="") as stream:
                report = profile_ratings(stream)
            report["source"] = "bundled-synthetic-sample"
        else:
            if args.output and args.output.resolve() == args.ratings.resolve():
                raise ValueError("Output must not overwrite the ratings input")
            with args.ratings.open("r", encoding="utf-8-sig", newline="") as stream:
                report = profile_ratings(stream)
            report["source"] = str(args.ratings)
        payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload, encoding="utf-8")
        print(payload, end="")
        return 0
    except (OSError, UnicodeError, ValueError, csv.Error, RuntimeError) as exc:
        print(f"recsys: {exc}", file=sys.stderr)
        return 2
