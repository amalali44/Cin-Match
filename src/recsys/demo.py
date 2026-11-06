"""A Java-free synthetic demonstration; not a substitute for Spark ETL."""

import csv
import json
from importlib.resources import files
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from recsys.data import parse_rating
from recsys.model import RecommendationModel, evaluate, train


def create_sample_run(output: Path):
    """Write the ETL reader contract using only the bundled synthetic fixture."""
    if output.exists():
        raise ValueError(f"Demo data already exists: {output}")
    output.mkdir(parents=True)
    marker = output / "INCOMPLETE"
    marker.write_text("Synthetic data generation is incomplete.\n", encoding="utf-8")
    with files("recsys").joinpath("fixtures/ratings.csv").open("r", encoding="utf-8") as stream:
        ratings = [parse_rating(row, line) for line, row in enumerate(csv.DictReader(stream), 2)]
    splits = ["train" if row.timestamp <= 1000000060 else
              "validation" if row.timestamp <= 1000000120 else "test" for row in ratings]
    table = pa.table({"userId": pa.array([r.user_id for r in ratings], type=pa.int32()),
                      "movieId": pa.array([r.movie_id for r in ratings], type=pa.int32()),
                      "rating": [r.rating for r in ratings],
                      "timestamp": [r.timestamp for r in ratings], "split": splits})
    pq.write_to_dataset(table, root_path=str(output / "interactions"), partition_cols=["split"])
    with files("recsys").joinpath("fixtures/movies.csv").open("r", encoding="utf-8") as stream:
        catalog = list(csv.DictReader(stream))
    (output / "movies").mkdir()
    pq.write_table(pa.table({"movieId": pa.array([int(r["movieId"]) for r in catalog], type=pa.int32()),
                             "title": [r["title"] for r in catalog],
                             "genres": [r["genres"].split("|") for r in catalog]}),
                   output / "movies" / "sample.parquet")
    manifest = {"schema_version": 1, "producer": "synthetic_demo_not_spark_etl",
                "counts": {"input": len(ratings), "splits": {name: splits.count(name)
                           for name in ("train", "validation", "test")}},
                "config": {"train_end": 1000000060, "validation_end": 1000000120}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    marker.unlink()


def run_demo(output: Path) -> dict:
    if output.exists():
        raise ValueError(f"Demo output already exists: {output}")
    output.mkdir(parents=True)
    marker = output / "INCOMPLETE"
    marker.write_text("Demo is incomplete.\n", encoding="utf-8")
    data, directory = output / "data", output / "model"
    create_sample_run(data)
    trained = train(data, directory, rank=2, seed=42)
    model, _ = RecommendationModel.load(directory)
    recommendations = model.recommend(1, k=3)
    evaluation = evaluate(data, directory, k=3)
    for name, report in (("recommendations", recommendations), ("evaluation", evaluation)):
        (output / f"{name}.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    marker.unlink()
    return {"kind": "synthetic_demo", "note": "12 synthetic interactions; metrics are wiring checks, not benchmarks",
            "output": str(output.resolve()), "model": trained["parameters"],
            "recommendations": recommendations, "validation_metrics": evaluation["metrics"]}
