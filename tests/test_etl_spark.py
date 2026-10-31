"""Opt-in integration checks: RECSYS_RUN_SPARK_TESTS=1, Java 17+, .[spark]."""

import json
import os
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(os.environ.get("RECSYS_RUN_SPARK_TESTS") == "1", "Spark integration tests are opt-in")
class SparkETLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pyspark.sql import SparkSession
        cls.spark = (SparkSession.builder.master("local[2]").appName("ETLIntegrationTests")
                     .config("spark.ui.enabled", "false")
                     .config("spark.sql.shuffle.partitions", "2")
                     .getOrCreate())
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_etl_artifacts_quality_temporal_splits_and_no_holdout_leakage(self):
        from recsys.etl import run_etl
        from recsys.etl_config import ETLConfig, fingerprint

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ratings, movies = root / "ratings.csv", root / "movies.csv"
            ratings.write_text(
                "userId,movieId,rating,timestamp\n"
                "1,1,4,100\n1,1,4.5,110\n1,1,3,110\n1,2,2,200\n2,1,5,300\n"
                "3,3,3.5,120\n4,2,4,130\n9,4,1,350\n1,999,4,100\n"
                "0,1,4,100\n2,2,NaN,100\n2,2,3.7,100\n2,2,4,-1\n"
                "2,2,4,100,extra\n2,2,4\n", encoding="utf-8")
            movies.write_text('movieId,title,genres\n1,"First, Movie",Drama|Comedy\n'
                              '2,Second,Comedy\n3,Third,Drama\n4,Fourth,Thriller\n', encoding="utf-8")
            config = ETLConfig(ratings, movies, root / "run", 130, 200, 2)
            report = run_etl(self.spark, config)
            self.assertEqual(report["counts"], {
                "input": 15, "invalid": 6, "missing_movie_metadata": 1,
                "duplicates_removed": 2, "accepted": 6,
                "splits": {"train": 3, "validation": 1, "test": 2}})
            self.assertEqual(report["inputs"]["ratings"], fingerprint(ratings))
            self.assertFalse((config.output / "INCOMPLETE").exists())
            self.assertEqual(json.loads((config.output / "manifest.json").read_text()), report)

            interactions = self.spark.read.parquet(str(config.output / "interactions"))
            self.assertEqual(dict(interactions.dtypes)["userId"], "int")
            rows = {(r.userId, r.movieId): r.asDict() for r in interactions.collect()}
            self.assertEqual(rows[(1, 1)]["rating"], 4.5)  # Latest duplicate wins.
            self.assertEqual(rows[(1, 2)]["split"], "validation")
            self.assertEqual(rows[(2, 1)]["split"], "test")
            self.assertEqual(rows[(1, 2)]["user_mean_rating"], 4.5)  # Excludes holdout rating 2.
            self.assertEqual(rows[(2, 1)]["movie_mean_rating"], 4.5)  # Excludes holdout rating 5.
            self.assertTrue(rows[(2, 1)]["user_cold_start"])
            self.assertIsNone(rows[(2, 1)]["user_mean_rating"])
            self.assertTrue(rows[(9, 4)]["movie_cold_start"])
            self.assertIsNone(rows[(9, 4)]["movie_mean_rating"])
            self.assertEqual(rows[(1, 1)]["title"], "First, Movie")
            self.assertEqual(set(rows[(1, 1)]["genres"]), {"Drama", "Comedy"})
            self.assertTrue(all(0 <= r["user_bucket"] < 2 for r in rows.values()))
            self.assertTrue(list((config.output / "interactions").glob("split=train/user_bucket=*")))

            users = self.spark.read.parquet(str(config.output / "user_features"))
            self.assertEqual({r.userId for r in users.collect()}, {1, 3, 4})
            genres = self.spark.read.parquet(str(config.output / "genre_analytics"))
            self.assertEqual({r.genre: r.rating_count for r in genres.collect()}, {"Drama": 2, "Comedy": 2})
            quarantine = self.spark.read.parquet(str(config.output / "quarantine"))
            reasons = {r.rejection_reason: r["count"] for r in quarantine.groupBy("rejection_reason").count().collect()}
            self.assertEqual(reasons, {"invalid_user_id": 1, "invalid_rating": 2,
                                      "invalid_timestamp": 1, "malformed_csv": 2,
                                      "missing_movie_metadata": 1})
            with self.assertRaisesRegex(ValueError, "already exists"):
                run_etl(self.spark, config)

    def test_inclusive_split_boundaries(self):
        from recsys.etl import split_ratings

        frame = self.spark.sql("""
            SELECT * FROM VALUES (1, 1, 4.0, 100), (2, 2, 2.0, 200), (3, 3, 5.0, 201)
            AS ratings(userId, movieId, rating, timestamp)
        """)
        rows = {r.userId: r for r in split_ratings(frame, 100, 200).collect()}
        self.assertEqual(rows[1].split, "train")
        self.assertEqual(rows[1].rating, 4.0)
        self.assertEqual(rows[2].split, "validation")
        self.assertEqual(rows[3].split, "test")
