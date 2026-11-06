import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

ML_AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("sklearn", "pyarrow", "numpy"))
if ML_AVAILABLE:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from recsys.demo import create_sample_run
    from recsys.ml_data import MAX_ROWS, Ratings, read_split, read_titles
    from recsys.model import RecommendationModel, evaluate, train

from recsys.cli import main


@unittest.skipUnless(ML_AVAILABLE, "Install .[ml] to run model tests")
class ModelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data, self.output = self.root / "data", self.root / "model"
        create_sample_run(self.data)

    def fit(self):
        return RecommendationModel.fit(read_split(self.data, "train"), read_titles(self.data), rank=2, seed=42)

    def train(self):
        return train(self.data, self.output, rank=2, seed=42)

    def test_parquet_reader_preserves_split_and_catalog(self):
        self.assertEqual(len(read_split(self.data, "train").values), 6)
        self.assertEqual(len(read_split(self.data, "validation").values), 3)
        self.assertEqual(len(read_split(self.data, "test").values), 3)
        self.assertEqual(read_titles(self.data)[1], "Synthetic Movie One")

    def test_recommendations_exclude_training_history_and_include_titles(self):
        result = self.fit().recommend(1, k=10)
        self.assertEqual(result["strategy"], "svd")
        self.assertEqual({row["movie_id"] for row in result["recommendations"]}, {3, 4})
        self.assertTrue(all(row["title"] and np.isfinite(row["score"]) for row in result["recommendations"]))

    def test_new_users_get_deterministic_training_popularity(self):
        result = self.fit().recommend(999, k=3)
        self.assertEqual(result["strategy"], "popularity_fallback")
        self.assertEqual([row["movie_id"] for row in result["recommendations"]], [2, 1, 3])

    def test_save_reload_preserves_recommendations_and_predictions(self):
        original = self.fit()
        metadata = original.save(self.output, {"fixture": True})
        restored, saved = RecommendationModel.load(self.output)
        self.assertEqual(saved, metadata)
        self.assertEqual(restored.recommend(1, 3), original.recommend(1, 3))
        self.assertEqual(restored.predict(1, 3), original.predict(1, 3))
        self.assertFalse((self.output / "INCOMPLETE").exists())

    def test_repeatable_fit_is_independent_of_input_order(self):
        ratings = read_split(self.data, "train")
        reversed_rows = Ratings(ratings.users[::-1], ratings.movies[::-1], ratings.values[::-1])
        other = RecommendationModel.fit(reversed_rows, read_titles(self.data), rank=2, seed=42)
        np.testing.assert_allclose(self.fit().user_factors, other.user_factors)
        np.testing.assert_allclose(self.fit().movie_factors, other.movie_factors)

    def test_constant_ratings_have_finite_factors(self):
        ratings = Ratings(np.array([1, 2]), np.array([1, 2]), np.array([4.0, 4.0]))
        model = RecommendationModel.fit(ratings, {1: "One", 2: "Two"})
        self.assertEqual(model.recommend(1)["recommendations"][0]["movie_id"], 2)
        self.assertEqual(model.predict(1, 2), (4.0, True))

    def test_fallback_predictions_and_coverage(self):
        model = self.fit()
        self.assertEqual(model.predict(999, 2), (4.0, False))
        self.assertEqual(model.predict(1, 999), (model.global_mean, False))
        self.assertEqual(model.predict(999, 999), (model.global_mean, False))
        self.assertTrue(model.predict(1, 2)[1])

    def test_evaluation_compares_baselines_and_reports_cold_items(self):
        self.train()
        result = evaluate(self.data, self.output, k=3)
        self.assertEqual(result["split"], "validation")
        self.assertEqual(result["holdout_rows"], 3)
        self.assertEqual(set(result["metrics"]), {"svd", "popularity"})
        for metrics in result["metrics"].values():
            self.assertEqual(metrics["evaluated_users"], 3)
            self.assertAlmostEqual(metrics["warm_prediction_coverage"], 1 / 3)
            self.assertTrue(np.isfinite(metrics["rmse_with_fallback"]))
            self.assertLessEqual(metrics["recall_at_k"], 1 / 3)
        self.assertEqual(evaluate(self.data, self.output, split="test")["split"], "test")

    def test_holdout_changes_do_not_change_training(self):
        before = self.fit()
        path = next((self.data / "interactions" / "split=validation").glob("*.parquet"))
        table = pq.ParquetFile(path).read()
        table = table.set_column(table.schema.get_field_index("rating"), "rating",
                                 pa.array([0.5] * table.num_rows))
        pq.write_table(table, path)
        after = self.fit()
        np.testing.assert_array_equal(before.user_factors, after.user_factors)
        np.testing.assert_array_equal(before.movie_factors, after.movie_factors)
        self.assertEqual(before.global_mean, after.global_mean)

    def test_training_change_rejects_evaluation_with_old_model(self):
        self.train()
        path = next((self.data / "interactions" / "split=train").glob("*.parquet"))
        table = pq.ParquetFile(path).read()
        pq.write_table(table.slice(0, 1), path)
        with self.assertRaisesRegex(ValueError, "source differs"):
            evaluate(self.data, self.output)

    def test_holdout_training_pair_overlap_is_rejected(self):
        self.train()
        path = next((self.data / "interactions" / "split=validation").glob("*.parquet"))
        table = pq.ParquetFile(path).read()
        table = table.set_column(table.schema.get_field_index("userId"), "userId", pa.array([1, 2, 3], type=pa.int32()))
        table = table.set_column(table.schema.get_field_index("movieId"), "movieId", pa.array([1, 5, 5], type=pa.int32()))
        pq.write_table(table, path)
        with self.assertRaisesRegex(ValueError, "overlaps training"):
            evaluate(self.data, self.output)

    def test_model_checksum_and_incomplete_marker_are_checked(self):
        self.train()
        with (self.output / "model.npz").open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "checksum"):
            RecommendationModel.load(self.output)
        (self.output / "INCOMPLETE").touch()
        with self.assertRaisesRegex(ValueError, "completed model"):
            RecommendationModel.load(self.output)

    def test_existing_model_is_never_overwritten(self):
        self.train()
        original = (self.output / "model.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.train()
        self.assertEqual((self.output / "model.json").read_bytes(), original)

    def test_model_output_cannot_pollute_etl_inputs(self):
        with self.assertRaisesRegex(ValueError, "outside the ETL"):
            train(self.data, self.data / "interactions" / "model")

    def test_large_manifest_is_rejected_before_reading_parquet(self):
        manifest_path = self.data / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["counts"]["splits"]["train"] = MAX_ROWS + 1
        manifest_path.write_text(json.dumps(manifest))
        with patch("recsys.ml_data._dataset", side_effect=AssertionError("Must not read Parquet")):
            with self.assertRaisesRegex(ValueError, "200,000"):
                read_split(self.data, "train")

    def test_inaccurate_counts_and_incomplete_etl_are_rejected(self):
        path = self.data / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["counts"]["splits"]["train"] += 1
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "row count differs"):
            read_split(self.data, "train")
        (self.data / "INCOMPLETE").touch()
        with self.assertRaisesRegex(ValueError, "completed ETL"):
            read_split(self.data, "train")

    def test_invalid_ratings_rank_seed_user_and_k_are_rejected(self):
        for values in ([float("nan"), 4.0], [0.0, 4.0], [3.7, 4.0]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                RecommendationModel.fit(Ratings(np.array([1, 2]), np.array([1, 2]), np.array(values)), {1: "One", 2: "Two"})
        ratings = read_split(self.data, "train")
        for rank, seed in ((0, 42), (129, 42), (2, -1)):
            with self.subTest(rank=rank, seed=seed), self.assertRaises(ValueError):
                RecommendationModel.fit(ratings, read_titles(self.data), rank, seed)
        with self.assertRaises(ValueError):
            self.fit().recommend(0)
        with self.assertRaises(ValueError):
            self.fit().recommend(1, 0)

    def test_empty_holdout_has_actionable_error(self):
        path = next((self.data / "interactions" / "split=validation").glob("*.parquet"))
        table = pq.ParquetFile(path).read().slice(0, 0)
        pq.write_table(table, path)
        manifest_path = self.data / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["counts"]["splits"]["validation"] = 0
        manifest_path.write_text(json.dumps(manifest))
        self.train()
        with self.assertRaisesRegex(ValueError, "validation split is empty"):
            evaluate(self.data, self.output)

    def test_cli_demo_train_recommend_and_evaluation_output(self):
        output = self.root / "demo"
        with redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(main(["demo", "--output", str(output)]), 0)
        self.assertEqual(json.loads(stream.getvalue())["kind"], "synthetic_demo")
        self.assertTrue((output / "evaluation.json").exists())
        with redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(main(["recommend", "--model", str(output / "model"), "--user-id", "1"]), 0)
        self.assertEqual(len(json.loads(stream.getvalue())["recommendations"]), 2)
        report = self.root / "validation.json"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["evaluate", "--data", str(output / "data"), "--model", str(output / "model"),
                                   "--output", str(report)]), 0)
        self.assertEqual(json.loads(report.read_text())["split"], "validation")
        with redirect_stderr(io.StringIO()):
            self.assertEqual(main(["evaluate", "--data", str(output / "data"), "--model", str(output / "model"),
                                   "--output", str(report)]), 2)

    def test_cli_train_command(self):
        with redirect_stdout(io.StringIO()) as stream:
            result = main(["train", "--data", str(self.data), "--output", str(self.output), "--rank", "2"])
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(stream.getvalue())["parameters"]["training_rows"], 6)
