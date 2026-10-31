import hashlib
import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from recsys.cli import main
from recsys.etl_config import ETLConfig, fingerprint


class ETLConfigTests(unittest.TestCase):
    def test_fingerprint_matches_known_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "ratings.csv"
            source.write_bytes(b"movie-ratings\n")
            result = fingerprint(source)
            self.assertEqual(result["sha256"], hashlib.sha256(b"movie-ratings\n").hexdigest())
            self.assertEqual(result["bytes"], 14)

    def test_rejects_bad_boundaries_and_buckets(self):
        for train, validation, buckets in [(200, 100, 1), (-1, 10, 1), (10, 10, 1),
                                           (0, 2**63, 1), (10, 20, 0), (10, 20, 1025)]:
            config = ETLConfig(Path("ratings.csv"), Path("movies.csv"),
                               Path("unused-output"), train, validation, buckets)
            with self.subTest(config=config), self.assertRaises(ValueError):
                config.validate()

    def test_refuses_existing_output_and_identical_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ratings, movies = root / "ratings.csv", root / "movies.csv"
            ratings.touch()
            movies.touch()
            with self.assertRaisesRegex(ValueError, "already exists"):
                ETLConfig(ratings, movies, root, 10, 20).validate()
            with self.assertRaisesRegex(ValueError, "different files"):
                ETLConfig(ratings, ratings, root / "run", 10, 20).validate()

    def test_cli_validates_before_starting_spark(self):
        with redirect_stderr(io.StringIO()) as error:
            result = main(["etl", "--ratings", "missing.csv", "--movies", "movies.csv",
                           "--output", "new-run", "--train-end", "20", "--validation-end", "10"])
        self.assertEqual(result, 2)
        self.assertIn("train-end", error.getvalue())
