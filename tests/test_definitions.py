from pathlib import Path

TEST_ROOT = Path(__file__).parent
TEST_DATA_DIR = ( TEST_ROOT / "test_samples").resolve()
TEST_OUTPUT_PATH = TEST_ROOT / "test_output"
TEST_DB_PATH = TEST_ROOT / "test_raw_db.db"
TEST_RESULTS_DB = TEST_ROOT / "test_pipeline_results.db"