from pathlib import Path
import os

ROOT = Path(__file__).parent

# DB_PATH = "/Users/jonathan/mres_thesis/wine_analysis_hplc_uv/wines.db"

if dd_res := os.environ["UV_DATA_DIR"]:
    DATA_DIR = Path(dd_res)
else:
    raise ValueError("no dd result found")

RAW_LIB_DIR = DATA_DIR / "raw_uv"

DIRTY_ST = DATA_DIR / "dirty_sample_tracker_names_corrected.parquet"

CT_UN = str(Path(os.getenv("CELLAR_TRACKER_UN")))

CT_PW = str(Path(os.getenv("CELLAR_TRACKER_PW")))

DB_PATH_UV = DATA_DIR / "raw_uv.db"

NC_RAW = DATA_DIR / "raw_uv.nc"
