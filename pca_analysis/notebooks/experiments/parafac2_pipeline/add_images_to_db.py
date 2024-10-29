"""in a reversal of the original design paradigm, I am not going to load the image data
into the db. The reason for this is that I am restricted in how much logic I can put
into python, and the storage of state in python objects due to the nature of Dash. Hence,
I/O with a database is the best course of action. How do I load it though? First use the
opportunity to define some ORMs, then for the runids in 'inc_chm', iterate over the parquets
and read them into the db.
"""

# from sqlalchemy.base import Base
from sqlalchemy.orm import Session

from database_etl.etl.etl_pipeline_raw import fetch_imgs
from pca_analysis.notebooks.experiments.parafac2_pipeline.orm import Images
from tests.test_definitions import TEST_DB_PATH
import duckdb as db
import duckdb_engine
import polars as pl

from sqlalchemy.orm import declarative_base
from sqlalchemy import create_engine, text
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()
# print(TEST_DB_PATH)
engine = create_engine("duckdb:///tests/test_raw_db.db")

logging.basicConfig(level="INFO")


def get_runids():
    query = "select runid from inc_chm"
    return [x[0] for x in engine.connect().execute(text(query))]


def get_imgs(runids: list[str]) -> pl.DataFrame:
    with engine.connect() as conn:
        paths = [
            x[0]
            for x in conn.execute(
                text("select path from inc_img_stats where runid in :runid"),
                {"runid": runids},
            ).fetchall()
        ]

        imgs = []
        for runid, path in zip(runids, paths):
            logger.info(f"extracting {runid} from {path}..")
            img = pl.read_parquet(path)
            print(img.head())
            img = img.with_columns(pl.lit(runid).alias("runid"))
            img = img.drop("id")
            img = img.unpivot(
                index=["runid", "mins"],
                variable_name="wavelength",
                value_name="abs",
            )
            img = img.with_columns(pl.col("wavelength").cast(int))
            imgs.append(img)

            logger.info(f"finished extracting {runid}.")

        result = pl.concat(imgs)

        logger.info("returning resulting concatenation.")

        return result


def main():
    """ """
    Base.metadata.create_all(engine)

    runids = get_runids()

    imgs = get_imgs(runids=runids)

    print(imgs.head())

    rows = imgs.shape[0]
    logger.info(f"total rows: {rows}..")
    with Session(engine) as session:
        for idx, row in enumerate(imgs.rows(named=True)):
            logger.info(f"inserting row {idx} / {rows}..")
            session.add(
                Images(
                    runid=row["runid"],
                    wavelength=row["wavelength"],
                    mins=row["mins"],
                    abs=row["abs"],
                )
            )
        logger.info("finished adding rows, committing now.")
        session.commit()

        logger.info("Commit complete.")


if __name__ == "__main__":
    main()
