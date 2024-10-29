from sqlalchemy import create_engine

engine = create_engine("duckdb:///tests/test_raw_db.db", echo="debug")
