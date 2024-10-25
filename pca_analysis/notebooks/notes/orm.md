# ORM

Notes on use of ORM in Python with DuckDB

## Setup

See [here](https://github.com/Mause/duckdb_engine?tab=readme-ov-file#installation).

## SQL ALchemy

contains Core and ORM layers.

The ORM maps python objects to database tables.

### Initialisation

Source: [here](https://docs.sqlalchemy.org/en/20/tutorial/engine.html#tutorial-engine), https://docs.sqlalchemy.org/en/20/tutorial/dbapi_transactions.html#executing-with-an-orm-session

- Connection is established with `sqlalchemy.create_engine` which takes a url string to the database.
- In the ORM, the Session objct manages the connection `sqlalchemy.orm.Session`
- queries are executed through the `Session.execute` method.

1. Establish declarative base `sqlalchemy.orm.DeclarativeBase
2. create mapped classes subclassing the base
3. 