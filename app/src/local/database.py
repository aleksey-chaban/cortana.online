"""Database interactions"""

import os

import pandas
import sqlalchemy


HOST = os.environ["PGHOST"]
PORT = os.environ["PGPORT"]
DATABASE = os.environ["PGDATABASE"]
USER = os.environ["PGUSER"]
PASSWORD = os.environ["PGPASSWORD"]

engine = sqlalchemy.create_engine(
    (
        f"postgresql+psycopg://"
        f"{USER}:{PASSWORD}"
        f"@{HOST}:{PORT}/{DATABASE}"
    )
)

def read_db(query: str) -> pandas.DataFrame:
    """Return database query results"""

    df = pandas.read_sql_query(query, engine)
    return df

def write_db(query: str, params: dict | None = None) -> None:
    """Write to database"""

    with engine.begin() as connection:
        connection.execute(sqlalchemy.text(query), params or {})
