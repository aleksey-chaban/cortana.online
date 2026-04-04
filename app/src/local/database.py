"""Database interactions"""

import os
import datetime

import pandas
import sqlalchemy

from app.src.helpers import loaders


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


def read_db(
    query: str,
    settings_true: bool = False,
    root: str = os.getcwd(),
) -> pandas.DataFrame:
    """Return database query results"""

    (
        _,
        _,
        _,
        _,
        _,
        _,
        history_len,
        _,
    ) = loaders.load_paths(
        settings_true=settings_true,
        root=root,
    )

    df = pandas.read_sql_query(
        sqlalchemy.text(query),
        engine,
        params={
            "limit": history_len,
        }
    )
    return df


def search_embeddings_db(
    vector: str,
    history_filter: datetime.datetime,
    settings_true: bool = False,
    root: str = os.getcwd(),
) -> pandas.DataFrame:
    """Return closest stored embeddings"""

    (
        _,
        _,
        _,
        _,
        _,
        context_len,
        _,
        _,
    ) = loaders.load_paths(
        settings_true=settings_true,
        root=root,
    )

    query = sqlalchemy.text(
        """
        SELECT
            t.created_at as datetime,
            a.author,
            c.channel,
            t.content,
            e.embedding::text,
            ty.type
        FROM embeddings e
        JOIN threads t ON t.embedding = e.embedding_id
        JOIN authors a ON t.author = a.author_id
        JOIN channels c ON t.channel = c.channel_id
        JOIN types ty ON t.type = ty.type_id
        WHERE e.embedding IS NOT NULL
            AND t.created_at < :history_filter
        ORDER BY e.embedding <=> CAST(:embedding AS vector)
        LIMIT :limit
        """
    )

    return pandas.read_sql_query(
        query,
        engine,
        params={
            "embedding": vector,
            "history_filter": history_filter,
            "limit": context_len,
        },
    )


def write_db(query: str, params: dict | None = None) -> None:
    """Write to database"""

    with engine.begin() as connection:
        connection.execute(sqlalchemy.text(query), params or {})
