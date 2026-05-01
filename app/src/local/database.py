"""Database interactions"""

import datetime

import pandas
import sqlalchemy

from app.src.helpers.variables import (
    HOST,
    PORT,
    DATABASE,
    USER,
    PASSWORD,
    HISTORY_LEN,
    CONTEXT_LEN,
)

engine = sqlalchemy.create_engine(
    (
        f"postgresql+psycopg://"
        f"{USER}:{PASSWORD}"
        f"@{HOST}:{PORT}/{DATABASE}"
    )
)


def read_db(
    query: str,
) -> pandas.DataFrame:
    """Return database query results"""

    df = pandas.read_sql_query(
        sqlalchemy.text(query),
        engine,
        params={
            "limit": HISTORY_LEN,
        },
    )
    return df


def search_embeddings_db(
    vector: str,
    history_filter: datetime.datetime,
) -> pandas.DataFrame:
    """Return closest stored embeddings"""

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
            "limit": CONTEXT_LEN,
        },
    )


def write_db(query: str, params: dict | None = None) -> None:
    """Write to database"""

    with engine.begin() as connection:
        connection.execute(sqlalchemy.text(query), params or {})
