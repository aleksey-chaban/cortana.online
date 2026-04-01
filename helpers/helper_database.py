"""PostgreSQL bootstrap and migration helpers"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Optional

import psycopg
from psycopg import sql


DB_NAME = os.environ["PGDATABASE"]
EMBEDDING_DIM = 768


class DatabaseConfigurationError(RuntimeError):
    """Raised when required PostgreSQL settings are missing"""


REQUIRED_ENV_VARS = (
    "PGHOST",
    "PGPORT",
    "PGUSER",
    "PGPASSWORD",
)


def _require_env() -> None:
    """Ensure required PostgreSQL environment variables exist"""

    missing = [name for name in REQUIRED_ENV_VARS if not os.getenv(name)]
    if missing:
        missing_csv = ", ".join(missing)
        raise DatabaseConfigurationError(
            f"Missing PostgreSQL environment variables: {missing_csv}"
        )


def _admin_conninfo(dbname: str = "postgres") -> str:
    """Build a psycopg connection string for the requested database"""

    _require_env()
    return (
        f"host={os.environ['PGHOST']} "
        f"port={os.environ['PGPORT']} "
        f"dbname={dbname} "
        f"user={os.environ['PGUSER']} "
        f"password={os.environ['PGPASSWORD']}"
    )


def get_connection(dbname: str = DB_NAME) -> psycopg.Connection:
    """Connect to the requested PostgreSQL database"""

    return psycopg.connect(_admin_conninfo(dbname))


def create_database(dbname: str = DB_NAME) -> None:
    """Create the application database if it does not already exist"""

    with psycopg.connect(_admin_conninfo("postgres"), autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (dbname,),
            )
            exists = cursor.fetchone()

            if exists:
                return

            cursor.execute(
                sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname))
            )


def create_extension(connection: psycopg.Connection) -> None:
    """Enable pgvector in the current database"""

    with connection.cursor() as cursor:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
    connection.commit()


def create_tables(connection: psycopg.Connection) -> None:
    """Create tables with keys and types"""

    with connection.cursor() as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS authors (
                author_id SERIAL PRIMARY KEY,
                author TEXT NOT NULL UNIQUE
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS channels (
                channel_id SMALLSERIAL PRIMARY KEY,
                channel TEXT NOT NULL UNIQUE
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS types (
                type_id SMALLSERIAL PRIMARY KEY,
                type TEXT NOT NULL UNIQUE
            )
            """
        )

        cursor.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS e_year (
                    e_year_id BIGSERIAL PRIMARY KEY,
                    year TEXT NOT NULL UNIQUE,
                    embedding vector({})
                )
                """
            ).format(sql.SQL(str(EMBEDDING_DIM)))
        )

        cursor.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS e_month (
                    e_month_id BIGSERIAL PRIMARY KEY,
                    month TEXT NOT NULL,
                    embedding vector({}),
                    embedding_year BIGINT REFERENCES e_year(e_year_id),
                    UNIQUE (month, embedding_year)
                )
                """
            ).format(sql.SQL(str(EMBEDDING_DIM)))
        )

        cursor.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    embedding_id BIGSERIAL PRIMARY KEY,
                    embedding vector({}) NOT NULL,
                    embedding_month BIGINT REFERENCES e_month(e_month_id),
                    embedding_year BIGINT REFERENCES e_year(e_year_id)
                )
                """
            ).format(sql.SQL(str(EMBEDDING_DIM)))
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS threads (
                content_id BIGSERIAL PRIMARY KEY,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                author INTEGER NOT NULL REFERENCES authors(author_id),
                channel SMALLINT NOT NULL REFERENCES channels(channel_id),
                type SMALLINT NOT NULL REFERENCES types(type_id),
                content TEXT NOT NULL,
                embedding BIGINT REFERENCES embeddings(embedding_id),
                e_month BIGINT REFERENCES e_month(e_month_id),
                e_year BIGINT REFERENCES e_year(e_year_id)
            )
            """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_threads_created_at
            ON threads (created_at)
            """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_threads_author
            ON threads (author)
            """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_threads_channel
            ON threads (channel)
            """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_threads_type
            ON threads (type)
            """
        )

    connection.commit()


def bootstrap_database() -> None:
    """Create the database, enable extensions, and create all tables"""

    create_database(DB_NAME)
    with get_connection(DB_NAME) as connection:
        create_extension(connection)
        create_tables(connection)


def seed_lookup_tables(connection: psycopg.Connection) -> None:
    """Seed base author, channel, and type values used by the app"""

    with connection.cursor() as cursor:
        cursor.executemany(
            "INSERT INTO authors (author) VALUES (%s) ON CONFLICT (author) DO NOTHING",
            [("model",), ("user",)],
        )
        cursor.executemany(
            "INSERT INTO channels (channel) VALUES (%s) ON CONFLICT (channel) DO NOTHING",
            [("user",), ("analysis",), ("final",)],
        )
        cursor.executemany(
            "INSERT INTO types (type) VALUES (%s) ON CONFLICT (type) DO NOTHING",
            [("text",)],
        )
    connection.commit()


def _get_or_create_lookup_id(
    connection: psycopg.Connection,
    table_name: str,
    id_column: str,
    value_column: str,
    value: str,
) -> int:
    """Resolve a lookup row id, creating the row when needed"""

    with connection.cursor() as cursor:
        cursor.execute(
            sql.SQL(
                "INSERT INTO {table} ({value_column}) VALUES (%s) "
                "ON CONFLICT ({value_column}) DO NOTHING"
            ).format(
                table=sql.Identifier(table_name),
                value_column=sql.Identifier(value_column),
            ),
            (value,),
        )
        cursor.execute(
            sql.SQL("SELECT {id_column} FROM {table} WHERE {value_column} = %s").format(
                id_column=sql.Identifier(id_column),
                table=sql.Identifier(table_name),
                value_column=sql.Identifier(value_column),
            ),
            (value,),
        )
        row = cursor.fetchone()

    connection.commit()
    if row is None:
        raise RuntimeError(f"Failed to resolve {table_name} id for value: {value}")

    return int(row[0])


def migrate_thread_csv(
    connection: psycopg.Connection,
    csv_path: str | Path,
) -> int:
    """Move legacy thread.csv rows into PostgreSQL threads"""

    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file does not exist: {csv_file}")

    imported_count = 0

    with csv_file.open(mode="r", encoding="utf8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)

        required_columns = {"datetime", "author", "channel", "type", "content"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            missing_csv = ", ".join(sorted(missing_columns))
            raise ValueError(f"CSV is missing required columns: {missing_csv}")

        with connection.cursor() as cursor:
            for row in reader:
                created_at = row["datetime"]
                author_name = (row.get("author") or "").strip()
                channel_name = (row.get("channel") or "").strip()
                type_name = (row.get("type") or "").strip()
                content = row["content"]

                author_id: Optional[int] = None
                if author_name:
                    author_id = _get_or_create_lookup_id(
                        connection=connection,
                        table_name="authors",
                        id_column="author_id",
                        value_column="author",
                        value=author_name,
                    )

                channel_id = _get_or_create_lookup_id(
                    connection=connection,
                    table_name="channels",
                    id_column="channel_id",
                    value_column="channel",
                    value=channel_name,
                )
                type_id = _get_or_create_lookup_id(
                    connection=connection,
                    table_name="types",
                    id_column="type_id",
                    value_column="type",
                    value=type_name,
                )

                cursor.execute(
                    """
                    INSERT INTO threads (
                        created_at,
                        author,
                        channel,
                        type,
                        content,
                        embedding,
                        e_month,
                        e_year
                    )
                    VALUES (%s, %s, %s, %s, %s, NULL, NULL, NULL)
                    """,
                    (created_at, author_id, channel_id, type_id, content),
                )
                imported_count += 1

        connection.commit()

    return imported_count


def initialize_database_and_seed() -> None:
    """Set up PostgreSQL and seed core lookup values"""

    bootstrap_database()
    root = os.getcwd()
    csv_path = Path(
        os.path.join(
            root,
            "storage",
            "thread.csv"
        )
    )

    with get_connection(DB_NAME) as connection:
        seed_lookup_tables(connection)

        if csv_path.exists():
            imported_count = migrate_thread_csv(connection, csv_path)
            print(f"Imported {imported_count} rows from {csv_path}")
        else:
            print(f"CSV file not found, skipping migration: {csv_path}")


if __name__ == "__main__":
    initialize_database_and_seed()
    print("Database bootstrap complete")
