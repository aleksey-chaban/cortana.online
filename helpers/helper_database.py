"""PostgreSQL bootstrap helper"""

import os

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
            CREATE TABLE IF NOT EXISTS public.authors
            (
                author_id integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                author text NOT NULL UNIQUE
            );
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS public.channels
            (
                channel_id smallint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                channel text NOT NULL UNIQUE
            );
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS public.types
            (
                type_id smallint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                type text NOT NULL UNIQUE
            );
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS public.e_year
            (
                e_year_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                year text NOT NULL UNIQUE,
                embedding vector(768)
            );
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS public.e_month
            (
                e_month_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                month text NOT NULL UNIQUE,
                embedding vector(768),
                embedding_year bigint REFERENCES public.e_year (e_year_id)
            );
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS public.e_day
            (
                e_day_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                date date NOT NULL UNIQUE,
                embedding vector(768),
                embedding_month bigint REFERENCES public.e_month (e_month_id),
                embedding_year bigint REFERENCES public.e_year (e_year_id)
            );
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS public.embeddings
            (
                embedding_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                embedding vector(768) NOT NULL,
                embedding_day bigint REFERENCES public.e_day (e_day_id),
                embedding_month bigint REFERENCES public.e_month (e_month_id),
                embedding_year bigint REFERENCES public.e_year (e_year_id)
            );
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS public.threads
            (
                content_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                created_at timestamptz NOT NULL DEFAULT now(),
                author integer NOT NULL REFERENCES public.authors (author_id),
                channel smallint NOT NULL REFERENCES public.channels (channel_id),
                type smallint NOT NULL REFERENCES public.types (type_id),
                content text NOT NULL,
                embedding bigint REFERENCES public.embeddings (embedding_id),
                e_day bigint REFERENCES public.e_day (e_day_id),
                e_month bigint REFERENCES public.e_month (e_month_id),
                e_year bigint REFERENCES public.e_year (e_year_id)
            );

            CREATE INDEX IF NOT EXISTS idx_threads_author
                ON public.threads (author);

            CREATE INDEX IF NOT EXISTS idx_threads_channel
                ON public.threads (channel);

            CREATE INDEX IF NOT EXISTS idx_threads_created_at
                ON public.threads (created_at);

            CREATE INDEX IF NOT EXISTS idx_threads_type
                ON public.threads (type);
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



def initialize_database_and_seed() -> None:
    """Set up PostgreSQL and seed core lookup values"""

    bootstrap_database()

    with get_connection(DB_NAME) as connection:
        seed_lookup_tables(connection)



if __name__ == "__main__":
    initialize_database_and_seed()
    print("Database bootstrap complete")
