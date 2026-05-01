"""Submit entry to model"""

import os
import sys

import pandas
import numpy

import openai_harmony

from app.src.local import model
from app.src.local import database
from app.src.local import embedding
from app.src.helpers import transformers

from app.src.helpers.variables import (
    ROOT,
    CUSTOM_PROFILE,
    RETURN_TOKENS,
    MODEL_NAME,
)


def main(
    author: str,
    entry: str,
):
    """Submit entry to model."""

    datetime_format = "%Y-%m-%dT%H:%M:%S.%fZ"

    config = os.path.join(
        ROOT,
        "config"
    )

    durable_memories = {
        "author": author,
        "channel": "user",
        "type": "text",
        "content": ""
    }

    if CUSTOM_PROFILE:
        memories = os.path.join(
            config,
            f"durable_memories-{author}.txt"
        )

        try:
            with open(memories, mode="r", encoding="utf8") as f:
                durable_memories["content"] = f.read()
        except FileNotFoundError:
            sys.exit("Author does not exist.")

    token_limit = RETURN_TOKENS

    if token_limit > 512:
        token_limit -= 256

    durable_memories["content"] = (
            f"**Please limit your responses to {token_limit}**\n\n"
            f"{durable_memories['content']}\n\n"
        )

    entry_dict = {
        "datetime": pandas.Timestamp.now(tz="utc").strftime(datetime_format),
        "author": author,
        "channel": "user",
        "type": "text",
        "content": entry,
        "embedding": embedding.main(
            text=entry,
        )
    }

    entry_df = pandas.DataFrame([entry_dict])

    conversation_df = database.read_db(
        """
        SELECT
            t.created_at AS datetime,
            a.author AS author,
            c.channel AS channel,
            ty.type AS type,
            t.content AS content,
            e.embedding::text AS embedding
        FROM threads t
        JOIN authors a ON t.author = a.author_id
        JOIN channels c ON t.channel = c.channel_id
        JOIN types ty ON t.type = ty.type_id
        LEFT JOIN embeddings e ON t.embedding = e.embedding_id
        ORDER BY t.content_id DESC
        LIMIT :limit
        """,
    )

    conversation_df["datetime"] = pandas.to_datetime(
        conversation_df["datetime"],
        format=datetime_format,
        utc=True,
        errors="raise",
    )

    oldest_history_datetime = conversation_df["datetime"].min()

    conversation_df["embedding"] = conversation_df["embedding"].apply(
        transformers.convert_pg_vector
    )

    conversation_df["content"] = conversation_df.apply(
        lambda row: (
            f"**Memory of {row['author']} on {row['datetime']}**\n\n"
            f"{row['content']}\n\n"
            f"**End memory of {row['author']} on {row['datetime']}**"
        ),
        axis=1,
    )

    embedding_search_results_df = database.search_embeddings_db(
        vector=transformers.convert_vector_pg(
            vector=entry_dict["embedding"],
        ),
        history_filter=oldest_history_datetime,
    )

    embedding_search_results_df["content"] = embedding_search_results_df.apply(
        lambda row: (
            f"**Potential context from {row['author']} on {row['datetime']}, employ only if relevant**\n\n"
            f"{row['content']}\n\n"
            f"**End potential context from {row['author']} on {row['datetime']}**"
        ),
        axis=1,
    )

    conversation_df = pandas.concat(
        [
            embedding_search_results_df,
            conversation_df,
            entry_df
        ],
        ignore_index=True
    )

    conversation_df = conversation_df.replace({numpy.nan: None})

    user_indices = conversation_df[conversation_df["channel"] == "user"].index
    last_user_index = user_indices[-1]
    conversation_df.loc[last_user_index, "datetime"] = (
        pandas.Timestamp.now(tz="utc").strftime(datetime_format)
    )

    conversation_df["datetime"] = pandas.to_datetime(
        conversation_df["datetime"],
        format=datetime_format,
        utc=True,
        errors="raise",
    )

    conversation_list = conversation_df.to_dict(orient="records")

    statements = model.submit_entry(
        entries=conversation_list,
        memories_entry=durable_memories,
    )

    entry_embedding = transformers.convert_vector_pg(
        embedding.main(
            text=entry_dict["content"],
        )
    )

    database.write_db(
        """
        WITH inserted_embedding AS (
            INSERT INTO embeddings (
                embedding,
                embedding_day,
                embedding_month,
                embedding_year
            )
            VALUES (
                CAST(:embedding AS vector),
                NULL,
                NULL,
                NULL
            )
            RETURNING embedding_id
        )
        INSERT INTO threads (
            created_at,
            author,
            channel,
            type,
            content,
            embedding
        )
        VALUES (
            :created_at,
            (SELECT author_id FROM authors WHERE author = :author),
            (SELECT channel_id FROM channels WHERE channel = :channel),
            (SELECT type_id FROM types WHERE type = :type),
            :content,
            (SELECT embedding_id FROM inserted_embedding)
        )
        """,
        {
            "created_at": entry_dict["datetime"],
            "author": author,
            "channel": entry_dict["channel"],
            "type": entry_dict["type"],
            "content": entry_dict["content"],
            "embedding": entry_embedding,
        },
    )

    final_text = None

    for statement in statements:
        if statement.get("role") != openai_harmony.Role.ASSISTANT:
            continue

        author = MODEL_NAME
        channel = statement.get("channel")
        contents = statement.get("content", [])

        for content in contents:
            content_type = content.get("type")
            content_value = content.get("text")

            if not author or not channel or not content_type or not content_value:
                continue

            content_embedding = transformers.convert_vector_pg(
                embedding.main(
                    text=content_value,
                )
            )
            database.write_db(
                """
                WITH inserted_embedding AS (
                    INSERT INTO embeddings (
                        embedding,
                        embedding_day,
                        embedding_month,
                        embedding_year
                    )
                    VALUES (
                        CAST(:embedding AS vector),
                        NULL,
                        NULL,
                        NULL
                    )
                    RETURNING embedding_id
                )
                INSERT INTO threads (
                    created_at,
                    author,
                    channel,
                    type,
                    content,
                    embedding
                )
                VALUES (
                    :created_at,
                    (SELECT author_id FROM authors WHERE author = :author),
                    (SELECT channel_id FROM channels WHERE channel = :channel),
                    (SELECT type_id FROM types WHERE type = :type),
                    :content,
                    (SELECT embedding_id FROM inserted_embedding)
                )
                """,
                {
                    "created_at": pandas.Timestamp.now(tz="utc").strftime(datetime_format),
                    "author": author,
                    "channel": channel,
                    "type": content_type,
                    "content": content_value,
                    "embedding": content_embedding,
                },
            )

            if channel == "final":
                final_text = content_value

    if final_text:
        return final_text

    raise Exception("Cortana did not respond.")
