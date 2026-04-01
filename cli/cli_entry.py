"""Submit entry to model"""

import argparse
import os
import sys

import pandas
import numpy

from app.src.local import model
from app.src.local import database

##
## Inputs
##

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Submit entry to model."
    )

    parser.add_argument(
        "--entry",
        type=str,
        required=True,
        help="User entry."
    )

    parser.add_argument(
        "--author",
        type=str,
        default=None,
        help="Source of the entry.",
    )

    parser.add_argument(
        "--settings-true",
        action="store_true",
        help="Employ true settings.",
    )

    parser.add_argument(
        "--developer-true",
        action="store_true",
        help="Employ true developer entry.",
    )

    parser.add_argument(
        "--memories-true",
        action="store_true",
        help="Employ memories entry.",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.35,
        help="Sampling temperature.",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.90,
        help="Top-p nucleus sampling.",
    )

    parser.add_argument(
        "--time-zone",
        type=str,
        default="US/Eastern",
        help="Set time zone to consider for daily memories."
    )

    return parser.parse_args()

##
## End inputs
##

def main():
    """Submit entry to model."""

    datetime_format = "%Y-%m-%dT%H:%M:%S.%fZ"
    model_name = os.getenv("AINAME")
    args = parse_args()
    root = os.getcwd()

    config = os.path.join(
        root,
        "config"
    )

    durable_memories = {
        "author": args.author,
        "channel": "user",
        "type": "text",
        "content": ""
    }

    if args.memories_true:
        memories = os.path.join(
            config,
            f"durable_memories-{args.author}.txt"
        )

        try:
            with open(memories, mode="r", encoding="utf8") as f:
                durable_memories["content"] = f.read()
        except FileNotFoundError:
            sys.exit("Author does not exist.")

    entry_dict = {
        "datetime": pandas.Timestamp.now(tz="utc").strftime(datetime_format),
        "author": args.author,
        "channel": "user",
        "type": "text",
        "content": args.entry,
    }

    entry_df = pandas.DataFrame([entry_dict])

    conversation_df = database.read_db(
        """
        SELECT
            t.created_at AS datetime,
            a.author AS author,
            c.channel AS channel,
            ty.type AS type,
            t.content AS content
        FROM threads t
        JOIN authors a ON t.author = a.author_id
        JOIN channels c ON t.channel = c.channel_id
        JOIN types ty ON t.type = ty.type_id
        ORDER BY t.content_id
        """
    )

    conversation_df = pandas.concat([conversation_df, entry_df], ignore_index=True)
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
    conversation_df_today = conversation_df[
        conversation_df["datetime"].dt.tz_convert(args.time_zone).dt.normalize() ==
        pandas.Timestamp.now(tz=args.time_zone).normalize()
    ]

    conversation_list = conversation_df_today.to_dict(orient="records")

    token_count, statements = model.submit_entry(
        entries=conversation_list,
        settings_true=args.settings_true,
        developer_true=args.developer_true,
        memories_entry=durable_memories,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        root=root,
    )

    database.write_db(
        """
        INSERT INTO threads (created_at, author, channel, type, content)
        VALUES (
            :created_at,
            (SELECT author_id FROM authors WHERE author = :author),
            (SELECT channel_id FROM channels WHERE channel = :channel),
            (SELECT type_id FROM types WHERE type = :type),
            :content
        )
        """,
        {
            "created_at": entry_dict["datetime"],
            "author": args.author,
            "channel": entry_dict["channel"],
            "type": entry_dict["type"],
            "content": entry_dict["content"],
        }
    )

    final_text = None

    for statement in statements:
        if statement.get("role") != "assistant":
            continue

        author = model_name
        channel = statement.get("channel")
        contents = statement.get("content", [])

        for content in contents:
            content_type = content.get("type")
            content_value = content.get("text")

            if not author or not channel or not content_type or not content_value:
                continue

            database.write_db(
                """
                INSERT INTO threads (created_at, author, channel, type, content)
                VALUES (
                    :created_at,
                    (SELECT author_id FROM authors WHERE author = :author),
                    (SELECT channel_id FROM channels WHERE channel = :channel),
                    (SELECT type_id FROM types WHERE type = :type),
                    :content
                )
                """,
                {
                    "created_at": pandas.Timestamp.now(tz="utc").strftime(datetime_format),
                    "author": author,
                    "channel": channel,
                    "type": content_type,
                    "content": content_value,
                }
            )

            if channel == "final":
                final_text = content_value

    if final_text:
        print(final_text)
        print(f"Conversation token count: {token_count}")
    else:
        raise Exception("Cortana did not respond.")


if __name__ == "__main__":
    main()
