"""Submit entry to model"""

import argparse
import os
import sys

import pandas
import numpy

from app.src.integrations import model

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

    return parser.parse_args()

##
## End inputs
##

def main():
    """Submit entry to model."""

    datetime_format = "%Y-%m-%dT%H:%M:%S.%fZ"

    args = parse_args()

    root = os.getcwd()

    config = os.path.join(
        root,
        "config"
    )

    storage = os.path.join(
        root,
        "storage"
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

    thread = os.path.join(
        storage,
        "thread.csv"
    )

    entry_dict = {
        "datetime": pandas.Timestamp.now(tz="utc").strftime(datetime_format),
        "author": args.author,
        "channel": "user",
        "type": "text",
        "content": args.entry,
    }

    entry_df = pandas.DataFrame([entry_dict])
    conversation_df = pandas.read_csv(thread, index_col="entry")
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
        conversation_df["datetime"].dt.tz_convert("US/Eastern").dt.normalize() ==
        pandas.Timestamp.now(tz="US/Eastern").normalize()
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

    for statement in statements:
        if statement.get("role") == "assistant":
            author = "Cortana"
            channel = statement.get("channel")
            contents = statement.get("content")
            for content in contents:
                content_type = content.get("type")
                content_value = content.get("text")
                conversation_df.loc[len(conversation_df)] = {
                    "datetime": pandas.Timestamp.now(tz="utc"),
                    "author": author,
                    "channel": channel,
                    "type": content_type,
                    "content": content_value
                }

    conversation_df["datetime"] = (
        conversation_df["datetime"]
        .dt.tz_convert("UTC")
        .dt.strftime(datetime_format)
    )

    conversation_df.to_csv(thread, index=True, index_label="entry")
    if conversation_df["channel"].iloc[-1] == "final":
        print(conversation_df["content"].iloc[-1])
        print(f"Conversation token count: {token_count}")
    else:
        raise Exception("Cortana did not respond.")


if __name__ == "__main__":
    main()
