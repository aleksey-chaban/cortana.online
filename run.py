"""Submit entry to model"""

import os
import argparse

from app.src.helpers import orchestrator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Submit entry to model."
    )

    parser.add_argument(
        "--author",
        type=str,
        help="User name."
    )

    parser.add_argument(
        "entry",
        type=str,
        help="User entry."
    )

    args = parser.parse_args()

    ROOT = os.getcwd()

    AUTHOR = "Chief"
    if args.author:
        AUTHOR = args.author

    ENTRY = args.entry

    received_statement = orchestrator.main(
        author=AUTHOR,
        entry=ENTRY,
    )

    print(received_statement)
