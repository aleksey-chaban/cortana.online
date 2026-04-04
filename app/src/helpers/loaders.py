"""Generic load functions"""

import os
import sys
import json


def load_settings(
    settings_true: bool,
    root: str,
) -> dict:
    """
    Load settings
    """

    settings_file = "settings.json"
    if settings_true:
        settings_file = "settings_true.json"
    
    settings_path = os.path.join(
        root,
        "config",
        settings_file
    )

    try:
        with open(settings_path, mode="r", encoding="utf-8") as settings:
            return json.load(settings)
    except FileNotFoundError:
        sys.exit("Settings not found")


def load_paths(
    settings_true: bool,
    root: str,
) -> tuple:
    """Resolve paths"""

    settings = load_settings(
        root=root,
        settings_true=settings_true
    )

    model_identity = settings.get("model_identity")
    embedder_max = settings.get("embedder_max")
    context_len = settings.get("context_len")
    history_len = settings.get("history_len")
    return_tokens = settings.get("return_tokens")

    try:
        models = settings.get("paths")
    except KeyError:
        sys.exit("Paths not found")

    model_path = models.get("model")
    embedder_path = models.get("embedder")

    embed_python = os.path.join(
        embedder_path,
        ".venv",
        "bin",
        "python",
    )

    embed_script = os.path.join(
        embedder_path,
        "embed_cli.py"
    )

    return (
        model_path,
        embed_python,
        embed_script,
        model_identity,
        embedder_max,
        context_len,
        history_len,
        return_tokens,
    )
