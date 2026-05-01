"""Generic load functions"""

import os
import sys
import json


def load_json(
    settings_path: str,
) -> dict:
    """
    Load settings
    """

    try:
        with open(settings_path, mode="r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        sys.exit("Required file not found")

ROOT = os.getcwd()

SETTINGS_PATH = os.path.join(ROOT, "settings.json")
if os.path.join(ROOT, "config", "settings_true.json"):
    SETTINGS_PATH = os.path.join (ROOT, "config", "settings_true.json")

settings = load_json(
    settings_path=SETTINGS_PATH,
)

MODEL_NAME = settings.get("model_name")
MODEL_IDENTITY = settings.get("model_identity")
EMBEDDER_MAX = settings.get("embedder_max")
CONTEXT_LEN = settings.get("context_len")
HISTORY_LEN = settings.get("history_len")
RETURN_TOKENS = settings.get("return_tokens")
TEMPERATURE = settings.get("temperature")
TOP_P = settings.get("top_p")
TIMEZONE = settings.get("timezone")

try:
    models = settings.get("statements")
except KeyError:
    sys.exit("Paths not found")

CUSTOM_DEVELOPER = models.get("custom_developer")
CUSTOM_PROFILE = models.get("custom_profile")

try:
    models = settings.get("paths")
except KeyError:
    sys.exit("Paths not found")

SERVER_PATH = models.get("server")
MODEL_PATH = models.get("model")
EMBEDDER_PATH = models.get("embedder")

EMBED_PYTHON = os.path.join(
    EMBEDDER_PATH,
    ".venv",
    "bin",
    "python",
)

EMBED_SCRIPT = os.path.join(
    EMBEDDER_PATH,
    "embed_cli.py",
)

ENV_PATH = os.path.join(ROOT, ".env")

env = load_json(
    settings_path=ENV_PATH,
)

HOST = env.get("PGHOST")
PORT = env.get("PGPORT")
DATABASE = env.get("PGDATABASE")
USER = env.get("PGUSER")
PASSWORD = env.get("PGPASSWORD")
