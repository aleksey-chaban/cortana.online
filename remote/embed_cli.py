"""Output embeddings"""

import json
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer


def main(text: str) -> str:
    """
    Call embedding model
    """

    model_path = Path(__file__).resolve().parent / "model"

    model = SentenceTransformer(
        str(model_path),
        local_files_only=True,
    )

    vectors = model.encode(
        text,
        normalize_embeddings=True,
    )

    return json.dumps(vectors.tolist())


if __name__ == "__main__":
    payload = json.loads(sys.stdin.read())

    text = payload.get("text")
    count_true = payload.get("count_true")

    if count_true:
        print(len(main(text)))
    else:
        print(main(text))
