"""Process strings into embeddings"""

import json
import math
import urllib.request

from app.src.helpers.variables import (
    EMBEDDER_SERVER_PATH,
)


def split_text(
        text: str,
        token_count: int,
        max_tokens: int = 1792
) -> list[str]:
    """Split text into equal parts"""

    split_str = [ text ]

    if token_count > max_tokens:
        part_count = math.ceil( token_count / max_tokens )
        part_size = math.ceil( token_count / part_count )

        split_str = [
            text [ i:i+part_size ] for i in range( 0, len( text ), part_size )
        ]

    return split_str


def normalize_vector(
    vector: list[float]
) -> list[float]:
    """Scale a vector to standard length"""

    norm = math.sqrt(
        sum( value * value for value in vector )
    )

    if norm == 0.0:
        raise ValueError("Cannot normalize a zero vector")

    return [ value / norm for value in vector ]


def mean_vectors(
    vectors: list[list[float]]
) -> list[float]:
    """Average out vectors"""

    if not vectors:
        raise ValueError("Vectors list is empty")

    if len(vectors) == 1:
        final_vector = vectors[0]

    else:
        dimension = len(vectors[0])
        totals = [0.0] * dimension

        for vector in vectors:
            if len(vector) != dimension:
                raise ValueError("Vectors don't have the same dimensions")
            for index, value in enumerate(vector):
                totals[index] += float(value)

        count = float(len(vectors))

        vector_avg = [ value / count for value in totals ]

        final_vector = normalize_vector(
            vector=vector_avg
        )

    return final_vector


def get_token_count(
    text: str,
) -> int:
    """Send token count request"""

    request = urllib.request.Request(
        url=f"{EMBEDDER_SERVER_PATH}/get_token_count",
        data=json.dumps({"text": text}).encode("utf8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request) as response:
        payload = json.loads(response.read().decode("utf8"))

    return int(payload["token_count"])


def get_embeddings(
    text: str,
) -> list[float]:
    """Send embedding request"""

    request = urllib.request.Request(
        url=f"{EMBEDDER_SERVER_PATH}/get_embeddings",
        data=json.dumps({"text": text}).encode("utf8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request) as response:
        payload = json.loads(response.read().decode("utf8"))

    return payload["embedding"]


def main(
    text: str,
):
    """Return embedding"""

    token_count = get_token_count(
        text=text,
    )

    text_list = split_text(
        text=text,
        token_count=token_count,
    )

    embed_list = []

    for text_str in text_list:
        text_str_embed = get_embeddings(
            text=text_str,
        )

        embed_list.append(text_str_embed)

    output = mean_vectors(
        vectors=embed_list,
    )

    return output
