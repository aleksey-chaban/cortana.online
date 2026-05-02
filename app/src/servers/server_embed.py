"""Initiate embeddings server"""

import threading

import flask
import sentence_transformers


server_embed_api = flask.Flask(__name__)

model_lock = threading.Lock()

model = sentence_transformers.SentenceTransformer(
    "/Users/{NAME}/Models/embeddinggemma/model",
    local_files_only=True,
)


@server_embed_api.post("/get_token_count")
def get_token_count():
    """Count tokens"""

    data = flask.request.get_json()
    text = data.get("text", "")

    with model_lock:
        tokens = model.tokenizer.encode(
            text,
            add_special_tokens=True,
        )

    return flask.jsonify(
        {
            "token_count": len(tokens),
        }
    )


@server_embed_api.post("/get_embeddings")
def get_embeddings():
    """Generate embedding"""

    data = flask.request.get_json()
    text = data.get("text", "")

    with model_lock:
        embedding = model.encode(
            text,
            normalize_embeddings=True,
        )

    return flask.jsonify(
        {
            "embedding": embedding.tolist(),
        }
    )


if __name__ == "__main__":
    server_embed_api.run(
        host="127.0.0.1",
        port=8081,
    )
