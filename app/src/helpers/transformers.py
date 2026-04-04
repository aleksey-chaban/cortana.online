"""Generic transformation functions"""

def convert_vector_pg(
    vector: list[float],
) -> str:
    """
    Convert Python vector for PostgreSQL
    """

    return "[" + ",".join(str(float(value)) for value in vector) + "]"

def convert_pg_vector(
    value: str
) -> list[float]:
    """
    Convert PostgreSQL vector for Python
    """

    stripped = value.strip()
    body = stripped[1:-1].strip()

    if not body:
        raise ValueError("Vector is empty")

    return [float(part.strip()) for part in body.split(",")]
