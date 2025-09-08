from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType, Collection, utility
)
from typing import List, Tuple
from src.config import Config

EMBED_DIM = 3072  # OpenAI text-embedding-3-large

def connect():
    connections.connect(alias="default", host=Config.MILVUS_HOST, port=str(Config.MILVUS_PORT))

def ensure_collection(name: str = Config.MILVUS_COLLECTION) -> Collection:
    connect()
    if utility.has_collection(name):
        return Collection(name)

    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBED_DIM),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="page", dtype=DataType.INT64),
    ]
    schema = CollectionSchema(fields, description="Aerospace chunks")
    col = Collection(name, schema)
    col.create_index("embedding", {"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 1024}})
    col.load()
    return col

def insert_rows(rows: List[Tuple[str, list, str, str, int]], name: str = Config.MILVUS_COLLECTION):
    col = ensure_collection(name)
    ids, embeds, texts, sources, pages = zip(*rows)
    col.insert([list(ids), list(embeds), list(texts), list(sources), list(pages)])
    col.flush()
