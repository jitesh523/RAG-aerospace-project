from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType, Collection, utility
)
from typing import List, Tuple
from src.config import Config
from prometheus_client import Counter
import time
try:
    import redis as _r
except Exception:
    _r = None

EMBED_DIM = 3072  # OpenAI text-embedding-3-large

DR_DUAL_WRITE_ERRORS_TOTAL = Counter(
    "dr_dual_write_errors_total",
    "Total errors when writing to secondary cluster",
)

def connect():
    connections.connect(alias="default", host=Config.MILVUS_HOST, port=str(Config.MILVUS_PORT))

def connect_secondary():
    if not Config.MILVUS_HOST_SECONDARY:
        return False
    try:
        connections.connect(alias="secondary", host=Config.MILVUS_HOST_SECONDARY, port=str(Config.MILVUS_PORT_SECONDARY))
        return True
    except Exception:
        return False

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

def ensure_collection_secondary(name: str = Config.MILVUS_COLLECTION_SECONDARY) -> Collection | None:
    if not connect_secondary():
        return None
    if utility.has_collection(name, using="secondary"):
        return Collection(name, using="secondary")
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBED_DIM),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="page", dtype=DataType.INT64),
    ]
    schema = CollectionSchema(fields, description="Aerospace chunks (secondary)")
    col = Collection(name, schema, using="secondary")
    col.create_index("embedding", {"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 1024}})
    col.load()
    return col

def _ensure_partition(col: Collection, partition: str):
    try:
        if partition and partition not in [p.name for p in col.partitions]:
            col.create_partition(partition)
    except Exception:
        pass

def insert_rows(rows: List[Tuple[str, list, str, str, int]], name: str = Config.MILVUS_COLLECTION, partition: str | None = None):
    # primary write
    col = ensure_collection(name)
    if partition:
        _ensure_partition(col, partition)
    ids, embeds, texts, sources, pages = zip(*rows)
    col.insert([list(ids), list(embeds), list(texts), list(sources), list(pages)], partition_name=partition)
    col.flush()
    # record primary write timestamp
    try:
        if _r is not None and Config.REDIS_URL:
            rc = _r.Redis.from_url(Config.REDIS_URL, decode_responses=True)
            rc.set("dr:last_write_ts:primary", str(int(time.time())))
    except Exception:
        pass
    # optional dual-write to secondary
    if Config.DR_ENABLED and Config.DR_DUAL_WRITE and Config.MILVUS_HOST_SECONDARY:
        try:
            scol = ensure_collection_secondary(Config.MILVUS_COLLECTION_SECONDARY)
            if scol is not None:
                if partition:
                    _ensure_partition(scol, partition)
                scol.insert([list(ids), list(embeds), list(texts), list(sources), list(pages)], partition_name=partition)
                scol.flush()
                try:
                    if _r is not None and Config.REDIS_URL:
                        rc = _r.Redis.from_url(Config.REDIS_URL, decode_responses=True)
                        rc.set("dr:last_write_ts:secondary", str(int(time.time())))
                except Exception:
                    pass
        except Exception:
            try:
                DR_DUAL_WRITE_ERRORS_TOTAL.inc()
            except Exception:
                pass

def check_milvus_readiness(name: str = Config.MILVUS_COLLECTION) -> dict:
    """Return readiness info for Milvus connection and collection.
    Example: {"connected": True, "has_collection": True, "loaded": True}
    """
    info = {"connected": False, "has_collection": False, "loaded": False}
    try:
        connect()
        info["connected"] = True
        info["has_collection"] = utility.has_collection(name)
        if info["has_collection"]:
            col = Collection(name)
            # Try to load; if already loaded this is no-op
            col.load()
            info["loaded"] = True
    except Exception:
        # Keep defaults; caller can inspect
        pass
    return info

def check_milvus_readiness_secondary(name: str = Config.MILVUS_COLLECTION_SECONDARY) -> dict:
    info = {"connected": False, "has_collection": False, "loaded": False}
    if not Config.MILVUS_HOST_SECONDARY:
        return info
    try:
        if not connect_secondary():
            return info
        info["connected"] = True
        info["has_collection"] = utility.has_collection(name, using="secondary")
        if info["has_collection"]:
            col = Collection(name, using="secondary")
            col.load()
            info["loaded"] = True
    except Exception:
        pass
    return info
