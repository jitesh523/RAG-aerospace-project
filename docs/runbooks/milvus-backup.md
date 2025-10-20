# Milvus Backup & Restore Runbook

This runbook outlines strategies to back up and restore Milvus collections used by the RAG API.

## Backup Strategies

- **Collection Export** (preferred for logical backups)
  - Use Milvus tools or scripts to export vector data and metadata to object storage (e.g., S3).
  - Consider chunked exports to avoid timeouts for large collections.
- **Storage Snapshots** (infrastructure-level)
  - If Milvus PVCs are backed by snapshot-capable storage (e.g., CSI snapshots), schedule periodic snapshots.
  - Coordinate snapshots with Milvus pause/flush to ensure consistency.
- **Dual-Write Ingestion**
  - During ingestion, write both to Milvus and to an immutable object store format (parquet/JSONL) to simplify disaster recovery.

## Example: Logical Export to S3

1) Enumerate all IDs and payloads for collection `MILVUS_COLLECTION`.
2) Serialize to parquet or JSONL with embeddings and metadata.
3) Upload to `s3://your-bucket/milvus-backups/<date>/`.

Pseudocode outline:
```python
# pseudocode only
from pymilvus import Collection
col = Collection("aero_docs_v1")
col.load()
res = col.query(expr="", output_fields=["id", "vector", "text", "source", "page"], limit=100000)
# write res to parquet and upload to S3
```

## Example: CSI Snapshot (Kubernetes)

- Ensure your storage class supports snapshots.
- Create a `VolumeSnapshot` for Milvus PVCs:
```yaml
apiVersion: snapshot.storage.k8s.io/v1
kind: VolumeSnapshot
metadata:
  name: milvus-data-snap
spec:
  source:
    persistentVolumeClaimName: milvus-standalone-data
  volumeSnapshotClassName: csi-snap-class
```
- Schedule with a `CronJob` or external snapshot controller.

## Restore Procedures

- **From Logical Export**
  1) Restore exported parquet/JSONL from S3.
  2) Recreate Milvus collection schema (match original dim and fields).
  3) Batch insert vectors and payloads, then build/load indexes.

- **From CSI Snapshot**
  1) Create `VolumeSnapshotContent`/`PVC` from the snapshot.
  2) Reattach PVC to Milvus StatefulSet/Deployment.

## Scheduling & Retention

- Run backups daily off-peak.
- Retain at least 7–30 days based on RPO requirements.
- Verify restore regularly (canary environment).

## Validation Checklist

- Export files present in S3 for latest run.
- Snapshot created successfully for PVCs.
- Restore test completed within acceptable RTO.

## Observability

- Expose metrics/counters for backup success/failure in your backup jobs.
- Alert on missing backups or failed snapshot operations.
