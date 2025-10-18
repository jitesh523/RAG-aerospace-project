# Scaling Runbook

- **Horizontal scaling**: Enable HPA in Helm and tune min/max replicas and CPU target.
- **Vertical scaling**: Adjust `resources` in `k8s/helm/templates/deployment.yaml` via values.
- **Milvus throughput**: Scale Milvus cluster and provision faster storage for high QPS.
- **Redis**: Enable Redis (`redis.enabled=true`) for shared rate limiting and cache.

## Steps
- **Enable HPA**:
  ```bash
  helm upgrade --install rag-aerospace ./k8s/helm -n rag-aerospace \
    --set hpa.enabled=true --set hpa.minReplicas=2 --set hpa.maxReplicas=8 --set hpa.targetCPUUtilizationPercentage=70
  ```
- **Increase replicas** (without HPA): set `spec.replicas` via values or `--set replicas=N` if templated.
- **Observe**: Use `/metrics` and Grafana dashboard to verify latency and error rates.
