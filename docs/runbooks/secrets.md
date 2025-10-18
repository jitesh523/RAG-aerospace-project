# Secrets Runbook

- **Kubernetes Secret (`api-secrets`)**
  - Keys: `api_key`, `jwt_secret`, `jwt_issuer`, `jwt_audience`, `redis_url`.
  - Create example:
    ```bash
    kubectl create secret generic api-secrets \
      --from-literal=api_key=YOUR_API_KEY \
      --from-literal=jwt_secret=YOUR_JWT_SECRET \
      --from-literal=jwt_issuer=YOUR_ISSUER \
      --from-literal=jwt_audience=YOUR_AUDIENCE \
      --from-literal=redis_url=redis://:password@redis:6379/0 \
      -n rag-aerospace
    ```
- **Rotation**
  - Create new secret with a version suffix; update Deployment to reference new secret; rollout restart; delete old secret when all pods updated.
- **Local dev**
  - Populate `.env` with `API_KEY`, `JWT_*`, `REDIS_URL` as needed.
