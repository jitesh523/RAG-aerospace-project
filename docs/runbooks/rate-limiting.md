# Rate Limiting Runbook

- **Where**: Implemented in `src/app/fastapi_app.py` using Redis when `REDIS_URL` is set, else in-memory per replica.
- **Configuration**: `RATE_LIMIT_PER_MIN` (per API key/IP), `REDIS_URL` for shared state.
- **Bursts**: Current simple counter per minute window; consider leaky bucket/token bucket at gateway for precise shaping.
- **Bypass for internal**: Use internal network paths or a separate service account with a different limit if needed.

## Verify
- Check `/metrics` for request counters and status breakdown.
- In CI, use k6 smoke to validate 429 under sustained load.

## Tune
- Increase/decrease `RATE_LIMIT_PER_MIN`.
- Enable Redis for multi-replica correctness.
