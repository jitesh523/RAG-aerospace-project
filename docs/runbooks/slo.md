# Service Level Objectives (SLOs)

These SLOs define expected reliability targets for the Aerospace RAG API.

## Scope
- Endpoints: `/ask`, `/ask/stream`, `/metrics`, `/ready`, `/health`
- Environments: staging, production

## Objectives
- **Availability**
  - Target: ≥ 99.9% monthly for `/ask`
  - Measurement: ratio of successful 2xx/4xx (excluding 429) to total requests
- **Latency**
  - `/ask`: p95 ≤ 1.5s, p99 ≤ 3.0s
  - `/metrics`: p95 ≤ 300ms
- **Error rate**
  - 5xx rate ≤ 0.5% (excluding planned maintenance/canary windows)
- **Readiness**
  - Readiness probe success ≥ 99.9%

## Alerts (suggested)
- Burn-rate alerts for availability (multi-window)
- p95 `/ask` latency > 1.5s for 15m
- Circuit open count spikes
- Rate limit saturation (sustained 429s)
- DLQ growth (if async ingestion enabled)

## Dashboards
- Latency histograms, error rates, circuit breaker state, cache hit ratios
- Cost per tenant, token estimates, embedding batch metrics

## Review
- Quarterly SLO review with product + SRE
