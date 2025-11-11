# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
- Phase 1 planning and upcoming reliability features (autoscaling, dependency circuit breakers)

## [0.1.0] - 2025-11-11
### Added
- Phase 0 baseline hardening
  - Non-local readiness requires API_KEY or JWT configuration
  - Metrics protection enforced in non-local environments
  - Environment-aware default rate limits (local: 60/min, non-local: 30/min)
- Runbooks and SLOs documentation
  - `docs/runbooks/on-call.md`
  - `docs/runbooks/slo.md`
