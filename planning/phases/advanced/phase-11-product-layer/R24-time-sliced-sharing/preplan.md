# PRE-PLAN: R24 — Time-Sliced GPU Sharing

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** DRAFT
**Depends On:** P10 (Multi-Node), R17 (Topology-Aware Scheduling)

## Purpose

Define WHAT needs to be planned and built to turn OuterLink into a multi-user GPU cloud with time-slicing, quotas, fair scheduling, and usage accounting.

---

## Scope Definition

### In Scope

1. **Multi-user GPU access** — Multiple users share the same GPU pool concurrently
2. **VRAM quota enforcement** — Per-user/group memory limits at the CUDA interception layer
3. **Compute time scheduling** — Fair-share and priority-based kernel dispatch ordering
4. **Usage metering** — Track GPU-seconds, VRAM-hours, transfer volume per user
5. **Authentication** — Token-based identity for CLI/API, mTLS for inter-node
6. **Job queue** — Submit, schedule, run, complete lifecycle for batch workloads
7. **Interactive sessions** — Direct LD_PRELOAD sessions with quota enforcement
8. **API** — gRPC control plane, REST management, CLI tool
9. **Monitoring** — Prometheus metrics, Grafana dashboard templates

### Out of Scope (for this phase)

- Kubernetes scheduler integration (separate follow-on, depends on R24 core)
- Billing/payment systems (export usage data, external systems handle billing)
- Web UI beyond Grafana dashboards
- VM-level isolation (container runtimes handle this)
- GPU kernel preemption (requires NVIDIA driver changes, not possible in userspace)

---

## Key Decisions Needed

| # | Decision | Options | Research Doc |
|---|---|---|---|
| D1 | Scheduling granularity | Per-kernel vs cooperative vs hybrid | [02-scheduling-and-isolation.md](research/02-scheduling-and-isolation.md) |
| D2 | Use MPS on GPU nodes? | MPS for concurrency vs context-switch isolation | [02-scheduling-and-isolation.md](research/02-scheduling-and-isolation.md) |
| D3 | Fair-share algorithm | DRF vs weight-based deficit scheduling | [02-scheduling-and-isolation.md](research/02-scheduling-and-isolation.md) |
| D4 | Authentication method | JWT vs mTLS vs SSH keys | [03-gpu-cloud-architecture.md](research/03-gpu-cloud-architecture.md) |
| D5 | API protocol | gRPC + REST vs REST-only vs gRPC-only | [03-gpu-cloud-architecture.md](research/03-gpu-cloud-architecture.md) |
| D6 | Standalone vs K8s-first | Build standalone core first? Or K8s-native? | [03-gpu-cloud-architecture.md](research/03-gpu-cloud-architecture.md) |

### Preliminary Recommendations

- **D1:** Hybrid per-kernel + cooperative. Intercept `cuLaunchKernel` for admission control, use sync points as natural yield boundaries.
- **D2:** Yes, use MPS. Eliminates context switch overhead, enables concurrent multi-user execution, Volta+ provides address space isolation.
- **D3:** Weight-based deficit scheduling initially (simpler). Graduate to DRF when multi-resource contention becomes real.
- **D4:** JWT for users, mTLS for node-to-node. Add LDAP/OAuth2 as optional backends later.
- **D5:** gRPC for control plane (scheduling, quotas), REST for dashboards/monitoring.
- **D6:** Standalone first. The LAN GPU cloud use case does not require Kubernetes. Add K8s integration as a follow-on.

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Long-running kernels block fair scheduling | HIGH | MEDIUM | Kernel time estimation + admission control. Warn users about long kernels. |
| MPS + LD_PRELOAD interaction issues | MEDIUM | HIGH | Prototype MPS integration early. Fallback: no MPS, use context switching. |
| Usage accounting overhead | LOW | MEDIUM | Per-session tracking, not per-kernel. Batch writes to metrics store. |
| Users gaming the scheduler | MEDIUM | LOW | Audit logging, rate limiting, admin alerts for anomalous behavior. |
| VRAM fragmentation under multi-user | MEDIUM | MEDIUM | Memory compaction (depends on R10 tiering), over-provisioning headroom. |

---

## Dependencies

| Dependency | Status | Why Needed |
|---|---|---|
| P10: Multi-Node pooling | Required | Multiple users need multi-node GPU pool to exist |
| R17: Topology-Aware Scheduling | Required | Placement decisions need topology awareness |
| R10: Memory Tiering | Recommended | VRAM eviction to host/NVMe when quota enforcement triggers |
| P6: CUDA Completeness | Required | Must intercept all allocation/launch functions for quota tracking |

---

## Deliverables for Planning Phase

| # | Deliverable | Description |
|---|---|---|
| 1 | Architecture document | Central scheduler + per-GPU agent design |
| 2 | Quota enforcement design | Detailed algorithm for VRAM and compute time quotas |
| 3 | Scheduling algorithm spec | Fair-share + priority scheduling with kernel admission |
| 4 | Auth/authz design | JWT tokens, RBAC roles, permission model |
| 5 | API specification | Protobuf definitions for gRPC, OpenAPI for REST |
| 6 | Metrics specification | Prometheus metric names, labels, collection frequency |
| 7 | MPS integration prototype | Test MPS + LD_PRELOAD on RTX 3090 |
| 8 | Acceptance criteria | Definition of done for each sub-component |

---

## Estimated Effort

| Component | Complexity | Estimate |
|---|---|---|
| Usage metering | Low | 1-2 weeks |
| VRAM quota enforcement | Medium | 2-3 weeks |
| Auth system (JWT + mTLS) | Medium | 2-3 weeks |
| Scheduling engine (fair-share + priority) | High | 3-4 weeks |
| Job queue management | Medium | 2-3 weeks |
| gRPC API | Medium | 2-3 weeks |
| REST API + CLI | Low | 1-2 weeks |
| Prometheus metrics | Low | 1 week |
| MPS integration | Medium | 2 weeks |
| **Total** | | **~16-23 weeks** |

---

## Open Questions

1. How do we handle the transition from single-user OuterLink (current) to multi-user? Is it backward-compatible?
2. Should the central scheduler be a separate process/service, or integrated into the existing OuterLink server?
3. What's the minimum viable multi-user feature set? (Probably: auth + VRAM quotas + usage tracking)
4. How does R24 interact with R22 (live migration)? Can we migrate a user's session to a different GPU to rebalance?

---

## Related Documents

- [research/01-gpu-virtualization-landscape.md](research/01-gpu-virtualization-landscape.md)
- [research/02-scheduling-and-isolation.md](research/02-scheduling-and-isolation.md)
- [research/03-gpu-cloud-architecture.md](research/03-gpu-cloud-architecture.md)
- [R17: Topology-Aware Scheduling](../../../phase-09-distributed-os/R17-topology-aware-scheduling/)
- [R10: Memory Tiering](../../phase-07-memory-intelligence/R10-memory-tiering/)
