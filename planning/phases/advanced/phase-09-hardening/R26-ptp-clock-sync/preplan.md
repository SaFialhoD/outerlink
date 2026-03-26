# R26: Hardware Clock Sync via PTP — Pre-Plan

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** DRAFT
**Phase:** 9 — Hardening
**Priority:** HIGH

## Purpose

Define the scope, dependencies, risks, and implementation phases for adding PTP-based hardware clock synchronization to OuterLink. This enables sub-microsecond clock alignment across cluster nodes, coordinated GPU kernel launches, and accurate distributed profiling.

---

## 1. Scope Definition

### In Scope

| Component | Description |
|-----------|-------------|
| PTP daemon management | OuterLink spawns, monitors, and configures ptp4l + phc2sys |
| Grandmaster selection | Topology-aware Priority1 assignment via BMCA |
| Clock offset monitoring | Read PTP sync status and offset from ptp4l (pmc UDS) |
| System clock health | Detect when PTP sync is lost, alert cluster |
| Coordinated launch API | `schedule_at(ptp_time)` for synchronized kernel launches across nodes |
| GPU-to-PTP calibration | Periodic offset measurement between GPU %globaltimer and PTP-synced system clock |
| Distributed profiling timestamps | PTP-synced timestamps on all transport and compute events |
| Configuration | PTP config generation, per-node priority assignment, transport selection |

### Out of Scope (for now)

| Excluded | Reason | Where It Goes |
|----------|--------|---------------|
| Custom PTP implementation in Rust | linuxptp is proven; evaluate statime later | R26 Phase 2 consideration |
| GPS/atomic clock grandmaster | We use freerunning oscillators on LAN | Not needed for LAN-only deployment |
| PTP over WAN | OuterLink is LAN-only | Not applicable |
| One-step PTP mode | ConnectX-5 doesn't support it | Revisit if upgrading to ConnectX-6 |
| GPU clock adjustment | GPU %globaltimer cannot be modified | Work around via calibration |

### Deliverables

1. PTP management subsystem in `outerlink-server` (Rust)
2. `ptp4l.conf` and `phc2sys.conf` template generation
3. Grandmaster election integration with topology manager (R17)
4. `ClockSync` API for Rust code: `fn now_ptp() -> Instant`, `fn schedule_at(time: PtpTime, action: F)`
5. GPU-to-PTP calibration module
6. Monitoring dashboard data (offset, servo state, grandmaster ID)
7. Integration tests verifying <1us cross-node offset

---

## 2. Dependencies

### Upstream (what R26 needs)

| Dependency | Component | Status | Why |
|------------|-----------|--------|-----|
| ConnectX-5 hardware | Hardware | Available | PHC for hardware timestamping |
| mlx5 driver with PTP | Kernel driver | Available (MLNX_OFED or inbox) | Exposes /dev/ptpN |
| linuxptp package | System package | Available (apt install linuxptp) | ptp4l + phc2sys daemons |
| P8 working transport | OuterLink core | In progress | Need network path for PTP messages |
| R17 topology manager | OuterLink advanced | In progress | Provides node roles for grandmaster selection |

### Downstream (what depends on R26)

| Dependent | Component | How It Uses R26 |
|-----------|-----------|-----------------|
| R25 | Cooperative Kernel Splitting | Coordinated kernel launches across GPUs |
| R30 | Persistent Kernels | Timing coordination for network-fed data handoff |
| R17 | Topology-Aware Scheduling | Accurate one-way latency measurement with PTP timestamps |
| R20 | NCCL Backend | Optional: PTP-synced profiling of collective operations |
| General | Distributed profiling | Common timeline for all cluster events |

---

## 3. Key Decisions

### Decision 1: linuxptp vs Custom Rust PTP

| Option | Pros | Cons |
|--------|------|------|
| **A: linuxptp (ptp4l + phc2sys)** | Battle-tested, handles edge cases, maintained upstream | External process, parsing UDS protocol |
| **B: statime (Rust crate)** | Native integration, no external processes | Less mature, we own all edge cases |
| **C: Hybrid** | Start with linuxptp, migrate to statime later | Two paths to maintain during transition |

**Recommendation:** Option A (linuxptp) for initial implementation. Evaluate Option C if we need tighter integration.

### Decision 2: PTP Transport

| Option | Accuracy | Complexity |
|--------|----------|------------|
| **L2 (raw Ethernet)** | Best (~20-80ns) | Requires same L2 domain |
| **UDP/IPv4** | Good (~50-150ns) | Works across routers |

**Recommendation:** L2 transport. Our ConnectX-5 links are direct or through a single switch — always same L2 domain.

### Decision 3: Coordinated Launch Strategy

| Strategy | Jitter | GPU Waste | Complexity |
|----------|--------|-----------|------------|
| **Host-side PTP scheduling** | ~5-20us | None | Low |
| **GPU-side spin-wait** | ~0.5-2us | Wastes cycles spinning | Medium |
| **Hybrid (host + GPU spin)** | ~1-2us | Minimal (short spin) | Medium |

**Recommendation:** Start with host-side scheduling (simplest). Add GPU-side spin for R25 when sub-microsecond coordination is needed.

### Decision 4: Grandmaster Selection

| Option | Description | Tradeoff |
|--------|-------------|----------|
| **Static priority** | Admin configures Priority1 per node | Simple, deterministic |
| **Topology-driven** | Topology manager assigns Priority1 based on role | Automatic, adapts to cluster changes |
| **Best oscillator** | Measure clock stability, promote most stable | Optimal accuracy, complex |

**Recommendation:** Topology-driven (Option 2). The coordinator node becomes grandmaster via Priority1=100, with the secondary at Priority1=110.

---

## 4. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| PTP accuracy insufficient (<1us not achieved) | Low | High | ConnectX-5 HW TS achieves <100ns on LAN. Test on actual hardware early. |
| Power management destroys PTP accuracy | Medium | High | Disable C-states and PCIe ASPM on OuterLink nodes. Document in setup guide. |
| phc2sys fights with NTP/chrony | Medium | Medium | Disable NTP on the PTP interface. Document conflict. |
| GPU timer drift exceeds calibration ability | Low | Medium | Calibrate every 1-10 seconds. Linear drift model handles most cases. |
| linuxptp not available on target distro | Low | Low | Available on all major distros (Ubuntu, Fedora, Arch). Include in setup requirements. |
| Switch without PTP support degrades accuracy | Medium | Medium | Document switch requirements. Back-to-back links are always best. |
| Grandmaster failover causes clock jump | Low | Medium | BMCA handles failover automatically. Monitor for offset spikes during transition. |
| Kernel launch jitter dominates PTP precision | High | Low | This is inherent — PTP still provides accurate timestamps even if launch jitter is larger. CUDA graphs can reduce launch jitter. |

---

## 5. Implementation Phases

### Phase 1: PTP Infrastructure (2-3 weeks)

**Goal:** PTP running and verified on all nodes, system clocks synced to <1us.

| Task | Estimate | Details |
|------|----------|---------|
| ptp4l config generation from OuterLink config | 2 days | Template `ptp4l.conf` with node-specific Priority1 |
| phc2sys config generation | 1 day | Match ptp4l settings, automatic mode |
| Process management (spawn, monitor, restart) | 3 days | In `outerlink-server`, manage ptp4l/phc2sys lifecycle |
| PTP health monitoring via pmc UDS | 3 days | Parse offset, servo state, port state from pmc |
| Grandmaster selection integration | 2 days | Topology manager sets Priority1 based on node role |
| Verification test: cross-node offset measurement | 2 days | Automated test confirming <1us offset |
| Documentation: setup guide, troubleshooting | 1 day | Power management, NTP conflicts, switch requirements |

### Phase 2: Clock API & Profiling (1-2 weeks)

**Goal:** Rust API for PTP time, distributed profiling with common timeline.

| Task | Estimate | Details |
|------|----------|---------|
| `ClockSync` Rust module | 2 days | `now_ptp()`, `is_synced()`, `offset_ns()` |
| GPU-to-PTP calibration | 3 days | Bracketing technique, periodic recalibration |
| Distributed event timestamps | 2 days | PTP timestamps on transport and compute events |
| Profiling data collection & export | 2 days | Collect events from all nodes, export timeline |

### Phase 3: Coordinated Launches (1-2 weeks)

**Goal:** Ability to launch kernels at a coordinated PTP time across nodes.

| Task | Estimate | Details |
|------|----------|---------|
| `schedule_at(ptp_time)` API | 2 days | Host-side spin-wait until PTP time, then launch |
| Launch jitter measurement | 1 day | Benchmark actual jitter on our hardware |
| GPU-side spin-wait (optional) | 3 days | PTX %globaltimer spin for sub-us coordination |
| Integration with R25 (when ready) | 2 days | Hook coordinated launches into kernel splitting |

### Phase 4: Hardening (1 week)

**Goal:** Production-ready, handles all failure modes.

| Task | Estimate | Details |
|------|----------|---------|
| Grandmaster failover testing | 2 days | Kill grandmaster, verify recovery |
| Clock jump handling | 1 day | Detect and handle large offset changes |
| Long-running drift verification | 1 day | Run for 24h, verify offset stays <1us |
| Performance impact measurement | 1 day | Verify PTP doesn't affect transport throughput |

**Total estimate: 5-8 weeks**

---

## 6. Success Criteria

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Cross-node clock offset | **<1 microsecond** sustained | ptp4l master offset log |
| Clock offset after grandmaster failover | **<10 microseconds** within 5 seconds | Kill grandmaster, measure recovery time |
| Coordinated kernel launch jitter (host-side) | **<20 microseconds** | Measure cudaLaunchKernel timing spread across nodes |
| Coordinated kernel launch jitter (GPU spin) | **<5 microseconds** | Measure %globaltimer alignment at kernel start |
| GPU-to-PTP calibration accuracy | **<2 microseconds** | Compare GPU timestamps to PTP-synced system clock |
| PTP convergence time (cold start) | **<2 minutes** to <1us offset | Measure time from ptp4l start to stable s2 state |
| Profiling timeline accuracy | **<1 microsecond** across nodes | Cross-correlate known events (RDMA completions) |
| Zero impact on transport throughput | **<1% bandwidth reduction** | iperf3/perftest with and without PTP running |

---

## 7. Open Questions

### Must Answer Before Implementation

- [ ] **NTP conflict**: Should we disable NTP entirely on OuterLink nodes, or only on the ConnectX-5 interface? Disabling NTP means no wall-clock sync from internet time sources.
- [ ] **Multiple ConnectX-5 ports**: If a node has two ConnectX-5 ports (dual-port card), do we run PTP on both? Or just one? What about bonded interfaces?
- [ ] **ptp4l permissions**: Does ptp4l require root? Can it run as a non-root user with appropriate capabilities? (CAP_NET_RAW, CAP_SYS_TIME)

### Can Answer During Implementation

- [ ] **UDS protocol format**: What's the exact binary format for pmc UDS queries? (Can reverse-engineer from pmc source or linuxptp docs)
- [ ] **statime evaluation**: When should we seriously evaluate the statime Rust crate as a replacement for linuxptp?
- [ ] **CUDA graph launch jitter**: Do CUDA graphs significantly reduce coordinated launch jitter compared to regular cudaLaunchKernel?

### Research Needed

- [ ] **Holdover accuracy**: If PTP connectivity is lost, how long can ConnectX-5's oscillator maintain <10us accuracy? (Determines failure detection urgency)
- [ ] **Temperature sensitivity**: How much does clock drift change with ambient temperature on our hardware? (Determines calibration frequency)

---

## Related Documents

- [research/01-ptp-protocol-and-hardware.md](research/01-ptp-protocol-and-hardware.md) — PTP protocol and ConnectX-5 capabilities
- [research/02-linux-ptp-stack.md](research/02-linux-ptp-stack.md) — linuxptp configuration and integration
- [research/03-gpu-clock-integration.md](research/03-gpu-clock-integration.md) — GPU clock correlation and coordinated launches
- [R17: Topology-Aware Scheduling](../../../phase-08-smart-memory/R17-topology-aware-scheduling/README.md) — Provides node roles for grandmaster selection
- [R25: Cooperative Kernel Splitting](../../../phase-12-moonshot/R25-cooperative-kernel-splitting/README.md) — Primary consumer of coordinated launches
- [R30: Persistent Kernels](../../../phase-10-compute-distribution/R30-persistent-kernels/README.md) — Timing coordination dependency
