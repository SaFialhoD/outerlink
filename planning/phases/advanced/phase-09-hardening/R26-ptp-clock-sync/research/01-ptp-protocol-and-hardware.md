# R26 Research: PTP Protocol Fundamentals & ConnectX-5 Hardware Support

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** DRAFT

## Purpose

Document IEEE 1588 PTP protocol mechanics, hardware timestamping advantages, and ConnectX-5's specific PTP capabilities. This is the foundation for understanding what clock synchronization precision OuterLink can achieve across nodes.

---

## 1. IEEE 1588 Precision Time Protocol Overview

### What PTP Is

IEEE 1588 PTP (Precision Time Protocol) is a network protocol for synchronizing clocks across distributed systems to sub-microsecond accuracy. Unlike NTP (which achieves ~1ms over LAN), PTP with hardware timestamping reaches **<100ns** on a well-configured LAN.

PTP operates in a master-slave hierarchy: one clock (the **grandmaster**) serves as the reference, and all other clocks discipline themselves to it.

### Protocol Version History

| Version | Standard | Key Changes |
|---------|----------|-------------|
| PTPv1 | IEEE 1588-2002 | Original spec, software timestamps only |
| PTPv2 | IEEE 1588-2008 | Hardware timestamping, transparent clocks, profiles |
| PTPv2.1 | IEEE 1588-2019 | Security (AUTHENTICATION TLV), enhanced BMCA, renamed terminology |

PTPv2 is the relevant version for OuterLink. PTPv2.1 adds security but is backwards-compatible.

### Message Exchange (Delay Request-Response Mechanism)

PTP computes the clock offset between master and slave using four timestamps:

```
Master                          Slave
  |                               |
  |--- Sync (t1) --------------->|  (slave records arrival as t2)
  |--- Follow_Up (carries t1) -->|
  |                               |
  |<--- Delay_Req (t3) ----------|  (slave records departure as t3)
  |--- Delay_Resp (carries t4) ->|
  |                               |
```

**Four Timestamps:**
- **t1**: Master's egress time of Sync message (in Follow_Up, or embedded in one-step mode)
- **t2**: Slave's ingress time of Sync message
- **t3**: Slave's egress time of Delay_Req message
- **t4**: Master's ingress time of Delay_Req message

**Offset Calculation:**

```
offset = ((t2 - t1) - (t4 - t3)) / 2
delay  = ((t2 - t1) + (t4 - t3)) / 2
```

This assumes symmetric path delay. On a direct Ethernet link (OuterLink's case with ConnectX-5 back-to-back or through a switch), this assumption holds well.

### One-Step vs Two-Step Mode

| Mode | Description | Accuracy |
|------|-------------|----------|
| **One-step** | Timestamp embedded in Sync message itself at wire time | Best (no Follow_Up jitter) |
| **Two-step** | Sync sent first, Follow_Up carries the timestamp | Slightly worse (Follow_Up processing delay) |

ConnectX-5 supports two-step mode. ConnectX-6 Dx and later support one-step hardware timestamping for better accuracy.

### PTP Profiles

| Profile | Standard | Use Case | Transport |
|---------|----------|----------|-----------|
| **Default (1588)** | IEEE 1588 | General purpose | UDP/IPv4, UDP/IPv6, L2 |
| **gPTP (802.1AS)** | IEEE 802.1AS | Automotive, A/V bridging | L2 only, peer-to-peer delay |
| **Telecom** | ITU-T G.8275.1/2 | Telco frequency sync | UDP or L2, tight holdover |

**For OuterLink:** The default profile over L2 (raw Ethernet) is ideal. L2 transport avoids IP stack overhead and provides the lowest jitter. gPTP is unnecessarily restrictive (designed for bridged networks). Telecom profiles add complexity we don't need.

---

## 2. Hardware vs Software Timestamping

### The Critical Difference

| Aspect | Software Timestamping | Hardware Timestamping |
|--------|----------------------|----------------------|
| **Where timestamp is taken** | Kernel network stack (socket layer) | NIC hardware (PHY/MAC boundary) |
| **Typical accuracy** | 1-100 microseconds | 10-100 nanoseconds |
| **Jitter sources** | IRQ latency, scheduler, kernel overhead | Clock quantization only |
| **CPU load impact** | Yes — busy CPU = worse timestamps | No — timestamps are in hardware |
| **Achieved PTP offset** | ~10-50 microseconds | **<100 nanoseconds** |

### Why Hardware Timestamping Matters for OuterLink

OuterLink needs sub-microsecond synchronization for coordinated kernel launches. At 100Gbps wire speed, 1 microsecond of clock error equals ~12.5KB of data position uncertainty. For coordinating GPU kernel launches across nodes, we need offsets under 1 microsecond to make "launch at time T" meaningful.

Software timestamping (10-50us) would make coordinated launches effectively useless — the jitter would be larger than many kernel execution times. Hardware timestamping (<100ns) gives us the precision budget we need.

### How Hardware Timestamping Works

The NIC contains a free-running counter (the PTP Hardware Clock, or PHC) that increments at the NIC's core frequency. When a PTP packet crosses the PHY/MAC boundary:

1. The hardware latches the current PHC counter value
2. This timestamp is attached to the packet descriptor
3. The driver reads the timestamp when processing the packet
4. No kernel scheduling jitter, no interrupt latency — just clock quantization noise

---

## 3. ConnectX-5 PTP Capabilities

### Hardware Clock (PHC)

The ConnectX-5 contains a PTP Hardware Clock exposed to Linux as `/dev/ptpN`. Key characteristics:

| Feature | ConnectX-5 | ConnectX-6 Dx (for comparison) |
|---------|------------|-------------------------------|
| PHC present | Yes | Yes |
| PHC format | Free-running counter (needs translation) | True UTC/TAI real-time clock |
| HW TX timestamping | Yes | Yes |
| HW RX timestamping | Yes | Yes |
| One-step PTP | No (two-step only) | Yes |
| PPS output | Limited (1 GPIO) | 2 GPIO (PPS in + configurable PPS out) |
| Accuracy achievable | <100ns typical, <50ns optimized | <20ns |
| Time-triggered TX scheduling | Via DPDK `tx_pp` devarg | Native hardware support |

### Driver Support

The `mlx5` kernel driver provides PHC support via `CONFIG_PTP_1588_CLOCK`. The driver:

- Registers a PTP clock device (`/dev/ptpN`)
- Uses reader/writer spinlocks to protect timecounter access (important for high packet rates with RSS)
- Supports `ethtool -T` for capability reporting
- Tested with linuxptp project (ptp4l + phc2sys)

### Verifying ConnectX-5 PTP Support

```bash
# Check hardware timestamping capabilities
ethtool -T enp1s0f0

# Expected output includes:
# Hardware Transmit Timestamp Modes: ON
# Hardware Receive Filter Modes: ALL
# PTP Hardware Clock: N (some integer)
```

### Configuration Requirements

1. **Firmware**: Must be up-to-date (MLNX_OFED or inbox mlx5 driver)
2. **Kernel config**: `CONFIG_PTP_1588_CLOCK=y` or `=m`
3. **Driver**: `mlx5_core` loaded (standard with MLNX_OFED or modern kernels)
4. **PHC sync**: Must run `phc2sys` to keep system clock aligned with PHC

### ConnectX-5 vs ConnectX-6 for PTP — Does It Matter?

For OuterLink's needs, ConnectX-5 is sufficient:

- **Sub-microsecond target**: ConnectX-5 achieves <100ns offset, well within our <1us target
- **Two-step mode**: Slightly more overhead than one-step, but accuracy difference is <10ns
- **No native UTC clock**: Requires PHC-to-system-clock translation, but `phc2sys` handles this automatically
- **Limited GPIO**: Only matters if we need PPS output for external devices (we don't)

ConnectX-6 Dx would be nicer (true UTC clock, one-step mode, better accuracy), but ConnectX-5 already exceeds our requirements.

---

## 4. Boundary Clocks vs Transparent Clocks

### Relevance to OuterLink

If OuterLink nodes are connected through a switch (rather than back-to-back), the switch's PTP behavior matters.

| Clock Type | Description | Impact |
|------------|-------------|--------|
| **Boundary Clock** | Switch participates in PTP, has its own PHC, syncs to master, re-distributes to downstream | Good accuracy, adds one hop of error |
| **Transparent Clock** | Switch modifies PTP packets to account for its own queuing delay | Best accuracy, switch doesn't need its own clock |
| **No PTP support** | Switch is "dumb" — adds variable queuing delay to PTP packets | Worst accuracy, 1-10us jitter from switch |

**For OuterLink:**
- **Back-to-back ConnectX-5 links**: No switch involved, best accuracy
- **Through a managed switch with PTP**: Boundary or transparent clock mode, good accuracy
- **Through a cheap unmanaged switch**: Software timestamping-level accuracy (~10us), not ideal but potentially still usable

Our primary target is direct ConnectX-5 links (RDMA topology), where PTP accuracy is best.

---

## 5. Achieved Accuracy — What to Expect

Based on published results and community experience:

| Configuration | Typical Offset | Notes |
|---------------|---------------|-------|
| ConnectX-5, L2 transport, HW TS, direct link | **20-80ns** | Best case for our hardware |
| ConnectX-5, L2 transport, HW TS, through PTP switch | **50-200ns** | Depends on switch quality |
| ConnectX-5, UDP transport, HW TS | **50-150ns** | IP stack adds some jitter |
| Any NIC, software TS | **5-50 microseconds** | Not useful for coordinated launches |
| NTP over LAN | **0.5-5 milliseconds** | Not useful for any of our goals |

**Verdict:** ConnectX-5 with hardware timestamping over L2 transport comfortably achieves our <1 microsecond target, with typical offsets in the 20-80ns range on a direct link.

---

## 6. Key Takeaways for OuterLink

1. **PTP with HW timestamping is the right choice** — achieves 20-80ns offset on ConnectX-5, far better than our 1us target
2. **Use L2 transport** — lowest jitter, no IP stack overhead, perfect for our dedicated RDMA network
3. **Two-step mode is fine** — ConnectX-5 doesn't support one-step, but two-step still achieves <100ns
4. **Default PTP profile** — no need for gPTP or telecom profiles
5. **Direct links are best** — our ConnectX-5 RDMA topology is already optimal for PTP
6. **PHC requires phc2sys** — system clock won't be PTP-accurate unless we run phc2sys
7. **BMCA handles grandmaster election** — automatic, but we should set Priority1 for deterministic selection

---

## Related Documents

- [02-linux-ptp-stack.md](02-linux-ptp-stack.md) — Configuration of linuxptp for our hardware
- [03-gpu-clock-integration.md](03-gpu-clock-integration.md) — Connecting PTP time to GPU operations
- [R17: Topology-Aware Scheduling](../../phase-08-smart-memory/R17-topology-aware-scheduling/README.md) — Uses PTP for latency measurement

## Open Questions

- [ ] Does ConnectX-5 firmware version affect PTP accuracy? (Need to test with our specific cards)
- [ ] What's the actual offset we achieve on Pedro's hardware? (Benchmark needed)
- [ ] Should we use L2 or UDP transport? (L2 is lower jitter, but UDP works through routers — not relevant for our LAN)
- [x] Is ConnectX-5 sufficient vs ConnectX-6? **Yes — <100ns meets our <1us target with margin**
