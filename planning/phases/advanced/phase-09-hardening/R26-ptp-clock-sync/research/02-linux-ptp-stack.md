# R26 Research: Linux PTP Stack (linuxptp)

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** DRAFT

## Purpose

Evaluate the Linux PTP implementation (linuxptp project) for OuterLink's clock synchronization needs. Determine whether to use linuxptp directly, wrap it, or implement PTP in Rust. Cover configuration, monitoring, and integration with the RDMA stack.

---

## 1. linuxptp Project Overview

linuxptp is the standard PTP implementation for Linux, consisting of several cooperating daemons:

| Component | Role | Required for OuterLink? |
|-----------|------|------------------------|
| **ptp4l** | PTP daemon — synchronizes the NIC's PHC to the grandmaster | **Yes** — core PTP engine |
| **phc2sys** | Synchronizes system clock to/from PHC | **Yes** — GPU timing uses system clock |
| **pmc** | PTP management client — query/configure running ptp4l | **Yes** — monitoring and diagnostics |
| **phc_ctl** | Direct PHC manipulation (read, set, adjust) | Optional — debugging tool |
| **hwstamp_ctl** | Enable/disable hardware timestamping | Optional — ptp4l does this |
| **ts2phc** | Synchronize PHC to external time source (GPS, PPS) | No — we sync to grandmaster node |
| **tz2alt** | Timezone to PTP alternate time offset | No |

### Architecture

```
               +-----------+     PTP messages      +-----------+
               |  ptp4l    |<------- network ------>|  ptp4l    |
               |  (master) |                        |  (slave)  |
               +-----------+                        +-----------+
                     |                                    |
                PHC (NIC HW clock)                  PHC (NIC HW clock)
                     |                                    |
               +-----------+                        +-----------+
               | phc2sys   |                        | phc2sys   |
               +-----------+                        +-----------+
                     |                                    |
               System Clock                         System Clock
               (CLOCK_REALTIME)                     (CLOCK_REALTIME)
```

---

## 2. ptp4l Configuration for ConnectX-5

### Basic Configuration File (`/etc/ptp4l.conf`)

```ini
[global]
# Use hardware timestamping (default, but be explicit)
time_stamping           hardware

# L2 transport for lowest jitter on dedicated RDMA network
network_transport       L2

# Sync interval: 2^0 = 1 second (default)
# Can reduce to 2^-1 = 0.5s or 2^-3 = 125ms for faster convergence
logSyncInterval         0

# Announce interval: 2^1 = 2 seconds
logAnnounceInterval     1

# Delay request interval: match sync interval
logMinDelayReqInterval  0

# PI servo constants (0.0 = auto-select for HW timestamping)
# Auto values: Kp=0.7, Ki=0.3 for hardware TS
pi_proportional_const   0.0
pi_integral_const       0.0

# Step threshold: step the clock if offset > 1 second on startup
step_threshold          1.0

# First step threshold: allow stepping on first measurement
first_step_threshold    0.00002

# Clock class for grandmaster (248 = default/freerunning)
clockClass              248

# Priority for BMCA grandmaster selection
# Lower = higher priority
priority1               128
priority2               128

# Domain (all OuterLink nodes must use the same domain)
domainNumber            0

# Logging
verbose                 1
summary_interval        1

[enp1s0f0]
# ConnectX-5 interface
```

### Starting ptp4l

```bash
# As grandmaster (on the designated master node):
ptp4l -f /etc/ptp4l.conf -i enp1s0f0

# As slave (on all other nodes):
# Same command — BMCA auto-selects master based on priority1
ptp4l -f /etc/ptp4l.conf -i enp1s0f0
```

### Key ptp4l Options

| Option | Description | Our Setting |
|--------|-------------|-------------|
| `-H` | Hardware timestamping (default) | Yes |
| `-S` | Software timestamping | No |
| `-2` | L2 transport | Yes (via config) |
| `-4` | UDP/IPv4 transport | No |
| `-i <iface>` | Network interface | ConnectX-5 interface |
| `-f <config>` | Configuration file | `/etc/ptp4l.conf` |
| `-m` | Print messages to stdout | Yes (for debugging) |

---

## 3. phc2sys Configuration

phc2sys synchronizes the system clock (`CLOCK_REALTIME`) to/from the NIC's PHC. This is critical because GPU timing APIs use the system clock domain.

### Starting phc2sys

```bash
# Automatic mode: follows ptp4l's state changes
phc2sys -a -r -r

# Manual mode: sync system clock from PHC on enp1s0f0
phc2sys -s enp1s0f0 -c CLOCK_REALTIME -O 0 -w

# -s: source clock (PHC)
# -c: sink clock (system clock)
# -O 0: offset between TAI and UTC (0 if ptp4l uses UTC)
# -w: wait for ptp4l to synchronize before starting
```

### phc2sys Tuning Parameters

| Parameter | Description | Default | Recommendation |
|-----------|-------------|---------|----------------|
| `pi_proportional_const` | Kp for system clock servo | 0.0 (auto) | Keep auto |
| `pi_integral_const` | Ki for system clock servo | 0.0 (auto) | Keep auto |
| `step_threshold` | Step clock if offset > threshold | 0.0 (disabled) | 0.00002 (20us) |
| `servo_num_offset_values` | Samples to confirm locked state | 10 | 10 |
| `servo_offset_threshold` | Offset threshold for LOCKED_STABLE | 0 (disabled) | 50 (50ns for HW TS) |

### Power Management Warning

Power-saving modes (C-states, PCIe ASPM) can add >100us wake-up latency, destroying PTP accuracy. For OuterLink nodes:

```bash
# Disable CPU C-states deeper than C1
echo 1 > /sys/devices/system/cpu/cpu*/cpuidle/state2/disable

# Disable PCIe ASPM
echo performance > /sys/module/pcie_aspm/parameters/policy

# Or via kernel parameter: pcie_aspm=off
```

---

## 4. Grandmaster Selection (BMCA)

### How BMCA Works

The Best Master Clock Algorithm runs on every PTP-capable port continuously. It selects the grandmaster based on a strict priority ordering:

1. **Priority 1** (0-255, lower wins) — administratively set, overrides everything
2. **Clock Class** — quality category (6 = GPS-locked, 248 = freerunning, etc.)
3. **Clock Accuracy** — estimated offset from UTC
4. **Clock Variance** — stability metric (Allan variance)
5. **Priority 2** (0-255, lower wins) — tiebreaker
6. **Clock Identity** — MAC-based unique ID (final tiebreaker)

### OuterLink Grandmaster Strategy

Since all our nodes have the same hardware (ConnectX-5, freerunning oscillators), clock class/accuracy/variance will be identical. We control grandmaster selection via **Priority 1**:

```
Node             Priority 1    Role
─────────────────────────────────────
Primary server   100           Grandmaster (preferred)
Secondary server 110           Backup grandmaster
Worker node 1    128           Slave
Worker node 2    128           Slave
Worker node N    128           Slave
```

**Failover:** If the grandmaster goes down, BMCA automatically promotes the next-best node (priority 110) within one announce interval (~2 seconds). All slaves re-sync to the new grandmaster.

**OuterLink integration:** The topology manager (R17) should set `priority1` based on node role. The designated "coordinator" node becomes grandmaster.

---

## 5. Servo Algorithms (PI Controller)

### How the Clock Servo Works

ptp4l uses a PI (Proportional-Integral) controller to discipline the PHC:

```
frequency_adjustment = Kp * offset + Ki * integral(offset)
```

**Servo States:**
- **s0 (UNLOCKED)**: Collecting initial samples
- **s1 (STEP)**: Clock stepped to correct large initial offset
- **s2 (LOCKED)**: Frequency steering, offset converging
- **LOCKED_STABLE**: Offset consistently below threshold (optional)

**Default PI Constants (Hardware Timestamping):**
- Kp = 0.7 (proportional gain)
- Ki = 0.3 (integral gain)

**Convergence Timeline:**
- Initial step: within first 2-3 sync intervals
- Locked (s2): ~10-30 seconds
- Stable (<100ns): ~1-2 minutes
- Optimal (<50ns): ~5 minutes with stable network

### Drift Compensation for Long-Running Workloads

The PI controller continuously adjusts the clock frequency to track drift. For long-running ML training jobs:

- **Short-term drift**: Handled automatically by the PI controller
- **Temperature-induced drift**: Crystal oscillators drift ~1ppm per degree C. The PI controller tracks this with ~30 second lag
- **Worst case**: If ptp4l loses contact with grandmaster, the PHC free-runs with its last frequency correction. Typical holdover: <1us/hour for a good oscillator

---

## 6. Monitoring and Diagnostics

### pmc (PTP Management Client)

```bash
# Query current clock state
pmc -u -b 0 'GET CURRENT_DATA_SET'

# Query port state (MASTER, SLAVE, LISTENING)
pmc -u -b 0 'GET PORT_DATA_SET'

# Query grandmaster identity
pmc -u -b 0 'GET PARENT_DATA_SET'

# Query time properties
pmc -u -b 0 'GET TIME_PROPERTIES_DATA_SET'
```

### Offset Monitoring

ptp4l outputs continuous offset reports:

```
ptp4l[1234.567]: master offset         23 s2 freq  +1234 path delay        456
```

Fields:
- `master offset`: Current offset from grandmaster in nanoseconds
- `s2`: Servo state (s0=unlocked, s1=step, s2=locked)
- `freq`: Frequency adjustment in ppb
- `path delay`: Measured one-way delay in nanoseconds

### Programmatic Monitoring

For OuterLink integration, we can:

1. **Parse ptp4l stdout** — simple but fragile
2. **Use pmc programmatically** — pmc supports UDS (Unix Domain Socket) protocol
3. **Read PHC directly** — `clock_gettime(CLOCK_REALTIME, ...)` after phc2sys sync
4. **Query offset from ptp4l's UDS** — most reliable for integration

---

## 7. Rust Integration Options

### Option A: Use linuxptp as External Process (Recommended)

Run ptp4l and phc2sys as system services, managed by OuterLink:

```
OuterLink daemon
    ├── Spawns/monitors ptp4l process
    ├── Spawns/monitors phc2sys process
    ├── Reads offset via pmc (UDS protocol)
    └── Uses clock_gettime(CLOCK_REALTIME) for PTP-synced time
```

**Pros:**
- linuxptp is battle-tested, widely deployed
- Handles all edge cases (BMCA, servo tuning, failover)
- Zero Rust PTP implementation effort
- Updates via package manager

**Cons:**
- External process dependency
- Parsing pmc output or implementing UDS protocol
- Less control over servo behavior

### Option B: Statime (Pure Rust PTP)

The `statime` crate (from Pendulum Project) implements IEEE 1588-2019 in pure Rust:

- **Crate**: `statime` (core library, `no_std` compatible)
- **Linux daemon**: `statime-linux` (ready-to-use PTP daemon)
- **Features**: BMCA, ordinary/boundary clock, platform-agnostic, virtual clock overlay
- **Maturity**: Active development, usable but less battle-tested than linuxptp

**Pros:**
- Native Rust integration, no external processes
- Can embed PTP logic directly into OuterLink daemon
- Full control over servo and timing behavior

**Cons:**
- Less mature than linuxptp
- We'd need to handle all edge cases ourselves
- Hardware clock access still requires Linux kernel PHC interface
- More maintenance burden

### Option C: Hybrid (Recommended for Production)

Phase 1: Use linuxptp (ptp4l + phc2sys) for proven reliability
Phase 2: Evaluate statime for tighter integration once PTP needs are well understood

### Accessing PTP Time from Rust

Regardless of which PTP daemon runs, Rust code reads PTP-synced time via:

```rust
use std::time::SystemTime;
// After phc2sys has synced CLOCK_REALTIME to PHC:
let now = SystemTime::now();

// For nanosecond precision:
use libc::{clock_gettime, timespec, CLOCK_REALTIME};
let mut ts = timespec { tv_sec: 0, tv_nsec: 0 };
unsafe { clock_gettime(CLOCK_REALTIME, &mut ts); }
// ts.tv_sec and ts.tv_nsec are now PTP-synchronized
```

---

## 8. Integration with RDMA Stack

### Timestamped RDMA Operations

ConnectX-5 can timestamp RDMA operations (not just Ethernet packets). This enables:

- **Measuring actual RDMA transfer latency**: Timestamp at send, timestamp at receive, compute difference using PTP-synced clocks
- **Profiling transport layer**: Know exactly when each RDMA operation completes
- **Correlating with GPU events**: Match RDMA completion to GPU kernel launch times

### RDMA Completion Timestamps

The `ibv_wc` (work completion) structure includes a `timestamp` field when completion timestamping is enabled. This timestamp comes from the same PHC used by PTP, so it's directly comparable across PTP-synced nodes.

```
Node A: RDMA Send timestamp = T_send (from PHC_A, PTP-synced)
Node B: RDMA Recv timestamp = T_recv (from PHC_B, PTP-synced)
One-way latency = T_recv - T_send  (accurate to ~100ns)
```

This is a huge win for R17 (Topology-Aware Scheduling) — we can measure actual transfer latencies with hardware precision.

---

## 9. Verdict for OuterLink

| Decision | Choice | Rationale |
|----------|--------|-----------|
| PTP implementation | linuxptp (ptp4l + phc2sys) initially | Proven, handles all edge cases |
| Future consideration | statime (Rust) for tighter integration | Evaluate once PTP needs are stable |
| Transport | L2 (raw Ethernet) | Lowest jitter on dedicated RDMA network |
| Grandmaster selection | Priority1-based, set by topology manager | Deterministic, matches cluster roles |
| Monitoring | pmc UDS protocol from Rust | Programmatic access to offset data |
| System clock sync | phc2sys in automatic mode | Required for GPU timing correlation |
| Time access from Rust | `clock_gettime(CLOCK_REALTIME)` | Simple, nanosecond resolution, PTP-synced after phc2sys |

---

## Related Documents

- [01-ptp-protocol-and-hardware.md](01-ptp-protocol-and-hardware.md) — PTP protocol fundamentals
- [03-gpu-clock-integration.md](03-gpu-clock-integration.md) — Connecting PTP to GPU timing
- [R17: Topology-Aware Scheduling](../../phase-08-smart-memory/R17-topology-aware-scheduling/README.md) — Uses PTP for latency measurement

## Open Questions

- [ ] What's the UDS protocol format for pmc? (Need to implement in Rust for monitoring)
- [ ] Does statime-linux support ConnectX-5 PHC out of the box? (Need to verify)
- [ ] Can we read RDMA completion timestamps from ibverbs in Rust? (rdma-core Rust bindings)
- [ ] How does phc2sys interact with NTP if ntpd/chronyd is also running? (Potential conflict)
- [ ] Should we disable NTP on OuterLink nodes to avoid clock fighting? (Probably yes on the RDMA interface)
