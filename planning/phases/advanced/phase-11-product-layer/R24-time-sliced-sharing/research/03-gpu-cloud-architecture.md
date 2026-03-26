# R24 Research: LAN GPU Cloud Architecture

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Complete
**Priority:** HIGH

## Purpose

Define the architecture for turning an OuterLink-connected LAN into a multi-user GPU cloud. Covers authentication, usage accounting, job queue management, API design, and integration with container orchestration.

---

## TL;DR

The GPU cloud layer sits ON TOP of OuterLink's existing GPU pooling. It adds: (1) identity and access control, (2) resource quotas and scheduling policies, (3) usage metering for accountability, and (4) a job submission interface. The recommended API is gRPC for the control plane (scheduling, quota management) and REST for the dashboard/monitoring. For Kubernetes environments, integrate as a custom scheduler extender or work alongside KAI Scheduler / Volcano.

---

## 1. Authentication and Authorization

### Identity Model

| Level | Description | Example |
|---|---|---|
| User | Individual human or service account | `pedro`, `training-pipeline-sa` |
| Group | Collection of users sharing quotas | `ml-team`, `interns`, `production` |
| Project | Logical grouping with own budget | `llm-finetuning`, `image-gen-v2` |

### Authentication Options

| Method | Complexity | Best For |
|---|---|---|
| mTLS (mutual TLS) | Medium | Service-to-service, automated pipelines |
| Token-based (JWT) | Low | CLI and API access |
| LDAP/AD integration | High | Enterprise with existing directory |
| SSH key-based | Low | Small teams, LAN environments |
| OAuth2/OIDC | Medium | Web dashboard, SSO |

**Recommendation for initial implementation:** Token-based (JWT) for CLI/API access, with mTLS for inter-node communication. Add LDAP/OAuth2 later for enterprise deployments.

### Authorization Model

Role-Based Access Control (RBAC) with GPU-specific permissions:

| Role | Permissions |
|---|---|
| Admin | Manage users, groups, quotas. View all usage. Configure nodes. |
| Operator | View all usage. Manage job queues. Cannot change quotas. |
| User | Submit jobs within own quota. View own usage. |
| Viewer | Read-only access to own project's metrics. |

### GPU-Specific Permissions

| Permission | Description |
|---|---|
| `gpu.allocate` | Can request GPU resources |
| `gpu.allocate.priority.high` | Can submit high-priority jobs |
| `gpu.allocate.vram.unlimited` | No VRAM quota (admin only) |
| `gpu.node.specific` | Can request specific GPU nodes |
| `gpu.preempt` | Can preempt lower-priority jobs |

---

## 2. Usage Accounting and Billing

### Metered Resources

| Resource | Unit | Measurement Method |
|---|---|---|
| GPU compute time | GPU-seconds | Time from first kernel dispatch to last kernel completion per session |
| VRAM allocation | GB-seconds | Integral of allocated VRAM over time |
| Host memory (staging) | GB-seconds | Pinned memory used for transfers |
| Network transfer | GB | Total bytes transferred to/from remote GPUs |
| Job count | Count | Number of jobs submitted |

### Accounting Granularity

| Level | What It Tracks |
|---|---|
| Per-kernel | GPU-time for each `cuLaunchKernel` (finest grain, highest overhead) |
| Per-session | GPU-time from connect to disconnect (practical for interactive use) |
| Per-job | GPU-time from job start to completion (best for batch workloads) |
| Per-user/day | Aggregate daily usage per user (billing level) |

**Recommendation:** Track per-session for real-time quota enforcement, aggregate to per-user/day for reporting and billing.

### Usage Record Schema

```
{
  "user_id": "pedro",
  "project_id": "llm-finetuning",
  "session_id": "uuid-...",
  "gpu_node": "node-02",
  "gpu_index": 0,
  "start_time": "2026-03-25T10:00:00Z",
  "end_time": "2026-03-25T10:45:00Z",
  "gpu_seconds": 2700,
  "peak_vram_gb": 18.5,
  "vram_gb_seconds": 49950,
  "transfer_gb": 12.3,
  "kernel_launches": 15420,
  "priority": "NORMAL"
}
```

### Billing Models

| Model | Description | Best For |
|---|---|---|
| Fair-share (no billing) | Users share equally, no money | Lab/team environments |
| Quota-based | Each user/group has a fixed budget of GPU-hours/month | Departments sharing hardware |
| Pay-per-use | Track usage, charge back to cost centers | Enterprise/multi-department |
| Spot pricing | Unused capacity at discount, preemptable | Maximize utilization |

---

## 3. Job Queue Management

### Job Lifecycle

```
SUBMITTED -> QUEUED -> SCHEDULING -> RUNNING -> COMPLETING -> COMPLETED
                |                       |
                v                       v
            REJECTED               FAILED / PREEMPTED
```

### Job Types

| Type | Description | Scheduling Behavior |
|---|---|---|
| Interactive | User connected via LD_PRELOAD, running CUDA app live | Immediate allocation, session-based |
| Batch | Submit script, runs unattended | Queued, scheduled when resources available |
| Array | Multiple instances of same job (hyperparameter sweep) | Scheduled as group or individually |
| Gang | Multi-GPU job, all GPUs needed simultaneously | All-or-nothing scheduling (Volcano/Kueue pattern) |

### Queue Policies

| Policy | Description |
|---|---|
| FIFO | First in, first out (default) |
| Priority | Higher priority jobs jump the queue |
| Fair-share | Users who have used less GPU time get priority |
| Backfill | Small jobs can jump ahead if they fit in gaps without delaying larger jobs |
| Deadline | Jobs with deadlines get priority as deadline approaches |

### Job Specification

```
{
  "name": "train-llama-7b",
  "user": "pedro",
  "project": "llm-finetuning",
  "resources": {
    "gpus": 2,
    "vram_per_gpu_gb": 20,
    "min_compute_capability": "8.0"
  },
  "priority": "HIGH",
  "max_runtime_hours": 24,
  "preemptable": false,
  "command": "python train.py --model llama-7b",
  "env": {
    "CUDA_VISIBLE_DEVICES": "auto"
  }
}
```

---

## 4. API Design

### Control Plane API (gRPC)

gRPC for the control plane because:
- Strongly typed (protobuf schemas)
- Efficient binary serialization
- Bidirectional streaming (for real-time status updates)
- Native support in Rust (tonic crate)

**Key Services:**

| Service | RPCs |
|---|---|
| `AuthService` | `Login`, `Refresh`, `Validate` |
| `JobService` | `Submit`, `Cancel`, `Status`, `List`, `Logs` |
| `QuotaService` | `Get`, `Set`, `Usage` |
| `SchedulerService` | `GetPolicy`, `SetPolicy`, `Drain`, `Resume` |
| `NodeService` | `List`, `Status`, `Drain`, `AddGPU`, `RemoveGPU` |
| `MetricsService` | `Query`, `Stream` |

### Management API (REST)

REST for the web dashboard and simple integrations:

| Endpoint | Method | Description |
|---|---|---|
| `/api/v1/jobs` | GET, POST | List/submit jobs |
| `/api/v1/jobs/{id}` | GET, DELETE | Get/cancel job |
| `/api/v1/users/{id}/usage` | GET | Usage report |
| `/api/v1/nodes` | GET | List GPU nodes |
| `/api/v1/nodes/{id}/gpus` | GET | List GPUs on node |
| `/api/v1/quotas/{user_id}` | GET, PUT | Get/set quota |
| `/api/v1/metrics` | GET | Prometheus-compatible metrics |

### CLI Interface

```bash
# Submit a job
outerlink job submit --name "train-llama" --gpus 2 --vram 20G -- python train.py

# Check status
outerlink job status train-llama

# View queue
outerlink queue list

# Check usage
outerlink usage --user pedro --period month

# Interactive GPU session
outerlink session start --gpus 1 --vram 24G
# Returns: export LD_PRELOAD=/path/to/outerlink.so OUTERLINK_SESSION=uuid-...
```

---

## 5. Container Orchestration Integration

### Kubernetes Integration

**Option A: Device Plugin Extension**

Extend the NVIDIA device plugin to advertise OuterLink-managed remote GPUs:

- OuterLink registers remote GPUs as `outerlink.io/gpu` resources
- Pods request `outerlink.io/gpu: 1` in their resource spec
- Device plugin injects LD_PRELOAD and connection config into the pod

**Option B: Custom Scheduler Extender**

Add OuterLink as a scheduler extender that overrides GPU scheduling:

- Default scheduler handles CPU/memory
- OuterLink extender handles GPU placement across the network
- Uses Kubernetes webhook to inject LD_PRELOAD

**Option C: Integrate with KAI Scheduler / Volcano**

Work alongside existing GPU-aware schedulers:

- KAI Scheduler (open-source Run:AI, Apache 2.0) provides queue management, fair-share, gang scheduling
- Volcano (CNCF) provides batch scheduling, topology-aware placement
- OuterLink provides the GPU pooling layer underneath

**Recommendation:** Option C for Kubernetes environments. Don't reinvent queue management. Let KAI/Volcano handle scheduling policy, OuterLink handles the GPU virtualization and network transport.

### Docker (Standalone)

For non-Kubernetes environments:

```bash
docker run --runtime=outerlink \
  -e OUTERLINK_SERVER=192.168.1.100 \
  -e OUTERLINK_GPUS=2 \
  my-training-image python train.py
```

OuterLink provides a custom OCI runtime that:
1. Injects LD_PRELOAD into the container
2. Configures OUTERLINK_* environment variables
3. Establishes connection to GPU servers

---

## 6. Existing Open-Source GPU Cloud Projects

### KAI Scheduler (formerly Run:AI Scheduler)

| Property | Details |
|---|---|
| Origin | Open-sourced by NVIDIA (from Run:AI acquisition) |
| License | Apache 2.0 |
| Focus | GPU-aware Kubernetes scheduling for AI workloads |
| Features | Queue management, fair-share, gang scheduling, bin-packing, topology-aware scheduling, GPU sharing |
| Scale | Designed for thousands of nodes |
| Release | v0.10.0 (October 2025) added topology-aware scheduling and time-based fairshare |

**Relevance:** KAI handles the SCHEDULING of GPU workloads. OuterLink handles the GPU POOLING across the network. They complement each other perfectly. We should design R24 to integrate with KAI as the scheduling backend for Kubernetes deployments.

### Volcano

| Property | Details |
|---|---|
| Governance | CNCF Incubating project |
| License | Apache 2.0 |
| Focus | Cloud-native batch scheduling for HPC/AI |
| Features | Gang scheduling, fair-share, network topology-aware, MIG/MPS dynamic partitioning, multi-cluster |
| Scale | 10,000+ nodes, 100,000+ pods |
| Release | v1.11 (2025) added GPU dynamic partitioning and multi-cluster scheduling |

**Relevance:** Volcano provides batch scheduling semantics (gang scheduling, queue management) that OuterLink needs for multi-GPU jobs. Like KAI, it handles scheduling while OuterLink handles the underlying GPU virtualization.

### Kueue (Kubernetes SIG)

| Property | Details |
|---|---|
| Governance | Kubernetes SIG Scheduling |
| License | Apache 2.0 |
| Focus | Job queueing and admission control |
| Features | ClusterQueues, LocalQueues, ResourceFlavors, fair-share |

**Relevance:** Kueue provides admission control (should this job be allowed to start?) while OuterLink provides the GPU resources. Can work together.

---

## 7. Monitoring and Observability

### Metrics to Export (Prometheus Format)

| Metric | Type | Labels |
|---|---|---|
| `outerlink_gpu_utilization_percent` | Gauge | `node`, `gpu`, `user` |
| `outerlink_vram_used_bytes` | Gauge | `node`, `gpu`, `user` |
| `outerlink_vram_quota_bytes` | Gauge | `user`, `project` |
| `outerlink_gpu_seconds_total` | Counter | `user`, `project`, `priority` |
| `outerlink_jobs_submitted_total` | Counter | `user`, `project`, `status` |
| `outerlink_queue_depth` | Gauge | `queue`, `priority` |
| `outerlink_kernel_launches_total` | Counter | `node`, `gpu`, `user` |
| `outerlink_transfer_bytes_total` | Counter | `node`, `direction` |

### Dashboard (Grafana)

Pre-built dashboards for:
- **Cluster overview:** Total GPUs, utilization, VRAM usage, active users
- **Per-user view:** Usage over time, quota consumption, job history
- **Per-node view:** GPU health, temperature, memory, network throughput
- **Queue view:** Pending jobs, wait times, scheduling decisions

---

## Recommended Implementation Order

| Step | Component | Depends On |
|---|---|---|
| 1 | Usage metering (kernel-level GPU-second tracking) | Core OuterLink (P6) |
| 2 | VRAM quota enforcement | Step 1 |
| 3 | Multi-user authentication (JWT) | Step 1 |
| 4 | Priority-based kernel admission control | Steps 1-2 |
| 5 | Fair-share scheduling | Steps 1-4 |
| 6 | Job submission API (gRPC) | Steps 1-5 |
| 7 | REST API + CLI | Step 6 |
| 8 | Kubernetes device plugin | Step 6 |
| 9 | KAI/Volcano integration | Step 8 |
| 10 | Web dashboard (Grafana) | Step 6 |

---

## Open Questions

1. **Standalone vs. Kubernetes-first:** Should R24 work standalone first and add K8s later, or build K8s-native from the start? Standalone is simpler and serves the LAN use case directly.
2. **Billing system scope:** Do we build billing into OuterLink or just export usage data and let external systems handle billing?
3. **Multi-cluster:** Should a single OuterLink control plane span multiple LANs (e.g., office + home lab)?
4. **Spot/preemption economics:** How do we implement spot instances? What's the preemption notice period?

---

## Related Documents

- [01-gpu-virtualization-landscape.md](01-gpu-virtualization-landscape.md)
- [02-scheduling-and-isolation.md](02-scheduling-and-isolation.md)
- [R17: Topology-Aware Scheduling](../../phase-09-distributed-os/R17-topology-aware-scheduling/)
- [R23: Heterogeneous GPU Mixing](../../phase-10-ecosystem/R23-heterogeneous-gpu-mixing/)
