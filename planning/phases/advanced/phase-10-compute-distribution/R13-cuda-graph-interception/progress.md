# R13: CUDA Graph Interception — Progress

| Date | Status | Summary |
|------|--------|---------|
| 2026-03-25 | NOT STARTED | Folder created, awaiting research phase |
| 2026-03-25 | RESEARCH COMPLETE | Completed 3 research documents and preplan |

## Research Phase Summary

Three research documents completed covering:

1. **01-cuda-graph-api.md** — Full CUDA Graph Driver API surface (~78 functions). Covers graph creation (stream capture + explicit construction), all 14 node types (including CUDA 12.3+ conditional nodes), inspection APIs for DAG reconstruction, manipulation APIs for graph modification, update APIs for parameter changes without re-instantiation, and CUDA 12+ advanced features (device-side launch, programmatic dependent launch).

2. **02-graph-analysis-and-splitting.md** — DAG analysis algorithms (topological sort, critical path, parallelism detection), HEFT partitioning algorithm for heterogeneous GPU assignment, data dependency tracking through memory regions, three splitting strategies (per-GPU subgraphs recommended), communication insertion at partition boundaries, and how PyTorch/TensorFlow/TensorRT build CUDA graphs in practice.

3. **03-distributed-graph-execution.md** — Coordinator-executor architecture, three synchronization mechanisms (RDMA signals, host-staged, CUDA IPC), compute-transfer overlap strategies, dynamic shape handling with partition plan caching, performance comparison (graph mode vs eager mode), integration with R11 prefetching (graph provides perfect access prediction) and R20 NCCL (collective operations in graphs).

## Key Findings

- Stream capture (`cuStreamBeginCapture` / `cuStreamEndCapture`) is the dominant graph construction method in ML frameworks — this is the primary interception path.
- CUDA provides rich inspection APIs that give us full DAG access at interception time.
- We can clone and manipulate graphs (`cuGraphClone`, `cuGraphDestroyNode`, `cuGraphAddDependencies`) to build per-GPU subgraphs.
- `cuGraphExecUpdate` enables fast parameter updates without topology re-analysis — critical for training loop efficiency.
- The Mustard paper (ICS 2025) validates that CUDA Graph partitioning across multiple GPUs works in practice, using a device-side execution model.
- Graph mode eliminates per-kernel decision overhead (10-50us per call) by pre-planning the entire execution, providing 100x fewer coordinator messages.

## Preplan Status

Preplan created with 5 key technical unknowns identified, dependency map, risk assessment, and 7 implementation milestones defined. Ready for detailed plan creation.

## Next Steps

- [ ] Get approval on preplan approach
- [ ] Decide: single plan or sub-plans (R13a/R13b/R13c)?
- [ ] Optional: prototype PyTorch graph capture + shadow graph validation
- [ ] Write detailed implementation plan
