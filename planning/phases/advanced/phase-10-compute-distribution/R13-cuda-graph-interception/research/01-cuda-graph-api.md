# R13 Research: CUDA Graph API Internals

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Complete
**Priority:** HIGH

## Purpose

Document the full CUDA Graph Driver API surface that OuterLink must intercept, including graph construction, inspection, modification, and execution APIs. This is the foundation for knowing exactly what hooks we need and what information we can extract from captured graphs.

---

## TL;DR

The CUDA Graph Driver API has ~80 functions spanning graph creation, node management, inspection, modification, instantiation, and execution. OuterLink must intercept all of them. The critical insight: CUDA provides rich inspection APIs (`cuGraphGetNodes`, `cuGraphNodeGetType`, `cuGraphNodeGetDependencies`, `cuGraphGetEdges`) that let us fully reconstruct the DAG at interception time. We can also clone and manipulate graphs (`cuGraphClone`, `cuGraphDestroyNode`, `cuGraphAddDependencies`) to rebuild them for distributed execution. `cuGraphExecUpdate` allows in-place parameter updates without re-instantiation, but cannot change topology.

---

## Two Ways to Build a CUDA Graph

### 1. Stream Capture (Implicit Construction)

The application calls `cuStreamBeginCapture` on a stream, then issues normal CUDA work (kernel launches, memcpys, etc.) to that stream. The driver records all operations into an internal graph instead of executing them. `cuStreamEndCapture` returns the completed `CUgraph`.

**Driver API functions:**

| Function | Purpose |
|----------|---------|
| `cuStreamBeginCapture(stream, mode)` | Put stream into capture mode |
| `cuStreamEndCapture(stream, *graph)` | End capture, return CUgraph |
| `cuStreamIsCapturing(stream, *status)` | Query if stream is in capture mode |
| `cuStreamGetCaptureInfo(stream, ...)` | Get capture sequence ID and graph |
| `cuStreamUpdateCaptureDependencies(...)` | Modify capture dependencies mid-capture |

**Capture modes:**

| Mode | Behavior |
|------|----------|
| `CU_STREAM_CAPTURE_MODE_GLOBAL` | Default. Blocks unsafe API calls in all threads during capture |
| `CU_STREAM_CAPTURE_MODE_THREAD_LOCAL` | Only blocks unsafe calls in the capturing thread |
| `CU_STREAM_CAPTURE_MODE_RELAXED` | No restrictions on concurrent API calls |

**Cross-stream capture:** When a captured stream records an event (`cuEventRecord`) and another stream waits on it (`cuStreamWaitEvent`), the second stream joins the same capture graph. This is how multi-stream applications naturally express parallelism that shows up as independent branches in the graph.

**OuterLink interception point:** We intercept `cuStreamBeginCapture` and `cuStreamEndCapture`. When `cuStreamEndCapture` returns, we have the complete `CUgraph` to analyze before the application calls `cuGraphInstantiate`.

### 2. Explicit Construction (Manual API)

The application builds the graph node-by-node using `cuGraphCreate` and the various `cuGraphAdd*Node` functions.

**OuterLink interception point:** We intercept `cuGraphCreate` and every `cuGraphAdd*Node` call, building our own shadow representation of the graph as the application constructs it.

### Which Method Do Real Frameworks Use?

| Framework | Method | Notes |
|-----------|--------|-------|
| PyTorch (`torch.cuda.CUDAGraph`) | Stream capture | Wraps `cudaStreamBeginCapture` / `cudaStreamEndCapture` |
| TensorFlow/XLA | Stream capture | Uses `cudaStreamBeginCapture` for fused subgraphs |
| TensorRT | Both | Explicit construction for optimized inference graphs |
| Custom HPC codes | Explicit | Direct graph API for known static workloads |

Stream capture is the dominant method in ML frameworks. This means `cuStreamBeginCapture` / `cuStreamEndCapture` is the primary interception path.

---

## Graph Node Types

Every node in a CUDA graph has a type (`CUgraphNodeType`). The full enumeration:

| Value | Enum | Description | OuterLink Relevance |
|-------|------|-------------|---------------------|
| 0 | `CU_GRAPH_NODE_TYPE_KERNEL` | GPU kernel launch | **PRIMARY** — the work we distribute |
| 1 | `CU_GRAPH_NODE_TYPE_MEMCPY` | Memory copy (H2D, D2H, D2D) | **CRITICAL** — data movement we must intercept/redirect |
| 2 | `CU_GRAPH_NODE_TYPE_MEMSET` | Memory set | Must track (initializes buffers) |
| 3 | `CU_GRAPH_NODE_TYPE_HOST` | CPU callback function | **BARRIER** — forces CPU synchronization, cannot distribute |
| 4 | `CU_GRAPH_NODE_TYPE_GRAPH` | Child graph (embedded subgraph) | Recursive — must analyze child graphs too |
| 5 | `CU_GRAPH_NODE_TYPE_EMPTY` | No-op (dependency-only) | Useful as synchronization barriers |
| 6 | `CU_GRAPH_NODE_TYPE_WAIT_EVENT` | Wait on external event | Cross-stream synchronization |
| 7 | `CU_GRAPH_NODE_TYPE_EVENT_RECORD` | Record external event | Cross-stream synchronization |
| 8 | `CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL` | Signal external semaphore | Inter-process synchronization |
| 9 | `CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT` | Wait on external semaphore | Inter-process synchronization |
| 10 | `CU_GRAPH_NODE_TYPE_MEM_ALLOC` | Allocate memory | Must track — affects memory map |
| 11 | `CU_GRAPH_NODE_TYPE_MEM_FREE` | Free memory | Must track — affects memory map |
| 12 | `CU_GRAPH_NODE_TYPE_BATCH_MEM_OP` | Batch memory operations | Optimization hint |
| 13 | `CU_GRAPH_NODE_TYPE_CONDITIONAL` | Conditional execution (CUDA 12.3+) | **COMPLEX** — dynamic control flow within graph |

### Conditional Node Subtypes (CUDA 12.3+)

| Type | Introduced | Behavior |
|------|-----------|----------|
| IF | CUDA 12.3 | Execute body if condition non-zero |
| WHILE | CUDA 12.3 | Loop body while condition non-zero |
| IF/ELSE | CUDA 12.8 | Two bodies: condition true vs false |
| SWITCH | CUDA 12.8 | N bodies, execute body[condition_value] |

Conditional nodes use a handle (`cuGraphConditionalHandleCreate`) set by a kernel calling `cudaGraphSetConditional`. The graph topology is static but execution path is dynamic. This is critical for OuterLink: the bodies of conditional nodes must be analyzed as potential execution paths, but only one runs per iteration.

**Constraints on conditional node bodies:** Only kernel nodes, empty nodes, child graphs, memsets, memcpies, and nested conditionals are allowed. No dynamic parallelism or device-side graph launch inside conditional bodies.

---

## Node Creation Functions (Full Driver API)

These are all functions OuterLink must intercept for explicit graph construction:

| Function | Node Type Created |
|----------|-------------------|
| `cuGraphAddKernelNode` | Kernel |
| `cuGraphAddMemcpyNode` | Memcpy |
| `cuGraphAddMemsetNode` | Memset |
| `cuGraphAddHostNode` | Host callback |
| `cuGraphAddChildGraphNode` | Child graph |
| `cuGraphAddEmptyNode` | Empty/no-op |
| `cuGraphAddEventRecordNode` | Event record |
| `cuGraphAddEventWaitNode` | Event wait |
| `cuGraphAddExternalSemaphoresSignalNode` | Ext semaphore signal |
| `cuGraphAddExternalSemaphoresWaitNode` | Ext semaphore wait |
| `cuGraphAddMemAllocNode` | Memory allocation |
| `cuGraphAddMemFreeNode` | Memory free |
| `cuGraphAddBatchMemOpNode` | Batch memory ops |
| `cuGraphAddNode` | Generic (CUDA 12.2+, type specified via params) |

Each creation function takes: graph handle, dependency array, dependency count, and type-specific parameters.

---

## Graph Inspection APIs

These APIs let OuterLink fully reconstruct and analyze the DAG:

| Function | Returns |
|----------|---------|
| `cuGraphGetNodes(graph, nodes[], *count)` | All nodes in the graph |
| `cuGraphGetRootNodes(graph, roots[], *count)` | Root nodes (no dependencies) |
| `cuGraphGetEdges(graph, from[], to[], edgeData[], *count)` | All dependency edges |
| `cuGraphNodeGetType(node, *type)` | Node type enum |
| `cuGraphNodeGetDependencies(node, deps[], *count)` | Node's predecessors |
| `cuGraphNodeGetDependentNodes(node, deps[], edgeData[], *count)` | Node's successors |
| `cuGraphNodeGetLocalId(node, *id)` | Unique ID within graph |
| `cuGraphNodeGetContainingGraph(node, *graph)` | Which graph owns this node |

**Type-specific parameter queries:**

| Function | Node Type | Returns |
|----------|-----------|---------|
| `cuGraphKernelNodeGetParams(node, *params)` | Kernel | Function pointer, grid/block dims, shared mem, args |
| `cuGraphMemcpyNodeGetParams(node, *params)` | Memcpy | Src, dst, size, direction |
| `cuGraphMemsetNodeGetParams(node, *params)` | Memset | Ptr, value, size |
| `cuGraphHostNodeGetParams(node, *params)` | Host | Function pointer, user data |
| `cuGraphChildGraphNodeGetGraph(node, *graph)` | Child | The embedded CUgraph |
| `cuGraphEventRecordNodeGetEvent(node, *event)` | Event record | CUevent |
| `cuGraphEventWaitNodeGetEvent(node, *event)` | Event wait | CUevent |

**Debug output:**
`cuGraphDebugDotPrint(graph, path, flags)` writes a DOT/Graphviz file of the graph structure. Includes topology, node types, node IDs, kernel names, memcpy directions. Useful for debugging OuterLink's graph analysis.

---

## Graph Manipulation APIs

These APIs let OuterLink modify a graph before instantiation:

| Function | Purpose |
|----------|---------|
| `cuGraphClone(*clone, original)` | Deep copy a graph |
| `cuGraphDestroyNode(node)` | Remove a node and its edges |
| `cuGraphAddDependencies(graph, from[], to[], edgeData[], count)` | Add edges |
| `cuGraphRemoveDependencies(graph, from[], to[], edgeData[], count)` | Remove edges |
| `cuGraphNodeFindInClone(*clonedNode, originalNode, clonedGraph)` | Map original node to its clone |

**Key capability:** We can clone a graph, remove nodes assigned to other GPUs, add communication nodes (memcpy or host callbacks for RDMA triggers), and re-wire dependencies. This means we can build per-GPU subgraphs from the original.

**Thread safety warning:** Graph objects are NOT threadsafe. All manipulation must be serialized.

---

## Graph Instantiation and Execution

| Function | Purpose |
|----------|---------|
| `cuGraphInstantiate(*exec, graph, flags)` | Compile graph into executable |
| `cuGraphInstantiateWithParams(*exec, graph, *params)` | Instantiate with options |
| `cuGraphLaunch(exec, stream)` | Execute the graph |
| `cuGraphExecDestroy(exec)` | Free executable |
| `cuGraphDestroy(graph)` | Free graph template |

**Instantiation is expensive:** It validates the graph, allocates internal resources, and plans execution order. This is the "compile" step. `cuGraphLaunch` (the "run" step) is very fast because all setup is done.

**OuterLink's key interception point:** Between `cuStreamEndCapture` (or manual graph construction) and `cuGraphInstantiate`. This is where we analyze the graph, plan distribution, build per-GPU subgraphs, and instantiate them on the correct GPUs.

---

## Graph Update APIs (Modify Without Re-instantiation)

| Function | Purpose |
|----------|---------|
| `cuGraphExecUpdate(exec, graph, *resultInfo)` | Update all parameters from a topologically-identical graph |
| `cuGraphExecNodeSetParams(exec, node, *params)` | Update one node's parameters |
| `cuGraphExecKernelNodeSetParams(exec, node, *params)` | Update kernel parameters |
| `cuGraphExecMemcpyNodeSetParams(exec, node, *params)` | Update memcpy parameters |
| `cuGraphExecMemsetNodeSetParams(exec, node, *params)` | Update memset parameters |
| `cuGraphExecHostNodeSetParams(exec, node, *params)` | Update host node parameters |
| `cuGraphExecChildGraphNodeSetParams(exec, node, child)` | Replace child graph |
| `cuGraphNodeSetEnabled(exec, node, enabled)` | Enable/disable a node |
| `cuGraphNodeGetEnabled(exec, node, *enabled)` | Query if node is enabled |

**What can be updated:**
- Kernel arguments, grid/block dimensions (same function, same context)
- Memcpy src/dst pointers, sizes (same allocation contexts)
- Memset parameters (1D only)
- Host callback function pointer and user data

**What CANNOT be updated:**
- Graph topology (adding/removing nodes or edges)
- CUDA context of a kernel function
- Switching a kernel from non-CDP to CDP
- Source/destination memory allocation context for memcpy

**OuterLink implication:** After initial graph analysis and distributed instantiation, if the application calls `cuGraphExecUpdate`, we can potentially apply the same parameter changes to our distributed subgraph executables without re-analyzing the topology. This is a fast path for iterative workloads (training loops) where the same graph is reused with updated tensor pointers.

---

## CUDA 12+ Advanced Features

### Device-Side Graph Launch (CUDA 12.0+)

A GPU kernel can launch a pre-built graph from device code, without CPU involvement. Reduces latency vs host launch by over 2x.

**OuterLink impact:** If an application uses device-side launch, the launch happens inside a running kernel on the GPU. OuterLink cannot intercept device-side launches through LD_PRELOAD since they bypass the Driver API entirely. We would need to detect graphs that are set up for device-side launch (the `CU_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH` flag) and handle them specially.

### Device-Side Node Parameter Updates (CUDA 12.4+)

Kernel nodes can opt in to being device-updatable. GPU code can then modify kernel parameters without CPU round-trips.

**OuterLink impact:** Same as device-side launch. These updates happen on-device and cannot be intercepted via the Driver API. If a graph uses device-updatable nodes, our parameter tracking may become stale.

### Programmatic Dependent Launch (CUDA 12.3+)

Edge data type `CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC` enables overlap between a dependent node and its predecessor's tail. The dependent node's grid can start executing while the predecessor is still running.

**OuterLink impact:** This affects scheduling. If we split dependent nodes across GPUs, the programmatic overlap is lost. We should keep programmatically-dependent node pairs on the same GPU when possible.

---

## Complete Driver API Function Count

Categorized count of CUDA Graph Driver API functions OuterLink must intercept:

| Category | Count | Examples |
|----------|-------|----------|
| Graph lifecycle | 4 | Create, Destroy, Clone, DebugDotPrint |
| Node creation | 14 | AddKernelNode, AddMemcpyNode, AddNode, ... |
| Node parameter get/set | ~24 | KernelNodeGetParams, MemcpyNodeSetParams, ... |
| Graph inspection | 10 | GetNodes, GetEdges, GetRootNodes, NodeGetType, ... |
| Graph manipulation | 4 | AddDependencies, RemoveDependencies, DestroyNode, NodeFindInClone |
| Instantiation | 3 | Instantiate, InstantiateWithParams, InstantiateWithFlags |
| Execution | 2 | Launch, ExecDestroy |
| Update | 8 | ExecUpdate, ExecNodeSetParams, ExecKernelNodeSetParams, ... |
| Enable/disable | 2 | NodeSetEnabled, NodeGetEnabled |
| Stream capture | 5 | StreamBeginCapture, StreamEndCapture, StreamIsCapturing, ... |
| Conditional | 2 | ConditionalHandleCreate, AddNode (conditional type) |
| **Total** | **~78** | |

---

## Interception Strategy for OuterLink

### Phase 1: Transparent Pass-Through
Intercept all ~78 graph functions. Forward to real driver. Build shadow graph representation. Log and validate.

### Phase 2: Graph Analysis
On `cuStreamEndCapture` or `cuGraphInstantiate`, analyze the shadow graph:
1. Extract DAG topology (nodes, edges, types)
2. Identify kernel nodes and their memory access patterns
3. Detect parallelism (independent branches)
4. Estimate per-node execution cost
5. Identify data dependencies via memcpy nodes

### Phase 3: Distributed Execution
1. Clone the graph
2. Partition nodes across GPUs (see 02-graph-analysis-and-splitting.md)
3. Insert communication nodes at partition boundaries
4. Build per-GPU subgraphs
5. Instantiate on respective GPUs
6. Coordinate launch

---

## Open Questions

1. **Kernel argument inspection:** `cuGraphKernelNodeGetParams` gives us the kernel function pointer and argument buffer, but not the semantics of arguments (which are pointers to GPU memory vs scalars). We need R8 (Kernel Param Introspection) to resolve this.

2. **Stream capture of NCCL calls:** If an application captures NCCL collective operations within a graph (PyTorch does this), those nodes appear as kernel launches. We need to identify NCCL kernels specifically to handle them via R20 (NCCL Backend).

3. **Graph re-capture frequency:** ML training loops often re-capture graphs every N iterations (e.g., when shapes change). How much analysis overhead is acceptable per capture? Target: under 1ms for graphs with < 1000 nodes.

4. **Device-side features escape hatch:** Device-side launch and device-side parameter updates bypass our interception entirely. What percentage of real workloads use these features? Likely very low as of 2026, but growing.

---

## Related Documents

- R3: CUDA Interception Strategies (general interception architecture)
- R8: Kernel Parameter Introspection (understanding kernel arguments)
- R10: Memory Hierarchy (memory tracking for graph data dependencies)
- R20: NCCL Backend (handling collective operations in graphs)
- 02-graph-analysis-and-splitting.md (using this API knowledge for distribution)

## References

- NVIDIA CUDA Driver API — Graph Management: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html
- CUDA Programming Guide — CUDA Graphs: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html
- NVIDIA Blog — Dynamic Control Flow with Conditional Nodes: https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes/
- NVIDIA Blog — Device Graph Launch: https://developer.nvidia.com/blog/enabling-dynamic-control-flow-in-cuda-graphs-with-device-graph-launch/
- NVIDIA Blog — Constructing CUDA Graphs with Dynamic Parameters: https://developer.nvidia.com/blog/constructing-cuda-graphs-with-dynamic-parameters/
- NVIDIA Blog — Employing CUDA Graphs in a Dynamic Environment: https://developer.nvidia.com/blog/employing-cuda-graphs-in-a-dynamic-environment/
