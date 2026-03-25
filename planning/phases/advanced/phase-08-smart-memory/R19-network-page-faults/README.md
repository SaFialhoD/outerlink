# R19: Network Page Faults / Unified Memory

**Phase:** 8 — Smart Memory
**Status:** NOT STARTED
**Priority:** HIGH
**Depends On:** R10 (Memory Tiering)

## Summary

Implement transparent remote memory access via GPU page faults. When a GPU accesses a memory address that lives on a remote node, the page fault triggers an automatic fetch — the page is transferred, installed locally, and the GPU resumes. Applications use plain pointers, no explicit cudaMemcpy calls needed.

## What This Enables

- True unified address space across the cluster
- CUDA apps "just work" without any memory management code
- Combined with prefetching (R11), most faults are avoided entirely
- Simplest possible programming model: one flat memory space

## Key Questions

- Mechanism: userfaultfd, custom kernel handler, or CUDA UVM hooks?
- Page fault latency budget? (network RTT = minimum ~2μs for RDMA)
- How to handle thrashing (two nodes repeatedly faulting same page)?
- CUDA's own UVM (Unified Virtual Memory) — cooperate or replace?

## Folder Contents

- `research/` — Page fault mechanisms, UVM internals, userfaultfd
- `side-docs/` — Notes, prototypes
- `preplan.md` — TO BE CREATED
- `plan.md` — TO BE CREATED
- `progress.md` — Lifecycle tracker

## Related Topics

- R10 Memory Tiering (page faults trigger tier migration)
- R11 Speculative Prefetching (prefetch prevents faults)
- R18 Virtual NVLink (page faults are part of coherency)
