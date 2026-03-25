# R30: Persistent Kernels — Progress

| Date | Status | Summary |
|------|--------|---------|
| 2026-03-25 | NOT STARTED | Folder created, awaiting research phase |
| 2026-03-25 | RESEARCH COMPLETE | Completed 3 research documents: persistent kernel patterns (cooperative groups, occupancy, TDR, power), doorbell mechanisms (ring buffers, atomic counters, OpenDMA integration, cache coherency), network-fed execution (pipeline design, buffering, error handling, performance analysis). Pre-plan drafted with 3 implementation phases (A: standalone, B: OpenDMA-fed, C: full integration). 5 key unknowns identified, most critical being GPU cache coherency with NIC VRAM writes via BAR1. |
