# ADR-002: Non-Proprietary GPU DMA Feature Name

**Date:** 2026-03-19
**Status:** Accepted
**Deciders:** Pedro

## Context

OutterLink's killer feature is non-proprietary direct NIC-to-GPU VRAM access via PCIe BAR1, bypassing NVIDIA's artificial GPUDirect RDMA restriction. This needs a clear, memorable name.

Options considered: DirectLink, OpenDMA, FreeGPU Direct, BareLink, PeerDirect

## Decision

**OpenDMA**

## Consequences

- Clearly communicates "open" (non-proprietary) + "DMA" (direct memory access)
- Contrasts with NVIDIA's proprietary GPUDirect
- Short, memorable, easy to say
- Will be used in all documentation, architecture diagrams, and marketing
- Feature scope: direct PCIe BAR1 RDMA between any RDMA-capable NIC and any NVIDIA GPU VRAM
