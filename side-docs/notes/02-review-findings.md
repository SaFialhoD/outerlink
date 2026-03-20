# Review Findings - 2026-03-20

**Status:** Issues identified, fixes pending

## Summary of All Reviews

4 review agents ran on 2026-03-20. Combined findings across research docs, code, plans, and architecture:

### Research Docs Review - 3 Critical (FIXED)
1. ~~R6 purpose says 600 GB/s, should be 112.5 GB/s~~ FIXED
2. ~~R4 bandwidth table says NVLink 600 GB/s~~ FIXED
3. ~~SYNTHESIS is stale (written before R4-R7)~~ FIXED (marked superseded)

### Rust Code Review - 5 Critical (FIXED)
1. ~~Race condition in HandleMap::insert~~ FIXED (DashMap entry API)
2. ~~bincode pre-release dependency~~ FIXED (switched to stable 1.x)
3. ~~MessageHeader version not validated~~ FIXED
4. ~~Payload size not validated against MAX_PAYLOAD_SIZE~~ FIXED
5. ~~HandleMap::new private but HandleStore fields pub~~ FIXED

### Plan Review - 7 Critical (TO ADDRESS)
1. **P5 vs P6 wire format incompatible** - P5 uses 10-byte header, P6 uses 22-byte. Decision: use P6 format from start.
2. **P4 says bincode, P6 says custom binary** - contradiction. Decision: P6 wins (custom binary), P4 needs update.
3. **P5 open questions are actually blockers** - cuGetErrorName/String must be intercepted, cuGetProcAddress_v2 handling must be resolved.
4. **P8 batch handle prediction breaks on allocation failure** - if cuMemAlloc fails mid-batch, subsequent calls use wrong handles. Fix: don't batch handle-returning calls.
5. **P9 references non-existent R12 research** - tinygrad compatibility research not written yet.
6. **P6 ModuleGetFunction payload offset wrong** - off-by-one at offset 4 should be 8.
7. **P3 vs P13 doc CI flags disagree** - remove --document-private-items from CI.

### Plan Review - Key Suggestions
- **Port standardization:** P5 uses 9370, P10/P13 use 9700. Pick one.
- **P5 demo script preloads wrong .so** - should preload C interpose.so, not Rust cdylib
- **P11 mock server uses P5 protocol, will break at P6**
- **P10 GPU discovery functions undefined**
- **opendma/ directory location inconsistent** between P3 and P4

### Rust Code Review - Key Suggestions (For Implementation Phase)
- CUDA handle types should be newtypes not type aliases (compile-time safety)
- CuResult::Unknown should preserve raw value as CuResult::Other(u32)
- Transport trait &self may prevent exclusive-access RDMA transports
- Single Response message type is too opaque - consider typed responses
- flags field in header needs documented semantics

## Priority Order for Fixes

### Must Fix Before Implementation
1. Standardize port number across all plans
2. Decide: use P6 wire format from P5 onward (avoid throwaway code)
3. Remove handle prediction from P8 batching design
4. Create R12 stub or remove P9's dependency on it
5. Fix P6 payload offset

### Nice to Have Before Implementation
6. Update P11 mock server to note P6 protocol transition
7. Standardize opendma/ directory location
8. Add cuGetErrorName/String to P5 function list
9. Resolve cuGetProcAddress_v2 variant handling in P5

### Architecture Review - 3 Critical (FIXED)
1. ~~Serialization contradiction (P4 bincode vs P6 custom)~~ Documented - P6 wins
2. ~~P9 references non-existent ADR-003~~ Noted for creation during P9
3. ~~Vision doc: NVLink 600 GB/s wrong, stale open questions~~ FIXED

### Architecture Review - Key Suggestions
- Confirm project name spelling: repo is "outerlink" (one T), docs say "OutterLink" (two T)
- MS-01 Ultra CPU/PCIe lane budget unknown - could affect bandwidth thesis
- P3 (CI/CD) missing from "next steps" sequence in pre-plan
- No contingency plan for protocol/serialization format issues

### Implementation Review Round 2 (2026-03-20) - Code Reviews

**TCP Transport - 3 Critical:**
1. `recv_message` error loses msg_type diagnostic value
2. `set_nodelay` error mapped inconsistently + doc claims buffer tuning that doesn't exist
3. `recv_bulk` has no upper bound on size - OOM DoS from malicious peer

**Server - 4 Critical:**
1. Handshake message type never handled (returns error)
2. TOCTOU race in mem_alloc (concurrent allocs can exceed VRAM limit)
3. main.rs bypasses TcpTransportConnection (no TCP_NODELAY, duplicated logic)
4. MemcpyHtoD with zero bytes has ambiguous behavior

**Client FFI - 5 Critical:**
1. ol_cuDeviceGetName null termination bug on exact-length buffers
2. ol_cuCtxCreate inserts wrong remote value (deduplication collision)
3. dlsym override -> ensure_init -> ol_client_init may trigger recursive dlsym (latent)
4. cuGetProcAddress hook skips version negotiation with real driver
5. OuterLinkClient.connected is plain bool on shared reference (needs AtomicBool)

**Integration Readiness - 5 Critical:**
1. OuterLinkClient has no transport connection (just a struct with no networking)
2. No sync/async bridge (CUDA API is sync, tokio is async, no block_on anywhere)
3. No serialization code in any FFI function (all stubs return hardcoded values)
4. Handle translation wired backwards (inserts fake remote values, not real server handles)
5. Server main.rs doesn't use TcpTransportConnection

**Integration: 9-Step Roadmap to First E2E Call:**
1. Add tokio runtime + TCP connection to OuterLinkClient
2. Add send_request() helper method
3. Wire ol_cuDeviceGetCount (FIRST REAL NETWORKED CALL)
4. Wire remaining device queries
5. Wire CtxCreate with real handle translation
6. Extend server handler for context operations
7. Wire MemAlloc/MemFree with handle translation
8. Wire MemcpyHtoD/DtoH (actual GPU data over network)
9. Migrate server main.rs to TcpTransportConnection

### For Implementation Phase (Not Plan Fixes)
10. Convert CUDA handle types to newtypes
11. Add CuResult::Other(u32) for unknown error codes
12. Add typed response messages or embed msg_type in response
13. Document flags field semantics
14. Add more unit tests per code review suggestions
