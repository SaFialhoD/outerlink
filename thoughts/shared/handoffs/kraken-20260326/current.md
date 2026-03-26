## Checkpoints
<!-- Resumable state for kraken agent -->
**Task:** Implement R19: Network page fault handling (CoherencyDirectory, FaultHandler, ThrashingDetector)
**Started:** 2026-03-26T00:00:00Z
**Last Updated:** 2026-03-26T00:00:00Z

### Phase Status
- Phase 1 (Tests Written): VALIDATED (21 tests, 20 failing as expected)
- Phase 2 (Implementation): IN_PROGRESS
- Phase 3 (Refactoring): PENDING
- Phase 4 (Commit): PENDING

### Validation State
```json
{
  "test_count": 21,
  "tests_passing": 1,
  "tests_failing": 20,
  "files_modified": [
    "crates/outerlink-common/src/lib.rs",
    "crates/outerlink-common/src/memory/mod.rs",
    "crates/outerlink-common/src/memory/coherency.rs",
    "crates/outerlink-common/src/memory/fault_handler.rs",
    "crates/outerlink-common/src/memory/tests.rs"
  ],
  "last_test_command": "cargo test -p outerlink-common --lib -- memory:: --test-threads=1",
  "last_test_exit_code": 1
}
```

### Resume Context
- Current focus: Implementing CoherencyDirectory methods
- Next action: Replace todo!() with real logic in coherency.rs, fault_handler.rs
- Blockers: None
