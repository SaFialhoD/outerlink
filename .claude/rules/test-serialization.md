# Test & Build Awareness on Dev Machine

**Multiple Claude sessions may be running Rust builds simultaneously on this machine. Be aware of system load.**

## Rules

1. **Check system load before heavy operations** — other sessions may already be compiling:
   ```
   wmic cpu get loadpercentage
   ```
   If above 70%, wait or ask the user before starting builds/tests.

2. **Prefer `--test-threads=1` for large test suites** — especially outerlink-server which has 150+ tests with TCP listeners.

3. **Agents use `cargo check -p <crate>`** — lighter than full builds. Full test runs are preferred in the main conversation where we can monitor load.

4. **One heavy operation at a time** — avoid spawning multiple agents that all run `cargo test --workspace` simultaneously.

## Why

This PC runs multiple Claude sessions and other workloads. Uncoordinated parallel compilation across sessions has caused system freezes. The fix is awareness and moderation, not banning parallelism.
