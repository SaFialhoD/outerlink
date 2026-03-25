#!/bin/bash
# Progressive real-hardware GPU test runner for OuterLink.
#
# Runs tests in phases. Each phase must pass before the next runs.
# Stop on first failure (set -e).
#
# Usage:
#   ./scripts/test-real-hardware.sh          # Run all phases
#   ./scripts/test-real-hardware.sh phase1   # Run only phase 1
#   ./scripts/test-real-hardware.sh phase2   # Run only phase 2
#   ./scripts/test-real-hardware.sh phase3   # Run only phase 3
#   ./scripts/test-real-hardware.sh phase4   # Run only phase 4

set -e

FILTER="${1:-}"
TEST_CMD="cargo test -p outerlink-server --features real-gpu-test --test real_hardware_progressive"
TEST_FLAGS="-- --test-threads=1 --nocapture"

run_phase() {
    local phase="$1"
    local desc="$2"
    echo ""
    echo "================================================================"
    echo "  Phase $phase: $desc"
    echo "================================================================"
    echo ""
    $TEST_CMD "${phase}_" $TEST_FLAGS
    echo ""
    echo "  Phase $phase: PASSED"
    echo ""
}

if [ -z "$FILTER" ] || [ "$FILTER" = "phase1" ]; then
    run_phase "phase1" "Sanity - Can we talk to the GPU?"
fi

if [ -z "$FILTER" ] || [ "$FILTER" = "phase2" ]; then
    run_phase "phase2" "Functional - Does every operation work?"
fi

if [ -z "$FILTER" ] || [ "$FILTER" = "phase3" ]; then
    run_phase "phase3" "Reliability - Does it work 30 times in a row?"
fi

if [ -z "$FILTER" ] || [ "$FILTER" = "phase4" ]; then
    run_phase "phase4" "Stress & Load - Can it handle pressure?"
fi

echo ""
echo "================================================================"
echo "  ALL PHASES PASSED"
echo "================================================================"
