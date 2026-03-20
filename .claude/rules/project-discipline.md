# Project Discipline Rules

## Planning Before Doing

1. **Pre-plan before planning** - For any major component, first document WHAT needs to be planned (scope, unknowns, dependencies). Only then create the actual plan.
2. **No code without a plan** - Every implementation must trace back to an approved plan document.
3. **Plans are living documents** - Update them as reality changes. A stale plan is worse than no plan.

## Decision Making

1. **Document every technical decision** - Use ADRs in `/docs/decisions/`. No "we just decided in a conversation" - write it down.
2. **Research before deciding** - Every non-trivial technical choice must have a research doc in `/planning/research/` BEFORE the decision is made.
3. **Reversibility matters** - Prefer reversible decisions. When making irreversible ones, get explicit confirmation.

## Progress Tracking

1. **Phase gates** - Each planning phase has clear deliverables. Don't move forward until they're met.
2. **Blockers are documented immediately** - If something blocks progress, it gets written into the relevant plan with a proposed resolution path.
3. **No silent failures** - If something doesn't work, document WHY and what we tried.

## This Is a Research-Heavy Project

1. **Expect unknowns** - This project pushes boundaries. Not everything will work on first try.
2. **Prototype before committing** - For risky technical approaches, build a minimal proof of concept before designing the full system.
3. **Measure everything** - Bandwidth, latency, memory overhead. Claims without numbers are opinions, not facts.
