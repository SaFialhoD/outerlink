# ADR-001: Project License

**Date:** 2026-03-19
**Status:** Accepted
**Deciders:** Pedro

## Context

OutterLink needs an open source license. Key considerations:
- Working in NVIDIA GPU/CUDA space where patents are relevant
- Want maximum community adoption
- Studying HAMi-core (Apache 2.0), Cricket (GPLv3), gVirtuS (GPL)
- Building clean-room in Rust, not forking any existing project
- Pedro: "I don't think I would make much money from this"

## Decision

**Apache 2.0**

## Consequences

- Anyone can use, modify, distribute freely (including commercial)
- Explicit patent grant protects against GPU/CUDA-adjacent patent claims
- Compatible with HAMi-core (Apache 2.0) patterns we're studying
- Companies can adopt without legal friction
- No obligation for users to share modifications (unlike GPL)
- Risk: someone could fork and close-source, but maintaining a fork is expensive and community wins long-term
