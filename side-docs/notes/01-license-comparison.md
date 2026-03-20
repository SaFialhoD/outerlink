# License Comparison for OutterLink

**Created:** 2026-03-19
**Status:** Needs Decision

## Purpose

Choose the right open source license for OutterLink.

---

## The Three Real Options

### Option 1: Apache 2.0 (RECOMMENDED)

**What it means:** Anyone can use, modify, sell, and distribute OutterLink freely. Companies can use it in proprietary products without releasing their changes.

| Pros | Cons |
|------|------|
| Maximum adoption - companies won't hesitate to use it | Someone could fork it, make it better, and sell it closed-source |
| Compatible with almost every other license | No "protection" against commercial exploitation |
| Used by: HAMi, Kubernetes, TensorFlow, most CNCF projects | |
| Companies can contribute back without legal friction | |
| Patent protection clause (important for NVIDIA-adjacent tech) | |

**Best for:** Building a standard that everyone adopts. Growing a community fast.

### Option 2: GPL v3

**What it means:** Anyone who modifies and distributes OutterLink MUST also release their changes as open source under GPL. "Viral" - if you link to GPL code, your code becomes GPL too.

| Pros | Cons |
|------|------|
| Forces all improvements to stay open source | Companies avoid GPL like the plague |
| Prevents proprietary forks | Incompatible with Apache 2.0 code (can't mix) |
| Used by: Cricket, gVirtuS, Linux kernel | Slows adoption in enterprise/commercial |
| Ideological protection of open source | Can't easily use in proprietary ML pipelines |

**Best for:** Ensuring code stays free forever. Ideological commitment.

### Option 3: MIT

**What it means:** Do whatever you want with it. Minimal restrictions, minimal protection.

| Pros | Cons |
|------|------|
| Simplest license | No patent protection (risky for GPU tech) |
| Maximum freedom | Someone could patent improvements and sue you |
| Used by: lots of small projects | Less protection than Apache 2.0 with no real upside |

**Best for:** Small utilities. NOT recommended for a project touching NVIDIA patents.

---

## Why Apache 2.0 Is Probably Right for OutterLink

1. **Patent protection** - Apache 2.0 includes an explicit patent grant. In GPU/CUDA territory where NVIDIA holds many patents, this matters. MIT does NOT have this.

2. **Adoption** - If OutterLink works well, we want companies and researchers to use it freely. GPL scares them away. Apache 2.0 doesn't.

3. **Ecosystem compatibility** - HAMi (whose interception code we'll study) is Apache 2.0. Kubernetes ecosystem is Apache 2.0. Being compatible makes integration easier.

4. **Contributors** - Companies are more likely to contribute to Apache 2.0 projects because they can also use the code internally without legal risk.

5. **The "someone will steal it" fear** - Yes, a company could fork and close-source it. But in practice, maintaining a fork is expensive, and the open version with community support always wins long-term (see: Linux, Kubernetes, PostgreSQL philosophy).

---

## License Compatibility with Dependencies

| Dependency | License | Compatible with Apache 2.0? | Compatible with GPL? |
|-----------|---------|---------------------------|---------------------|
| HAMi-core (study, not fork) | Apache 2.0 | YES | YES (one-way) |
| Cricket (study, not fork) | GPLv3 | Careful - can study but not copy code | YES |
| gVirtuS (study, not fork) | GPL | Careful - can study but not copy code | YES |
| CUDA toolkit | Proprietary | YES (we link dynamically at runtime) | YES (dynamic linking) |
| Rust standard library | Apache 2.0 + MIT | YES | YES |

**Note:** We're studying these projects for patterns, not forking them. Clean-room implementation in Rust avoids license contamination. But if we wanted to directly fork Cricket or gVirtuS, we'd be locked into GPL.

## Recommendation

**Go Apache 2.0.** Maximum adoption, patent protection, ecosystem compatibility. We're building this to be used, not to make an ideological statement.

## Open Questions

- [ ] Does Pedro have a preference for open source philosophy?
- [ ] Do we want to allow commercial use without giving back?
