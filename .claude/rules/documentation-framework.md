# Documentation Framework Rules

**Every piece of knowledge in this project MUST be documented. If it's not written down, it doesn't exist.**

## Document Structure

Every document MUST have:
1. **Header** with title, date created, date last updated, status (draft/review/approved/archived)
2. **Purpose** - one sentence explaining why this document exists
3. **Content** - the actual information
4. **Related Documents** - links to other relevant docs
5. **Open Questions** - things still unresolved (never delete these, mark them as resolved with answers)

## Folder Rules

### `/planning/pre-planning/`
- Documents that define WHAT we need to plan
- Scope definition, risk identification, dependency mapping
- These come BEFORE detailed plans

### `/planning/phases/`
- Detailed phase plans with milestones, deliverables, and acceptance criteria
- Each phase gets its own document
- Phases are sequential - phase N+1 cannot start until phase N deliverables are met

### `/planning/research/`
- Research findings on existing tools, libraries, protocols
- Each research topic gets its own document
- Must include: what it is, how it works, pros/cons for our use case, verdict

### `/docs/architecture/`
- System design documents
- Component diagrams and data flow
- Technical decisions and their rationale

### `/docs/guides/`
- How-to documents for setup, configuration, development
- Step-by-step, reproducible instructions

### `/docs/specs/`
- Technical specifications
- API definitions, protocol descriptions, data formats

### `/docs/decisions/`
- Architecture Decision Records (ADRs)
- Format: context, decision, consequences
- Decisions are NEVER deleted, only superseded by newer decisions

### `/side-docs/references/`
- External links, papers, articles we reference
- Each with a summary of why it's relevant

### `/side-docs/notes/`
- Working notes, brainstorm outputs, session summaries
- Less formal but still organized

### `/side-docs/diagrams/`
- All visual diagrams (ASCII, mermaid, or image files)
- Each diagram must have a companion .md explaining what it shows

## Writing Rules

1. **No orphan knowledge** - If you learn something during development, it goes in a doc
2. **Update, don't duplicate** - If info exists, update the existing doc. Don't create a second one.
3. **Date everything** - Every doc tracks creation and update dates
4. **Link aggressively** - Cross-reference related documents
5. **Questions are valuable** - Open questions in docs are features, not bugs. They show what we still need to figure out.
6. **Tables over paragraphs** - When comparing options or listing specs, use tables
7. **Code examples are docs** - Any config, command, or snippet in docs must be tested and working
