# Provenance & Review Status

What this bundle is derived from, and what has actually been verified versus merely designed.

> **striatica copy note (2026-08-01):** imported verbatim from
> `parallax-drift-mvp/docs/!PM/devcontainer-workflow-bundle/agent-swarm-bundle/`. The provenance
> below refers to the Parallax Drift repo and its reviews.

> **Note on paths below.** The cited source files live in a **private repo that does not ship with
> this bundle.** They are recorded for provenance and audit, not as links you can follow. Every
> conclusion drawn from them is restated in full here, so nothing in this bundle depends on reaching
> them.

## Source material

This bundle is a de-coupled repackaging of three things from the Parallax Drift MVP repo:

1. `docs/!PM/DEVCONTAINER-SWARM-WORKFLOW.md` — the architecture doc.
2. `docs/plans/prompts/maestri-roles/all-m-subagent-roles.md` and
   `practicality-gatekeeper_v2role.md` — the original role prompts.
3. Operational experience from two real runs (see `PLAYBOOK.md`).

## Was the source workflow doc reviewed?

**Yes.**

- **Reviewer:** Amp, independent second-opinion desk review
- **Date:** 2026-07-22, refreshed against `stage` 2026-07-24
- **Report:** `docs/agent_reports/zOld/!0722-devcontainer-swarm-operational-review.md`
- **Verdict:** *NOT READY AS AN OPERATIONAL OR RELEASE GATE*

The reason the workflow doc shows no inline edits is that the review was written as a **separate
report**, then summarized into the doc as a header block (the "2026-07-22 operational review"
section). Nothing was annotated in place. That is why it looks unreviewed on a fresh pull.

Corroborated independently by the project memory store, which records:
*"DEVCONTAINER-SWARM-WORKFLOW.md is marked provisional/advisory … not yet operationally proven;
Maestri remains canonical."*

## What the review actually rejected — and what it did not

This distinction is the whole reason this bundle exists.

| Component | Status | Why |
|---|---|---|
| **Read-only reviewer swarm** | ✅ **PROVEN** | Ran 2026-07-20: 4 roles, ~6 minutes, 4 distinct real findings, no rubber-stamping. Zero infrastructure cost — needs no container, no auth, no Docker. |
| **Sign-off freshness invariant** | ✅ **ACCEPTED** | The review explicitly accepted this. See `SIGNOFF-INVARIANT.md`. |
| **Devcontainer write-side swarm** | ❌ **REJECTED as a gate** | 9 findings, 5 CRITICAL: no git/GitLab auth path, no Docker access, orchestration tools not provisioned, secret-manager auth unproven, worktree isolation assumed rather than bootstrapped. |

**This bundle carries the proven half.** The read-only reviewer swarm needs nothing the review found
missing — no container, no credentials, no Docker, no worktrees. It runs on any agent substrate today.

The write-side implementer roles are included (`roles/api-implementer.md`,
`roles/frontend-implementer.md`) but carry an explicit isolation warning. See `DEPLOY.md`
§ Write-side. Do not run parallel writers against a shared git directory.

## What was NOT verified

Stated plainly so nobody inherits a false confidence:

- No clean-room container run was ever completed. The 0722 review was read-only desk analysis.
- The five CRITICAL findings against the devcontainer remain open as of 2026-07-28.
- The swarm's findings have never been used as a *release gate* — only as advisory input.
- These role prompts have been exercised on exactly one codebase. Their portability to a second
  project is a design intent, not a measured result.
