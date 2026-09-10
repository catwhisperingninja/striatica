# Review-Swarm Bundle

A portable, agent-agnostic kit for running a **parallel adversarial review swarm** on a codebase.
Drop this folder into any repo, fill in one file, and any coding agent can run it — Claude Code,
Amp, Codex, Cursor, or a plain chat window.

It ships six role prompts, deployment instructions for four substrates, and a playbook of what
actually happened the two times this was run for real.

**This is a review accelerator, not a governance system.** It produces findings fast. It does not
replace whatever merge gate your project already has. See `SIGNOFF-INVARIANT.md` for the one rule
that keeps its findings honest.

> ⛔ **striatica status (Laura ruling 2026-08-01): STAGED, NOT APPROVED, NOT IN USE.**
> This bundle was copied from parallax-drift-mvp **to be perfected, not to be run.** The
> workflow is not approved for striatica; the devcontainer form is not used at all; no
> agent may dispatch these roles here until Laura explicitly approves the workflow after
> hardening. The one pattern already permitted independently of this bundle: the
> practicality gatekeeper as a per-session **subagent advisor** (v5 plan §6.7).
> `PROJECT-PROFILE.md` here is a pre-filled **draft** — not authoritative (see its header).
> This directory was moved out of the gitignored `docs/**` and is now **tracked and public**;
> the staged-not-approved status is unchanged by that move. `GATEKEEPER-VERIFICATION.md` is
> specification only and authorizes no dispatch.

## Quickstart

1. Copy this directory into your repo (suggested: `swarm/`; anywhere is fine — no file in this
   bundle depends on its own path).
2. Fill in **`PROJECT-PROFILE.md`** — this is the *only* file you edit. Everything else reads it.
3. Read `DEPLOY.md` and pick the section matching your agent substrate.
4. Spawn the **three read-only reviewers in parallel** (verifier, test auditor, security), then the
   **gatekeeper last** — it reviews their reports.
5. Collect their report files. Read `PLAYBOOK.md` before you act on any finding, and run
   `GATEKEEPER-VERIFICATION.md` § Minimum honest version against the gatekeeper's report — two
   minutes, no tooling, and it catches the two failure modes nothing else can see.
6. Only then consider write-side roles (`api-implementer`, `frontend-implementer`) — these need
   real isolation. See `DEPLOY.md` § Write-side.

Total setup time: about ten minutes. A four-role read-only pass runs in roughly six.

## What's in here

| File | What it's for |
|---|---|
| `PROJECT-PROFILE.md` | **The only file you edit.** Stack, trust boundary, kill list, secrets policy. |
| `roles/_SHARED-PREAMBLE.md` | The contract every role inherits. Prepend it to each role prompt. |
| `roles/*.md` | Six role prompts — four read-only reviewers, two implementers. |
| `DEPLOY.md` | How to actually launch them, per substrate. |
| `PLAYBOOK.md` | Hard-won operational lessons. **Read before acting on findings.** |
| `GATEKEEPER-VERIFICATION.md` | How to check the gatekeeper itself — its failures are silent by construction. |
| `SIGNOFF-INVARIANT.md` | *Reference.* The freshness rule that stops stale approvals from merging. |
| `REVIEW-STATUS.md` | *Reference.* Provenance: where this came from and what was verified. |

**To dispatch run 1 you need only the first four** — that is setup. You still need `PLAYBOOK.md`
before you act on a finding, and `GATEKEEPER-VERIFICATION.md` before you act on the gatekeeper's
bottom line. The two marked *Reference* are worth reading after your first pass, not before it.

## The core idea

Four reviewers with **different adversarial postures**, run in parallel, find things one reviewer
doesn't. Not because four is better than one at the same job — because a verifier who tries to
*disprove* "tests pass" looks nowhere near where a gatekeeper who tries to *cut scope* looks.
Diversity of posture is the mechanism. Redundancy is not.

The roles are deliberately narrow and deliberately hostile to their own project. That is the point.
A reviewer that wants the work to succeed finds nothing.
