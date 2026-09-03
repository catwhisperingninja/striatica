# Playbook — what actually happened

Lessons from two real swarm runs. **Read this before acting on any finding.** The architecture doc
tells you what should happen; this tells you what did.

## Observed runs

| Date | Configuration | Wall clock | Result |
|---|---|---|---|
| 2026-07-20 | 4 read-only reviewers, native subagents, parallel | ~6 min | 4 distinct real findings, no rubber-stamping |
| 2026-07-27 | Full QA pass: verifier + test auditor + security + gatekeeper + domain role | one session | Found a live security regression a green pipeline had been hiding |

Both runs produced findings a single reviewer had missed. Both also produced **false alarms that
nearly shipped as headline findings.** The lessons below are the difference.

---

## Lesson 1 — Accepted policy is not a defect

**What happened:** a reviewer reported the CI audit gate's severity threshold as "the primary
lying-green surface." It was a deliberate project decision — fail on Critical, let High-severity
advisories ride green, revisit later.

**Why it matters:** reporting an intentional trade-off as a defect burns the human's time and, worse,
trains them to discount the whole report. An auditor that cries wolf on policy gets ignored on the
real finding three lines down.

**How to apply:** PROJECT-PROFILE § 7 has an **accepted-risk policy** field. Fill it in. Reviewers
read it. The gatekeeper cuts anything that contradicts it. When a reviewer genuinely believes an
accepted risk has become unacceptable, that is an *escalation with a reachability argument*, not a
finding.

## Lesson 2 — Risk is likelihood × reachability, never count

**What happened:** eight High-severity advisories in the production dependency tree looked alarming.
Assessed individually: several were in a browser bundle not present in the deployed API container;
one required an HTTP/2 configuration the project does not enable; one needed a route-guard feature
with zero usages in the codebase; one was build-time-only. Exactly **one** was a real target.

**How to apply:** never report a count. For each advisory establish whether the package ships in the
deployed artifact, whether exploitation needs a config the project does not use, and whether it is
reachable from untrusted input. Lead with the reachable one. Collapse the rest into a single line.

## Lesson 3 — Verify the wiring before reporting the outage

**What happened:** a reviewer probed a provider's default model constant, got an error, and drafted
a headline finding that a major dependency was "down" and needed a one-line fix. Reading the actual
engine wiring showed that constant is **dead code** — every call site passes an explicit model, and
that provider was primary on no slot at all. The finding was killed before it reached the human,
but only because someone read the wiring.

**How to apply:** before reporting any component as broken, grep for its **call sites**, not just
its definition. Confirm it is on the execution path. Probing dead code and reporting a live outage
is the most embarrassing failure mode this swarm has, and the easiest to prevent.

## Lesson 4 — Measure the tree you think you're measuring

**What happened:** a local dependency scan reported packages that a completed remediation sprint had
removed. Both were true: the **lockfile** was correct, so CI and production were fine — but the
local `node_modules` was stale, still holding the old packages on disk. Every local scan that
session had been measuring the wrong tree.

**How to apply:** before trusting any local scan, confirm the installed tree matches the lockfile.
And when a search tool returns nothing surprising, check whether it is silently excluding
directories — a search that skips `node_modules` by default will cheerfully report a clean tree.

## Lesson 5 — Reports go to files, always

Chat scrollback is lost on compaction, invisible to the next agent, and unciteable by a human three
days later. Every role writes a file. The gatekeeper reads files. The human reads the gatekeeper's
file. This is what makes findings survive the session that produced them.

## Lesson 6 — The gatekeeper runs last, and it reviews the reviewers

**What happened:** the gatekeeper was nearly skipped, and the reviewers' raw output nearly went
straight to the human. The correction was explicit: *every auditor runs through the gatekeeper
before submission.*

**Why:** auditors over-report — that is the correct behavior for an auditor. Four over-reporting
auditors produce a pile no one can act on. The gatekeeper's compression is what turns the pile into
a decision. It is not an optional fifth opinion; it is the output stage.

## Lesson 7 — Write deferrals as a trajectory, not an exclusion

**What happened:** a human said a certain kind of tracking was "not a priority right now." An agent
wrote it into a sprint doc as "explicitly not a task, ever." That invented boundary would have
outlived the conversation and become doctrine a later cycle cited to skip real work.

**How to apply:** "coarse now, thorough at maturity" — never "not needed." A permanent boundary
belongs in a durable doc only if a human actually stated it as permanent. This applies to every
`DEFER` verdict the gatekeeper issues.
