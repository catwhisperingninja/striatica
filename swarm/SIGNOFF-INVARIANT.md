# The sign-off freshness invariant

One rule. It is what keeps swarm findings honest, and it is the part of the original design an
independent review explicitly **accepted**.

> **A sign-off certifies exactly one commit SHA. Any commit to the branch after a sign-off
> invalidates every sign-off on that branch.**

## Why this exists

A real incident: a reviewer signed off on `466a7af`. The artifact that actually shipped was
`2bb1d0b` — an intervening fix, never re-verified. A green checkmark outlived the thing it
certified. Nobody lied; the approval simply aged out silently while still looking valid.

This is the default failure mode of every human-readable approval record. A checkmark has no
concept of time.

## What a compliant sign-off record contains

Whatever format you use, four fields are mandatory:

| Field | Why |
|---|---|
| `commit` | The exact SHA certified. Not a branch name — branches move. |
| `verdict` | `approved` / `rejected`. |
| `signer` | Who signed. Must be **different** from the worker. |
| `timestamp` | When. |

And four conditions must **all** hold at merge time, or the merge blocks:

1. A sign-off record exists for this branch.
2. Its `commit` **equals current HEAD**. Stale → block.
3. Its `verdict` is `approved`.
4. Its `signer` is **not** the worker. Self-sign-off → block.

**Fail closed.** If the record is missing, the parser is unavailable, or a field cannot be read —
block. A gate that fails open is not a gate.

## If you already have a gate — two holes to check for

*(Skip this on a new project; you have no gate yet. Come back when you build one.)*

**A — it no-ops outside its own substrate.** Gates written for a specific orchestration system often
skip when they detect they are not running inside it. A swarm running outside is then **not gated at
all.** Route swarm sign-offs through the same record; never write a second gate.

**B — it checks one record and misses tables.** A per-branch record file is blind to a per-role
sign-off **table inside a planning doc** — which is exactly where the `466a7af`/`2bb1d0b` mismatch
lived. Those `@<sha>` rows need the same `== HEAD` comparison.

## Do NOT build

The invariant is a comparison, not a platform. Building any of these means you have already lost:

- ❌ A dashboard.
- ❌ A database or new record format. The SHAs already exist in git and in your docs.
- ❌ A bespoke CI stage separate from the gate you already have.
- ❌ A second, parallel sign-off system. **Extend the existing one.** Two authority models is worse
  than none.
- ❌ A "remember to run this script" step. A gate an agent must remember to invoke is the same
  prose-drop this invariant exists to prevent. It must fire automatically or it does not exist.

## Minimum honest version

If you have no gate infrastructure at all, this is enough to start:

> Every report file in the report directory carries the SHA it was produced at, in its first three
> lines. Before merging, compare each against HEAD. Any mismatch means that review is void and must
> be re-run.

Manual, but honest — and it makes the staleness **visible**, which is the entire point. Automate it
when the manual comparison starts getting skipped, because it will.
