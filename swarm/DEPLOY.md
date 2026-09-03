# Deploying the swarm

How to actually launch these roles. Pick the section matching your substrate. The **shape** is
identical everywhere; only the invocation syntax differs.

## The shape (substrate-independent)

```
 PHASE 0   fill PROJECT-PROFILE.md      §3/§4/§6/§7 need a HUMAN. Do not skip.
                   ↓
        ┌─ adversarial-verifier ─┐
 PHASE 1├─ test-auditor ─────────┤  3 (+ optional domain role) in PARALLEL
        └─ security-reviewer ────┘  STRICTLY READ-ONLY, ~6 min
                   ↓
 PHASE 2   break-it / mutation checks   SERIALIZED, one agent, writes to tree
                   ↓
 PHASE 3   practicality-gatekeeper      reads their reports, CUTS, compresses
                   ↓
 PHASE 4   human reads the gatekeeper's bottom line
                   ↓
 PHASE 5   write-side implementers (only if needed, ISOLATED — see below)
```

**Four rules that are not optional:**

1. **Phase 1 is strictly read-only — no agent writes to a tracked file.** They cost nothing but
   tokens: no worktree, no container, no disk. Launch them in a single batch, not sequentially.
2. **Mutation is Phase 2, alone.** The break-it check (test auditor) and mutation checks
   (verifier) modify code. Run against a shared tree during Phase 1, one agent's mutation becomes
   another's phantom test failure — and the verifier reports it as a real regression. Serialize
   them, revert each with `git checkout -- <file>`, and confirm `git status --porcelain` is clean
   before Phase 3.
3. **The gatekeeper runs LAST.** It reviews the other reviewers' output, not just the code. Running
   it first or in parallel wastes it — its whole job is compression and cutting, and it needs
   something to cut. *This was learned the hard way; see `PLAYBOOK.md`.*
4. **Everything writes to a file**, with the three-line SHA header from `_SHARED-PREAMBLE.md`.
   Never chat scrollback. Report directory is PROJECT-PROFILE § 8. Before you act on any report,
   check its `sha:` against HEAD — see `SIGNOFF-INVARIANT.md`.

## Assembling a role prompt

Every dispatch is the same concatenation:

```
[contents of roles/_SHARED-PREAMBLE.md]
+
[contents of roles/<role>.md]
+
"PROJECT-PROFILE.md is at <path>. Read it first.
 Review scope: <what changed / which branch / which diff>.
 Write your report to <report-dir>/<role>.md."
```

Do not summarize the preamble or the role. Paste them whole. Summarized prompts produce
summarized reviewers.

---

## Substrate: Claude Code (subagents)

Native subagents. Launch all read-only roles in **one message with multiple tool calls** so they run
concurrently.

Set `$BUNDLE` to wherever you copied this directory (striatica: `swarm`) — every prompt needs
the **absolute or repo-relative path** to `PROJECT-PROFILE.md`, not the bare filename.

```
Agent(subagent_type: "general-purpose",
      description: "adversarial verify",
      prompt: "<contents of $BUNDLE/roles/_SHARED-PREAMBLE.md>
               <contents of $BUNDLE/roles/adversarial-verifier.md>
               PROJECT-PROFILE.md is at $BUNDLE/PROJECT-PROFILE.md. Read it first.
               Scope: <branch/diff>. PHASE 1 — READ-ONLY, do not modify any file.
               Report to <report-dir>/adversarial-verifier.md")

# same shape for test-auditor and security-reviewer
```

Wait for all three. Run Phase 2 (mutation) as a single agent if needed. **Then** dispatch the
gatekeeper with the three report paths in its prompt.

Last, dispatch a **verifier seat** — a fresh agent, not the gatekeeper and not the referee — with
`GATEKEEPER-VERIFICATION.md`, the input pile, and the gatekeeper's report. Checks 1, 2 and 4 there
are set and string comparisons; only the entailment check costs a model.

If your Claude Code install has purpose-built agent types (`adversarial-verifier`, `test-auditor`,
`security-reviewer`, `practicality-gatekeeper`), use those instead of `general-purpose` and still
prepend the profile pointer — the bundled role file carries project-portable content the installed
agent definition will not have.

## Substrate: Amp / Codex / Cursor / any agent CLI

Same shape. One process per role, launched concurrently, each with the assembled prompt as its
task. Give each its own report path so they cannot collide on a write.

```
BUNDLE=swarm                 # wherever you copied this directory
<agent-cli> exec --prompt "$(cat $BUNDLE/roles/_SHARED-PREAMBLE.md \
                                 $BUNDLE/roles/adversarial-verifier.md; \
  echo \"PROJECT-PROFILE.md is at $BUNDLE/PROJECT-PROFILE.md. Read it first.\"; \
  echo 'Scope: <branch/diff>. PHASE 1 — READ-ONLY, do not modify any file.'; \
  echo 'Report to <dir>/adversarial-verifier.md')" &
# ...repeat per role, then wait, then Phase 2, then the gatekeeper
```

The `PROJECT-PROFILE.md is at …` line is **mandatory**. The role files reference the profile eleven
times by bare filename; without a path a dispatched agent cannot resolve any of them.

Read-only roles need no special sandboxing. If your CLI has a read-only mode, use it — it makes the
read-only guarantee mechanical instead of prompt-dependent.

## Substrate: plain chat window (no agent tooling at all)

Fully supported and genuinely useful. Open one conversation **per role** — separation of context is
the mechanism, and separate tabs give you that for free.

1. Paste `_SHARED-PREAMBLE.md`, then the role file, then `PROJECT-PROFILE.md`, then the diff.
2. Ask for the report in the role's output format.
3. Save each reply to its report file yourself.
4. Open a final conversation for the gatekeeper and paste all the reports into it.

Slower, zero infrastructure, and it still catches real defects. This is the fallback that always works.

## Substrate: a governed floor/orchestration system

If your project already has one (Maestri-class: isolated environments, native role agents,
enforcement hooks), **that system stays canonical.** This bundle is an accelerator that runs on top
of it, not a replacement.

Adapt only the plumbing: drop any substrate-specific commands the role prompts reference, keep the
role substance verbatim. Findings from this swarm are **advisory input to** the governed sign-off,
never a substitute for it.

---

## Write-side — read this before dispatching an implementer

Read-only reviewers are free and safe. **Write-side implementers are neither.**

Two implementers writing in parallel against a **shared `.git` directory** will corrupt each other's
work. The file-territory split in the implementer roles (backend-only / frontend-only) reduces the
blast radius but does not eliminate it — index locks, branch state, and stashes are all shared.

Pick one, honestly:

| Approach | Isolation | Cost | Verdict |
|---|---|---|---|
| **Sequential writers** | Total | Slow | ✅ Always safe. Start here. |
| **Git worktrees** | Separate checkouts, shared `.git` | Low, some disk | ⚠️ Adequate for disjoint file territory. Watch disk sprawl. |
| **Separate clones** | Total | Disk + setup | ✅ Safe and simple. |
| **Devcontainers** | Total | High setup | ⚠️ Only if already proven. See below. |

**On devcontainers:** independently reviewed and rejected as a gate — the checked-in container could
not start the swarm it was meant to host (see `REVIEW-STATUS.md`). Do not adopt one on the assumption
it works. Use sequential writers or separate clones.
