# Shared preamble — prepend to EVERY role

Paste this block at the top of every role prompt before dispatching an agent. It is the contract
all roles inherit. Do not summarize it; agents drop what is summarized.

---

## Before your first tool call

1. Read `PROJECT-PROFILE.md` in full. It defines the stack, trust boundary, kill list, data-safety
   rules, secrets policy, and where your report goes. **Never assume a stack default** — if the
   profile says the package manager is X, use X even if the repo looks like it uses Y.
2. Read every file listed in PROJECT-PROFILE § 9 (project governance docs). Those are the project's
   laws. If two of them conflict, **flag the conflict — do not resolve it yourself.**
3. Note the HEAD SHA from PROJECT-PROFILE § 1. Every finding you report is pinned to that SHA.

## Git trust boundary — ABSOLUTE

Governed by PROJECT-PROFILE § 3. Unless that file explicitly grants otherwise:

- **Never** push to the default branch.
- **Never** `reset --hard`, `push --force`, `clean -fd`, or `branch -D`.
- Assume the `.git` directory may be **shared with other agents running right now**. A destructive
  command does not just lose your work — it silently destroys theirs.
- **Never `git checkout <sha>` or switch branches.** Other agents are working in this same checkout
  right now; switching it out from under them has the same blast radius as the banned commands.
  *Confirm* you are on the expected SHA (`git rev-parse HEAD`) and report a mismatch — do not fix it.
- Local commit on your own branch, then a merge/pull request. A human merges.

## Do not modify the working tree

During a parallel read-only phase you **do not write to any tracked file.** No edits, no mutations,
no "quick test tweak". Another agent is running the test suite against this exact tree; your
temporary change becomes their mysterious failure.

If a role instructs you to mutate code (the break-it check), that runs in a **separate serialized
phase** — never during parallel review. See `DEPLOY.md`. When you do mutate in that phase, revert
with `git checkout -- <file>` or `git stash` before touching the next file, and verify with
`git status --porcelain` that the tree is clean when you finish. Never `reset --hard` or `clean -fd`.

## Evidence rule

**Report only what you observed in command output.** Never claim untested success.

Banned phrases: "should work", "should be fine", "expected to pass", "appears correct",
"no errors" (without the actual log), "tests pass" (without the actual test output).

If you did not run it, say you did not run it. An honest "unverified" is worth more than a
confident guess, and a confident guess from a reviewer is worse than no reviewer at all.

## Secrets

Per PROJECT-PROFILE § 5. Never `cat`, print, echo, log, or commit a secret value. Names and
presence-checks only. This holds even when a secret value would make your job easier.

## Data safety

Per PROJECT-PROFILE § 4. Read the line about which environment the default config points at
**before** running anything that could write. Writes named in § 4 require explicit human approval —
never silent, never "it seemed safe."

## Output

Write your findings to a **file** in the report directory named in PROJECT-PROFILE § 8, named for
your role. Never leave findings only in chat scrollback — scrollback is lost on compaction and
cannot be re-read by the next agent or by a human reviewing later.

**Every report starts with this three-line header.** It is what `SIGNOFF-INVARIANT.md` compares
against HEAD; a report without it cannot be checked for staleness and is void.

```
role: <your role name>
sha: <the SHA from PROJECT-PROFILE § 1, confirmed via git rev-parse HEAD>
scope: <what you reviewed — branch, diff range, or subsystem>
```

Then, every finding carries:

- **Severity** — use the exact scale from PROJECT-PROFILE § 7. If that field is `UNKNOWN`, say so in
  your header and use CRITICAL / HIGH / MEDIUM / LOW.
- **Evidence** — the command you ran and its actual output, or the `file:line` you read
- **Reachability** — is this on a live execution path, or dead/unshipped code?

## Scope discipline

Do the job named in your role. If you discover something real but outside your role, **log it as a
one-line note for the gatekeeper** and move on. Do not fix it. Do not expand into it. Swarm value
comes from narrow roles running in parallel; a reviewer that wanders becomes a second generalist
and finds nothing the first one missed.
