# Session handoff — Striatica DAG forensics
**Session:** `cse_01WKQWZ6WC979k6NNRAAvxcW` · 2026-09-02 · Claude (Cowork)
**Purpose:** carry state to a fresh session. Every claim below has a re-run command.
**Trust rule:** treat my *assessments* as unverified. Treat items under ESTABLISHED
as re-runnable facts — check them yourself, they take seconds.

---

## 1. ESTABLISHED (mechanically verified against the repo)

**E-1 · There is ONE document, not four versions.**
`6.2.2.1` and `6.2.2.2` in `dashboards/grafana/` differ by exactly: the title
string `v6.2.2.1`→`v6.2.2.2`, and 13 markdown table-separator rows. Zero words
of prose, zero requirements, zero node IDs, zero hashes differ.
```bash
diff <(tr -s '[:space:]' '\n' < dashboards/grafana/6.2.2.1-implementation-control.md) \
     <(tr -s '[:space:]' '\n' < dashboards/grafana/6.2.2.2-implementation-control.md)
```

**E-2 · The DAG has never changed — not once, across every version.**
134 unique node/requirement/gate IDs, identical ID-set hash `ee2b75596df53470`,
in all four files *including* the original 1,604-line `4830997…` from git history.

**E-3 · The apparent "huge differences" are Prettier reflow.**
`6.2.2.1` is 105,966 chars; `6.2.2.2` is 91,491. The 14,475-char gap is padding
dashes in table separators. Line-based diffs report hundreds of changes; word-level
diffs report ~14. **Every line count quoted as evidence during this crisis —
1604 / 1637 / 1648 / 1649 — moved because of whitespace, not content.**

**E-4 · Amp's "current patch" is at a path Amp reports incorrectly.**
Amp links `img/correction/v6_plans/6.2.2-implementation-control.md` — that path
does not exist. The real file is one directory deeper:
`img/correction/v6_plans/v6-plan-archive/6.2.2-implementation-control.md`
(1,648 lines, sha `4378538d…`, modified 2026-09-02 00:31:08, gitignored).
Its content is the same document as E-1.

**E-5 · Three copies of that one document exist.**
```
dashboards/grafana/6.2.2.1-implementation-control.md    TRACKED
dashboards/grafana/6.2.2.2-implementation-control.md    TRACKED
img/correction/v6_plans/v6-plan-archive/6.2.2-...md     gitignored  (Amp edits this one)
```

**E-6 · Why CodeRabbit commented only on 6.2.2.2.**
In commit `6585da2`, CodeRabbit saw **1 changed line** in 6.2.2.1 and **589** in
6.2.2.2 (rename + reflow). It reviews diff hunks, not files. Its findings apply
equally to both — it only looked at one. **Formatting currently decides what gets
code-reviewed.**

**E-7 · The revision against the 1,604-line original IS real, and small.**
398 differing tokens. Concretely: `CHG-0902` 0→4 occurrences, `SRC-AMP22` absent→present,
`B-EUC-ROBUSTNESS` 3→7 occurrences, repository basis updated off the stale `13ff88b`.
It added a change record, a registry row, and three checkpoint prerequisites.
**It never touched the graph** — which is why every diff looked like nothing changed.

**E-8 · `control/dag.yaml` DOES NOT EXIST and never has.** No `control/` directory.
It is a path the spec *specifies* for node `K-SCHEMA-01`, which has never run.
**The DAG has only ever existed as English prose.** Amp's "52 nodes / 95 edges /
73 requirements / acyclic" came from a validator parsing that prose.

**E-9 · `striatica-v6.2.2-consolidated-plan.md` was deleted** in `6585da2` (−604).
A private copy survives at `img/correction/v6_plans/`, 604 lines.

**E-10 · Commit `6585da2` is UNPUSHED.** That is why no PR exists.
`such-wow-stack#4` in the VS Code status bar is a **different repository**.

---

## 2. CORRECTIONS TO MY OWN OUTPUT — distrust these earlier claims

| # | What I said | Truth |
|---|---|---|
| C-1 | Referred to `control/dag.yaml` as though it existed | It has never existed (E-8) |
| C-2 | "6.2.2.1 → 6.2.2.2 is +61/−28, sequential revisions" | **Wrong.** Siblings, identical content (E-1) |
| C-3 | "All four CHG-0902 corrections PASS across 12 locations" | True, but it reads as coverage. It is **4 findings of roughly 30** (`DS S-01`…`S-12`, `DS E-01`…`E-13`, §8.2 rows). ~26 remain unchecked |
| C-4 | "The screenshots settle it" | Overstated. They confirmed the §1.1 hashes match; they settled less than I implied |
| C-5 | Said your screenshots hadn't arrived | I checked the wrong directory. They were in project files |

---

## 3. UNVERIFIED — do not build on these

- **~26 of ~30 review findings** have never been checked for incorporation into node text.
- **The 398-token delta** between the 1,604 original and 6.2.2.2 — only the CHG-0902 portion checked.
- **Structural revalidation of 6.2.2.2.** The 52/95/73/acyclic figure was computed by
  **Amp's own validator** against the **old 1,604-line file**. Self-review on a stale version.
- **Any scientific claim.** Out of scope for all of the above.

---

## 4. DOCUMENT OUTPUTS I CREATED THIS SESSION

All live in the cloud session workspace (`~/striatica-forensics/`), **not** in your
repo. **They vanish when this session ends** — the copies delivered into the chat
are the durable ones.

| File | Lines | Status |
|---|---|---|
| `TIMELINE-what-went-wrong.md` | 264 | **Parts 1–3 stand. Part 4 contains error C-2** — it asserts a 6.2.2.1→6.2.2.2 lineage that does not exist. Use Parts 1–3; discard Part 4 §4.3. |
| `REVIEW-6.2.2.2-CHG0902.md` | 189 | Verification of 4 CHG-0902 findings + finding R-01 stands. **Read with correction C-3** — it is not whole-spec coverage. |
| `PUNCHLIST-6.2.2.2.md` | 157 | **RETIRED per your instruction.** Its substance is CodeRabbit's and the Oracle's findings; its ordering and assessment are mine, and you don't have grounds to trust those yet. Not deleted, not carried forward. |
| `THREAD-SUMMARY-2026-0902.md` | this file | handoff |

**One punch-list item survives because it is load-bearing right now** — see §5.

---

## 5. BLOCKING DECISION — read before pushing

`git push` would publish commit `6585da2`, which includes
`dashboards/grafana/6.2-private-changelog.md`. **That file's own second line reads:**

> "This ignored/private changelog records changes to v6 planning… It is not the
> public product changelog and must never contain semantic labels, secrets, or
> private evidence payloads."

It declares itself private and is currently staged to go public. Its visible
content includes raw working notes — `NOT NOT NOT review-ready`, `Amp unclear on
"inline patch edit meaning"`. CodeRabbit independently flagged this
(`6.2-private-changelog.md` Ln 12: *"Keep private planning notes out of tracked
`dashboards/`"*).

**I could not confirm whether the repo is public** — `gh` is unavailable in the
bridge shell. Given your threat model, the patent sensitivity, and that Striatica
is meant to carry scientific credibility, **I did not push.**

Three options, your call:
1. `git rm --cached dashboards/grafana/6.2-private-changelog.md`, amend, then push.
2. Confirm the repo is private and push as-is.
3. Push as-is knowingly.

§8.2 of the spec already rules on the general question ("public-safe neutral maps
are credibility artifacts… AMP F-07 applies to *private findings/strategy
documents* copied into `dashboards/`"). Whether this changelog is one is a
principal decision under §3.1. Not an agent's call.

---

## 6. YOUR STATED SEQUENCE FROM HERE

1. Resolve §5, commit, push.
2. Return to the timeline of what went wrong.
3. Then the YAML deliverable (`K-SCHEMA-01` / `control/dag.yaml`).
4. Only then, fix things.

**Root-cause note, offered not proposed:** until Prettier is pinned or these control
docs are added to `.prettierignore`, the phantom-diff in E-3 and the review-coverage
lottery in E-6 will both recur on the next edit.
