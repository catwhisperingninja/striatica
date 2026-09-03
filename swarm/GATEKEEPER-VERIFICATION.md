# Verifying the gatekeeper's output

The other roles are checked by the gatekeeper. **The gatekeeper is checked by nothing** — it runs
last, it filters everything, and by design nothing downstream sees what it removed. This file is the
check.

It is a **comparison procedure, not a platform.** Same discipline as `SIGNOFF-INVARIANT.md`: if you
find yourself building a system, you have already lost. See § Do NOT build.

## Why a gatekeeper needs verifying more than a reviewer does

A reviewer's failures are loud. It reports something wrong and someone argues with it.

A gatekeeper's failures are **silent by construction**:

| Failure | What the human sees |
|---|---|
| **Wrong cut** — a real blocker classified `CUT` | nothing. The item simply never appears. |
| **Silent drop** — an item that got no verdict at all | nothing, and it is indistinguishable from a wrong cut |
| **Invented authority** — cut under a rule that is not in the corpus | a plausible one-line reason |
| **Rubber-stamp** — everything `APPROVE`d without assessment | a clean cycle, identical to a real one |

Every one of these looks exactly like success. A gatekeeper that stopped working produces the same
artifact as a gatekeeper that is working perfectly: a short list and a calm bottom line. That is why
"it seemed fine" is not evidence, and why this cannot be checked by reading the output alone.

The mechanism generalizes past software. Any gate that compresses a large pile into a short
recommendation — a bill's markup before committee, an agency's regulatory review, a permit
screening — has this same property: **the deletions leave no trace in the artifact that gets read.**
Verification means making the deletions enumerable and each one grounded in something a third party
can independently re-fetch.

## The verdict record

The gatekeeper's report is the record. Beyond the three-line header every role writes
(`_SHARED-PREAMBLE.md` § Output), each verdict line carries five fields:

| Field | Why |
|---|---|
| `id` | the id the item arrived with. Enables conservation checking. |
| `verdict` | `APPROVE` / `CUT` / `DEFER` / `FLAG` / `EXCEPTION→HUMAN` |
| `authority` | a locator — profile field, clause, section, `file:line` — resolvable in the CORPUS at the pinned version. **Required for every `CUT`, `DEFER` and `FLAG`.** On `EXCEPTION→HUMAN` only, the literal `none — <what could not be resolved>`; see check 2. |
| `reason` | one line. Prose, not checkable, and not the basis of any check below. |
| `reachability` | exactly one of `operative` / `non-operative` / `unknown` (from the role's risk section). Required on every line; a missing or out-of-vocabulary value voids the line. |

`authority` is the whole design. It converts "do I trust this judgment?" — unanswerable — into "does
the cited text exist and does it say that?" — answerable by someone who did none of the work.

## The four mechanical checks

Run in order. Any failure voids the report and it is re-run; a partially-valid gate is not a gate.

**1. Conservation.** Every input item id appears **exactly once** in the output. No drops, no
duplicates, no ids that were never an input.

Compare **multiplicities, not sets** — sorted id lists, or a count per id. A set comparison cannot
express "exactly once": input `[A, B]` against output `[A, A, B]` has identical sets, so a duplicate
verdict — two different dispositions for one item — passes unnoticed. It still needs no judgment:
run it first, because it catches the failure that is otherwise invisible.

**2. Authority resolution.** Every `CUT`, `DEFER` and `FLAG` verdict's `authority` locator resolves
in the corpus at the pinned version. A dangling locator is an **invented rule**, which is the
gatekeeper committing the exact error `PLAYBOOK.md` Lesson 7 warns about — inventing a boundary that
outlives the conversation and becomes doctrine a later cycle cites. Also mechanical.

`EXCEPTION→HUMAN` is the one exemption, and it is deliberate: that verdict exists for the case where
no authority resolves, so requiring one would make the only honest disposition unrepresentable and
push the gatekeeper back toward an uncitable `CUT`. It is exempt from *resolution*, not from
*checking* — its `authority` must read `none — <what could not be resolved>`. A bare `none`, or the
literal `none` on any other verdict, fails this check like a dangling locator.

**3. Entailment.** The text at the locator must actually support the verdict. *This is the only step
requiring judgment,* and therefore the only step where an independent model is spent. Fetch the
cited text, present it with the verdict, ask one question: **does this authority entail this
disposition?** Answer `entails` / `does not entail` / `underdetermined`. Do not ask whether the
verdict was wise — that is the human's call and not verifiable.

**4. Freshness.** `SIGNOFF-INVARIANT.md`, applied to the gatekeeper's own report: it certifies
exactly one version. If HEAD moved, the report is void. Do not build a second freshness mechanism —
the gatekeeper's report is a report file in the report directory and the existing rule already
covers it.

Checks 1, 2 and 4 are comparisons. Only check 3 costs a model. That ratio is the point: **most of
the verification is free**, which is what makes it survivable as a standing obligation rather than a
ceremony that gets skipped.

## The adversarial check — escape rate

Checks 1–4 establish that the gatekeeper's output is *grounded*. They cannot establish that it is
*discriminating*. A gatekeeper that `APPROVE`s everything passes all four: conservation holds, there
are no uncited cuts, nothing is stale.

So measure it against items whose correct disposition is already known — the same reason a test
suite is worthless until something adversarial proves the tests can fail.

**Seed from real history, never from invented examples.** Every project that has run this swarm has
a stock of past findings with recorded human dispositions. `PLAYBOOK.md` alone supplies four:

| Seeded item | Correct disposition | Catches |
|---|---|---|
| Lesson 1 — the audit gate's severity threshold reported as a defect | must-CUT (accepted policy) | permissiveness |
| Lesson 2 — eight advisories, exactly one reachable | must-CUT seven, must-KEEP one | counting instead of assessing |
| Lesson 3 — a dead-code constant reported as a live outage | must-CUT (unreachable) | reachability blindness |
| Lesson 3 again — the one genuinely reachable advisory | must-KEEP | **over-cutting** |

Mix them into a real pile. Two rates fall out, and they trade against each other:

- **miss rate** — must-CUT items that came back `APPROVE`. The gate is too permissive; waste passes.
- **over-cut rate** — must-KEEP items that came back `CUT`. The gate is destroying real work.

**Over-cut rate is the number that matters**, because it is the only one you cannot observe in
production. A missed item eventually surfaces as a bug, an incident, a cost overrun. An over-cut
item surfaces as nothing at all, ever. A gatekeeper whose default is `CUT` — and this one's is, by
design — will fail in that direction first, and will look excellent while doing it.

Report both rates with the denominator. `2/14 over-cut` is a measurement; "the gatekeeper is
working well" is not.

**Honest limits, stated up front:**

- If the gatekeeper can tell a seeded item from a real one, the measurement is void. Historical real
  items with their metadata normalized are the mitigation; synthetic decoys are detectable and
  should not be used. (This is the same reason the project bans mock data.)
- A seed set is a floor, not a guarantee. It proves the gate catches *these*. It says nothing about
  a class of error absent from the set.
- Rates from a set under ~20 items are indicative, not statistical. Report the count; do not compute
  a percentage that implies precision the sample does not have.

## Who verifies

The verifier is **not** the gatekeeper (no role reviews its own output), and **not** the referee —
the referee adjudicates disputes about gatekeeper filtering and would be ruling on evidence it
produced. A fresh seat with no stake in the cycle, ideally a different model, given only: the input
pile, the gatekeeper's report, and read access to the corpus.

The verifier does not re-do the review. It answers four questions and reports two rates. A verifier
that starts finding new defects has become a fifth auditor and is no longer verifying anything.

## Do NOT build

The verification is a comparison. Building any of these means the gate has been replaced by a
project:

- ❌ A dashboard or a score. Two rates and a void/valid flag is the entire output.
- ❌ A database or new record format. The verdicts are already in a report file; the versions are
  already in git.
- ❌ A separate sign-off system for gatekeeper reports. Extend the existing one.
- ❌ A model call for checks 1, 2, or 4. They are list, count, and string comparisons. Spending a model on
  them is how this becomes too expensive to run every cycle, which is how it stops being run.
- ❌ A standing seed corpus maintained as its own artifact. Seeds come from the report archive.

## Minimum honest version

With no tooling at all:

> Before acting on a gatekeeper report, sort its verdict ids and sort the input pile's ids and
> compare the two lists. Anything missing is a silent drop; anything appearing twice is a double
> disposition; anything present that was never an input is invented. Then open two cited authorities
> at random and read them. If either does not say what the verdict claims, the report is void and
> re-run.

Two minutes, no code, and it catches the two failure modes that are otherwise invisible. Automate
checks 1 and 2 when the manual comparison starts getting skipped, because it will.
