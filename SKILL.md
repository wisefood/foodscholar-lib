---
name: progress-log
description: Maintain a technical project-management log in PROGRESS.md — append a new iteration entry, summarize an iteration from git history, query past iterations, or audit the log for staleness against the actual repo state. Use when the user says things like "log this iteration", "add a PROGRESS entry", "what did we do in iteration N", "summarize since last log entry", "close out this milestone", or otherwise asks to read/write the project's running iteration log. The log lives at the repo root as PROGRESS.md (newest entries on top) and exists alongside BRIEF.md (the plan) — this skill owns the log, not the plan.
---

# Progress Log

This skill maintains a single file: `PROGRESS.md` at the repo root. It is a running log of completed work, organized by iteration. Newest entries on top. Each entry covers **what changed, why, and how it was verified**.

The log is read by future-you (or a collaborator) walking in cold. Optimize entries for that reader: they have not seen the conversation that produced the work.

## When to use this skill

Invoke when the user asks to:

- **Append a new entry** — "log this iteration", "add a PROGRESS entry for what we just did", "close out milestone M4", "write up today's work".
- **Summarize from git** — "summarize commits since the last log entry", "draft a PROGRESS entry from the last N commits", "what's the diff between PROGRESS and `git log`".
- **Query the log** — "what did we do in iteration 6", "what came before X", "when did we land Y", "show me the last three iterations".
- **Audit the log** — "is PROGRESS still accurate", "find stale references in PROGRESS", "what's described in PROGRESS but no longer exists".

Do **not** invoke for: planning future work (that's BRIEF.md or a plan), tracking in-progress tasks (that's TodoWrite), or general code questions.

## File location and shape

- **Path:** `PROGRESS.md` at the repo root. Confirm with `ls PROGRESS.md` before assuming it exists.
- **Order:** newest entry on top, immediately after the file header.
- **Separator between entries:** a line with three hyphens (`---`) on its own.
- **Sibling file:** `BRIEF.md` holds the forward-looking plan (§12 step list, milestones). PROGRESS references BRIEF (e.g. "per BRIEF §12 step 10") but does not duplicate it.

## Entry structure

Each entry follows this shape. Match the existing style of the file — read the most recent two entries before writing to pick up local conventions.

```markdown
## YYYY-MM-DD — Iteration N.M (Milestone): one-line headline

**Goal:** one paragraph stating what this iteration was trying to accomplish and why
the prior state was insufficient. Future-you reads this first to understand the
motivation — make it self-contained.

### What changed (or: What landed)

Group by commit when commits are the natural unit, otherwise by subsystem. For each:

- **`<short-sha>` short-title.** What the commit does in 1–3 sentences. Use
  markdown links to files: [path/to/file.py](src/path/to/file.py) and
  [path/to/file.py:42](src/path/to/file.py#L42) for line-specific refs.
- Or group by module: **`src/foo/`** — what changed there.

### Design decisions worth remembering

Bullets capturing *why* a choice was made, especially when the choice is non-obvious
or there were viable alternatives that were rejected. Future-you returning to the
code in six months reads this section to avoid re-litigating settled trade-offs.

### Verification

How you confirmed it works. Test counts, ruff/mypy clean, manual smoke, notebook
cells that ran. Be specific: "**220 passed, 2 skipped**" beats "tests pass".

### Status at end of iteration

One short paragraph: what now works, what is committed vs uncommitted (notebook
state often lags), what the next milestone is. End with a pointer to BRIEF §X for
"what's next" — do not duplicate the plan here.
```

Optional sections that appear in some real entries and are good to use when they fit:
- **Tuning learned on the real corpus** — config-side knobs discovered during the iteration but not committed as library code. Worth recording so the value isn't relearned.
- **`<topic>` — investigated and CLOSED** — when an open question was resolved this iteration, state the resolution and the reasoning trail.
- **Handoff** — explicit "start here next session" pointer when the iteration ends mid-arc.
- **Environment** — only when the iteration changed the dev environment (Python version, new conda env, new system dep).

## Workflow: appending a new entry

When the user asks to log an iteration:

1. **Read `PROGRESS.md`** (at least the top two entries) to pick up the local headline style, milestone tag convention (e.g. `M3`, `M4`), and section ordering. The conventions evolve — match what's there, don't impose a template.
2. **Gather the raw material** in parallel:
   - `git log --oneline <since>..HEAD` for the commits this iteration covers. If the user didn't specify a range, find the SHA of the most recent commit referenced in PROGRESS.md and use that as the floor.
   - `git diff --stat <since>..HEAD` for a sense of scope.
   - For each commit you'll cite individually, `git show --stat <sha>` or read the commit message for the "why" — the body of a good commit message is often the seed of the PROGRESS bullet.
3. **Convert relative dates to absolute** — "today" → the actual ISO date. Today's date is available in the conversation context; do not guess.
4. **Draft the entry** following the structure above. Specifically:
   - Headline: `## YYYY-MM-DD — Iteration N.M (Milestone): <hook>`. Increment N.M from the most recent entry; the user may correct.
   - Cite commits by short SHA + a short title that matches the commit subject's spirit (not necessarily verbatim).
   - File references use markdown link syntax: `[file.py](src/file.py)` or `[file.py:42](src/file.py#L42)`.
   - Lead each "What changed" bullet with what the change *does*, not what files moved. "Stopped `foodon_ids` denorm from bypassing the link_blocklist" beats "Modified propagate.py".
5. **Insert at the top** of PROGRESS.md, immediately after the file header (the intro paragraph and the first `---`). Use Edit, not Write — preserve the rest of the file.
6. **Show the user the inserted entry** and ask whether anything's missing before considering it final. Do not commit unless explicitly asked.

## Workflow: querying the log

For "what did we do in iteration N" / "what came before X" style questions:

- `grep -n "^## " PROGRESS.md` to list all iteration headings.
- Read the relevant section with `Read` + `offset`/`limit`. Do not read the whole file unless needed.
- When the user asks about a commit or feature, also check `git log --oneline --all --grep=<term>` — PROGRESS is a summary, the git history is authoritative.
- Answer with markdown file links for any file paths you mention.

## Workflow: auditing the log against current repo state

When the user asks "is PROGRESS still accurate":

- A PROGRESS entry is a snapshot in time. Don't "correct" history — old entries describe what was true *then*. Only flag drift if the user wants to know what's stale, or if a *current* recommendation is being drawn from a stale claim.
- Specifically check: file paths cited still exist (`ls`), function names cited still appear (`grep`), commit SHAs still resolve (`git rev-parse <sha>`).
- Report findings as a punch list of "PROGRESS says X, repo now shows Y". The user decides whether to amend.

## Style guide

- **Voice:** terse, factual, past tense for landed work ("landed", "wired", "dropped"). No marketing tone.
- **No emojis.** Match the existing file.
- **Specifics over generalities.** "Raised cosine threshold from 0.88 to 0.94" beats "tightened threshold".
- **Name the alternative when a decision rejected one.** "Per-facet overrides, not per-facet config blocks" — the contrast carries the meaning.
- **Mark uncommitted notebook/config state explicitly.** Distinguish "landed" (committed library code) from "tuning learned on the real corpus" (notebook/config work not in the library).
- **Cross-reference, don't duplicate.** Forward-looking work points at BRIEF §X. Previous iterations are cross-linked by their date headline only when load-bearing.
- **One line ≠ one bullet.** A bullet can be a short paragraph when the change deserves it. Don't fragment a coherent change into five bullets.

## What NOT to write in PROGRESS

- **In-progress work.** PROGRESS is for landed iterations. Use TodoWrite or a plan for the live task.
- **Forward-looking plans.** Those live in BRIEF.md. PROGRESS may end with a pointer ("Next: Layer B per BRIEF §12") but should not enumerate the plan.
- **Routine refactors with no design content.** If "the change" is just a rename or a lint fix, fold it into the relevant commit bullet — don't give it its own section.
- **Conversation framing.** "We discussed and then decided to..." — drop the meta. State the decision.

## Memory note

This skill writes to PROGRESS.md, which is repo-tracked. It is not the same as the auto-memory system in `~/.claude/projects/.../memory/`. PROGRESS.md is shared with collaborators via git; memory is private to your Claude Code session.
