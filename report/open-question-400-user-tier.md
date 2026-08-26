# Brief: the 400-user tier of the scaling ladder contradicts the chapter-7 headline

Repository: `/Users/olivierrecher/Documents/IMT/SIO/training`. The LaTeX report is
in `report/` (`main.tex`, chapters in `report/sections/`, figures and tables in
`report/figures/` and `report/tables/`). Read `CLAUDE.md` before writing any
prose: it carries standing writing rules, in particular no em dash anywhere, the
`\figtext{}` convention, and footnotes that define rather than explain.

## The problem

The report states two different gaps for what appears to be the same run, on the
same corpus, against the same baseline, at 400 training users.

**The +3.7 pt claim.** `sections/07-groups.tex:216`

> The run after those four changes reached **46.8 % against the same 43.1 %
> baseline: +3.7 pt**, the first run of the whole project to clear "repeat the
> last cell" on a full test set rather than on a hand-picked segment.

`sections/07-groups.tex:247` confirms the population: "It is what the
configuration is worth on **four hundred training users**, and the configuration
itself, hour tokens plus one adapter per behavioural group, is carried unchanged
into Chapter 8." `tables/tab-recap.tex:17` records the same run as iteration 8,
corpus `\smob`, protocol "full seq.", baseline 43.1 %, model 46.8 %, gap +3.7.

**The +0.6 pt claim.** `sections/08-scale.tex:66`

> **Imitation, at 400 training users.** +0.6 pt. The model is barely
> distinguishable from the rule it is supposed to beat.

`sections/08-scale.tex:39` introduces that ladder as "The proper version fixes
everything except the number of users. Same architecture, same hyperparameters,
same context fraction, hour tokens and behavioural groups active throughout",
which reads as the chapter-7 configuration continued.

**Where it becomes an outright contradiction.** The ladder figure names the
corpus of each tier. `figures/fig-scaling-ladder.tex:63`

> The 900 to 1,900-user tiers are scored on `\smob` (100 test users, 48.2 %
> baseline) **and the 400-user point is the `\smob` run (43.1 % baseline)** and
> the 5,000 to 21,000-user tiers on `\seleven` (1,000 test users, 50.2 %
> baseline)

43.1 % is the full-sequence baseline of the mobile-user day, so that sentence
attributes the +0.6 point to exactly the corpus, protocol and baseline of the
run that chapter 7 reports at +3.7. Note also that the same sentence assigns the
48.2 % baseline to `\smob` where `tables/tab-datasets.tex` assigns it to
`\stwok`, so at least one of the two attributions in that caption is wrong.

**It reaches the three most-read pages.** `sections/00-abstract.tex:22`,
`sections/01-introduction.tex:126` and `sections/10-conclusion.tex:26` all state
"the edge grows from +0.6 to +4.2 points between 400 and 1,900 training users".
In the conclusion that sentence sits six lines after `10-conclusion.tex:20`,
"That single change took the model from 1.8 points below the rule to 3.7 points
above it".

## What is not in dispute

- The 43.1 % / 45.5 % pair is a protocol difference, not a data difference, and
  the report already handles it well (`sections/06-time.tex`, its Retrospective,
  and the "Two baselines for \smob" note in `tables/tab-datasets.tex`). Do not
  touch that machinery.
- The ladder's upper tiers are sound: +4.2 at 1,900 users on `\stwok`, +4.3 flat
  from 5,000 to 21,000 on `\seleven`, with the paired bootstrap of
  `tables/tab-bootstrap.tex`.
- `tables/tab-provenance.tex` lists no run producing a +0.6 pt gap, and
  `tables/tab-datasets.tex` lists `\stwok` training sizes as 900 / 1,900 only.
  Neither the 400-user nor the 1,400-user tier of the ladder is documented
  anywhere as belonging to a named corpus.

## The two hypotheses, and what settles each

1. **The 400-user tier is `\stwok`** (400 users drawn from the 2,000-user
   day-file), and the figure caption is simply wrong to call it the `\smob` run.
   Then nothing about either number changes: the fix is the caption, plus one
   sentence in chapter 8 saying the ladder's small tiers are not the chapter-7
   corpus and that the two 400-user figures answer different questions.
2. **The 400-user tier is the `\smob` adapter run re-trained at the ladder's
   constant-compute budget** (8,000 training examples, as `sections/08-scale.tex`
   describes for every tier). Then +0.6 and +3.7 are both real and the difference
   is the training budget, which is a substantive finding the report currently
   hides: the headline +3.7 would not survive the budget the scaling chapter
   imposes. That has to be said explicitly, in chapter 7's retrospective and in
   chapter 8.

What settles it: the notebook `train_cellid_final.ipynb` produced the eight-tier
ladder (`\figsource` of `figures/fig-scaling-ladder.tex`), and
`train_cellid_llm.ipynb` produced the adapter-routing run. Neither carries stored
outputs for these numbers in the repository, so the answer is in the Colab run
logs or in Olivier's notes. **Do not guess.** If you cannot establish it, report
what you found and stop.

## Two related wording problems that need no decision

Both are safe to fix once the above is settled.

1. **The group is described as a token in chapters that use adapters.**
   `sections/01b-state-of-the-art.tex:169` tells the reader that chapter 7's
   group is "a single token prepended to the sequence, so one model serves every
   group and the grouping is a hint rather than a partition", which is the
   mechanism that failed, and the opposite of four adapters over a frozen base.
   `sections/08-scale.tex:179` then justifies the whole scaling threshold with
   "A group token can only carry information if the group has enough inhabitants
   [...] which is precisely what the failed ablation showed it doing", in a
   chapter whose runs all use adapter routing. `sections/09-critique.tex:39`
   repeats "under-populated group token", though it hedges properly afterwards.
   `sections/09-critique.tex:87` and `sections/10-conclusion.tex:31` get the
   wording right and can serve as the model.

2. **The hour is shown to be read, then shown not to be, without reconciliation.**
   `sections/06-time.tex` establishes H1: shuffling the hour costs 3.0 pt, four
   times the cost of removing it. `sections/07-groups.tex:168` then reports
   "41.3 % with real hours against 41.8 % with no hours whatsoever, if anything,
   slightly better without them", and `:172` answers only "nor in the hours,
   which an earlier ablation had shown the model does use. It was in the
   channel." The channel argument covers a single token at the head of the
   sequence, not one hour token per event, so it does not explain the second
   result. What is missing is one sentence saying the two ablations are different
   measurements: chapter 6's runs on the targeted-sampling checkpoint at 80 %
   context over n = 1,747 predictions with the asymmetric x2 / x4 weights
   (`figures/fig-hour-ablation.tex`), chapter 7's on the group-token checkpoint
   over full sequences against 43.1 %.

## Constraints

- Never invent or interpolate a measured number. Every figure in the report is
  traceable through `tables/tab-provenance.tex`; if a claim needs a number that
  is not there, say so instead of producing one.
- Keep every edit consistent across the abstract, the introduction, chapters 7
  and 8, the critique, the conclusion, and the affected figure captions and
  tables. A number that moves has to move everywhere.
- The slide deck `Présentation Stage - Finale.pptx` has already been corrected
  for the same three problems. Leave it alone unless asked; if the resolution
  changes a number, flag which slides are affected (19, 21, 22) rather than
  editing them.
