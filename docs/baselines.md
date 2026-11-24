# Baselines: decisions and open questions

The baselines are the cheap evaluations that the capability probes are compared against. Each one spends at
most a share β of what a full long evaluation of the agent costs in tokens (`C_full`), and gives a number that
is used to predict the agent's long-task score. This file records what was decided about them, and what is
still open. Last updated 2026-11-23.

| Baseline | What it measures | Status |
| --- | --- | --- |
| B1 | the mean score over a few of the agent's long episodes | coded (`analysis/features.py b1`); needs the long runs |
| B2 | the agent's score on long tasks cut off after 20 steps | coded (`analysis/features.py b2`, `runs/b2_sw_cutoff.yaml`) |
| B3 | the share of a fixed random set of 100 non-agentic test questions (maths, general knowledge, reasoning) the plain model gets right | coded (`core.static.build`, `runs/b3_static.yaml`, `analysis/features.py b3`); pool not yet built |
| B3-PACE | the same, on 100 questions chosen the PACE way | question choice coded (`analysis/pace.py`); used by the prediction step, not yet written |

## Shared rules

- **The main score is progress:** the best ScienceWorld score an episode reached at any step, from 0 to 1. It
  is used for the long-task score being predicted, for B1 and for B2. The pilot (2026-10-27) showed that the
  score at the end of an episode is almost always 0 for small models (0.02 for granite-4.1-8b, against 0.07
  progress). This is mostly because choosing the wrong object as the answer ends the task at 0, even after
  partial credit was earned. The end score (`final_score`) is still saved for every episode and can be
  reported beside progress.
- **Budget:** β is one of 0.5%, 1%, 2%, 5% and 10% of `C_full`. `C_full` is the agent's mean tokens per seed
  of the full long evaluation, counting prompt and completion tokens, with cached tokens at full price.
- **Random draws:** where a baseline picks items at random under the budget, it repeats the pick 20 times with
  fixed seeds and writes one row per pick.
- **No answer if nothing fits:** if the budget is too small for even one item, the baseline gives no number
  at that budget, and the results show that. It does not overspend. This matters most for B1, where one long
  episode costs about 260k tokens and the smallest budget is about 350k.

## B1: fewer long tasks

Take the agent's long episodes in random order until the budget is spent, and use their mean progress as the
prediction directly (no regression). No extra episodes are run: B1 is drawn from the long-evaluation runs
themselves, one seed at a time.

**Open question for whoever builds the long-task scores.** The long-task score B1 is checked against is the
mean over all the agent's long episodes, including the ones B1 drew. B1 then partly contains the answer it is
predicting, which makes it look better than running a few new episodes would. The effect grows with the
budget (at 10%, B1 uses about 27 of the episodes). B2, B3 and the probes use separate tasks and questions, so
they do not have this advantage. Suggested fix: draw B1 from one seed and check it against the mean of the
other seeds. The long-task scores are already planned per seed, so this needs no new runs. Because the runs use
temperature 0, the seeds will be close to identical, so the fix changes the numbers little, but it keeps the
comparison fair.

## B2: short tasks

Dev-split tasks of the 19 ScienceWorld task types that have long test tasks, cut off after 20 steps. Each
episode is scored as its progress divided by the score the official solution has after 15 steps, at most 1.
Tasks where the official solution scores under 0.1 after 15 steps are left out, and at most 8 tasks are taken
per task type. The original definition ("dev tasks with an official solution of 10 steps or fewer") would only
contain task types that have no long tasks, so it compares different skills. It is kept as
`runs/b2_sw_short_plain.yaml` for comparison. The measurements behind this are kept outside git.

## B3: non-agentic questions, in two versions

Both versions use 100 questions from one pool of ordinary test questions, which every agent answers once.

- **B3** is the agent's accuracy on a fixed random set of 100 pool questions, the same set for every agent, drawn
  with a fixed seed. This is the baseline as decided on 2026-10-27. Nothing about it is fitted to the agents, so
  it cannot fit noise.
- **B3-PACE** follows PACE (Song et al., 2026, https://github.com/neulab/pace): keep the 100 pool questions whose
  right and wrong answers best follow the agents' long-task scores. It is the stronger version if the choice
  works, but the choice is made from about 13 agents in each round, so it may partly fit noise. PACE's published
  question set is not reused, because it was chosen to predict coding and web-browsing agents built on large
  models, and many of its questions need code to be run or images.

Both are reported. A claim that the capability probes predict better than non-agentic questions has to hold
against both versions.

- **Question pool:** about 500 text questions with one checkable answer: about 200 from MMLU-Pro (multiple
  choice), 150 from GSM8K (a number) and 150 from BBH (reasoning tasks with short answers). They are drawn once,
  with a fixed seed, and shared through the results dataset so every site answers the same questions.
- **Answering:** each agent's model answers every pool question at the same server as its episodes, with no
  scaffold and no environment. A thinking agent thinks, as it does in its episodes. The answer is written out
  and checked against the right one: the letter for multiple choice, the final number for GSM8K, an exact match
  for BBH. Comparing the model's probabilities for each choice was ruled out, because it needs a different
  server call and does not work for thinking agents.
- **B3's fixed random set** is drawn in `analysis/features.py b3`, which writes each agent's accuracy on it,
  left empty at a budget that the tokens of those questions pass.
- **For B3-PACE, choosing the 100 questions happens inside the prediction step.** In each round of testing (each
  leave-one-out fold), the 100 questions are chosen with PACE's rule using only the agents the predictor is fit
  on. The held-out agent's answers must not affect the choice, or B3-PACE would be predicting with knowledge of
  the answer. So B3-PACE is split across two parts of the code:
  - `analysis/features.py b3` also writes each agent's result on every pool question (right or wrong, and tokens
    spent).
  - The prediction step (`analysis/predict.py`, owned by whoever fits the predictors) chooses the 100 questions
    per fold with `analysis.pace.choose_questions`, then uses the held-out agent's accuracy on them
    (`analysis.pace.accuracy_on`) as its feature, in the same ridge regression as the other baselines. Only PACE's
    ranking of questions is used, not its second question set or its tuned blend of the two, because with about 14
    models there is nothing to tune them on. Two more things differ from PACE's code: a question's leverage is
    summed over all components of the table of answers, where PACE uses only the first few (1 to 10, tuned per
    target), and PACE predicts with its own Spearman-weighted regression, not plain accuracy on the questions.
- **Comparing the two versions** shows whether PACE's choosing beats random choice for small models, which
  Yauney et al. (2026) found it often does not for closely matched models.
- **Budget:** the 100 questions of either version are always used in full. They cost far less than the smallest
  budget for a non-thinking agent (about 0.06M tokens against 0.35M).

Known limits, to state in the write-up:

- B3 cannot tell scaffolds apart: two agents with the same model and different scaffolds answer the questions
  the same way.
- PACE chose its questions using 14 models. We will have about the same number of models that answer
  differently (base models, fine-tunes and thinking variants), so B3-PACE's choice may partly fit noise. B3,
  with its fixed random set, shows how much.

Open questions:

- Whether the cost of answering the whole pool (about 0.3M tokens per non-thinking agent, more with thinking)
  should count against B3-PACE's budget. Suggested: no, as in PACE, where it is a one-time setup cost, but report it.
- Whether thinking agents' answers to the 100 questions fit under the smallest budget. Check after the
  first thinking agent answers the pool.
