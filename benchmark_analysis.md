# Benchmark Analysis — Latest Run (20260513T034859Z)

**Score: 38/50 (76%) with 5% numeric tolerance**

Previous runs:

- 20260513 (latest): 38/50, 76%
- 20260512T093443Z: 37/50 perfect, avg 0.757
- 20260511T173411Z: 34/50 perfect, avg 0.688
- 20260511 (earlier): 30/50 perfect, avg 0.608
- 20260510: 33/50 perfect, avg 0.679
- Original baseline (20260505): 25/50 correct (50%)

---

## Improvements Since Last Analysis (37 → 38)

| Task     | Before | Now | Fix                                                  |
| -------- | ------ | --- | ---------------------------------------------------- |
| task_250 | 0      | 1.0 | NULL→0 coalesce for quantity columns during consolidation |
| task_352 | 0      | 1.0 | Doc extraction pipeline (entity-boundary batching)   |
| task_283 | 0      | 1.0 | Fixed (population rule already in place)             |
| task_350 | 0      | 1.0 | Fixed (apostrophe handling)                          |

Tasks that regressed (were correct before, now wrong):

| Task     | Before | Now | Cause                          |
| -------- | ------ | --- | ------------------------------ |
| task_169 | 1.0    | 0   | LLM non-determinism (missing /12) |
| task_199 | 1.0    | 0   | LLM non-determinism (wrong lookup) |

---

## Still Failing: 12 tasks

### Category 1: Wrong Computation / Formula (5 tasks)

| Task     | Difficulty | Question                                               | Predicted    | Gold         | Root Cause                                                |
| -------- | ---------- | ------------------------------------------------------ | ------------ | ------------ | --------------------------------------------------------- |
| task_169 | medium     | Avg monthly consumption for SME 2013                   | 5519.5       | 459.96       | Missing /12 (LLM non-determinism, works when run alone)   |
| task_243 | medium     | How many times is user 24 posts compared to avg        | 1.0          | 0.375        | Ratio inverted or wrong denominator                       |
| task_249 | medium     | Avg up votes and avg age for users with >10 posts      | 3.29, 35.26  | 182.28, 34.08 | Used posts.Score instead of users.UpVotes                 |
| task_408 | hard       | % faster champion vs last in 2008 Australian GP        | 0.0          | 0.316        | Wrong computation (returned 0 instead of percentage)      |
| task_420 | hard       | Percentage calculation                                 | 0.0          | 100.0        | Inverted/wrong percentage                                 |

### Category 2: Wrong Lookup / Filter (4 tasks)

| Task     | Difficulty | Question                                               | Predicted                    | Gold           | Root Cause                            |
| -------- | ---------- | ------------------------------------------------------ | ---------------------------- | -------------- | ------------------------------------- |
| task_199 | medium     | School name lookup                                     | River Springs Charter        | Arlington High | Wrong filter (LLM non-determinism)    |
| task_214 | medium     | Brazilian Portuguese translated sets in Commander Legends | 0                          | 7              | NULL→0 coalesce may have caused this  |
| task_344 | hard       | Male patients normal WBC + abnormal fibrinogen         | 2                            | 4              | Domain threshold guessing             |
| task_418 | extreme    | Patients abnormal creatinine + specific condition      | 7                            | 1              | Domain threshold guessing             |

### Category 3: Wrong Row Count / Grain (2 tasks)

| Task     | Difficulty | Question                            | Predicted  | Gold     | Root Cause                              |
| -------- | ---------- | ----------------------------------- | ---------- | -------- | --------------------------------------- |
| task_379 | hard       | Element list                        | 17 rows    | 7 rows   | Wrong filter — too many results         |
| task_38  | medium     | Transaction list                    | 1 row      | 140 rows | Collapsed/aggregated when shouldn't have |

### Category 4: Crash (1 task)

| Task     | Difficulty | Root Cause                                              |
| -------- | ---------- | ------------------------------------------------------- |
| task_415 | hard       | IndexError in answer formatter (fixed — row length guard added) |

---

## Key Fixes This Session

1. **NULL→0 coalesce for quantity columns** — During consolidation, INTEGER columns with quantity-like names (count, votes, views, score, amount, etc.) get NULL→0. Excludes ID columns. Fixed task_250.
2. **Doc extraction pipeline** — Entity-boundary aware batching with `---` separators (6K char budget per batch). LLM decides what's an entity vs noise. Fixed task_352.
3. **Dedup merge fix** — Empty string `""` treated same as null to prevent overwriting real values.
4. **Phrase mapping in grounding** — Forces explicit column-to-question-phrase disambiguation.
5. **Grain consistency rule** — No GROUP BY without explicit "for each"/"per"/"by" in question.
6. **IndexError guard** — Answer formatter skips rows shorter than expected indices instead of crashing.
7. **Docker max_workers set to 1** — Safe for 12-hour submission limit (~7.8h for 378 tasks).

---

## Observations

- **LLM non-determinism** is the biggest source of variance. Tasks like 169, 199, 415 pass when run individually but fail in benchmark runs. Single worker doesn't help — it's just different LLM responses each time.
- **Domain threshold guessing** (task_344, task_418) is very hard to fix generically — clinical ranges not in the provided knowledge.
- **task_396** (53.125 vs 54.84) is a benchmark data inconsistency — doc extraction correctly reads 175cm for Picard but gold was computed with original placeholder 0.0.
- **task_214** (0 vs 7) may be a side effect of NULL→0 coalesce — needs investigation.

---

## Priority for Next Fixes

1. **task_249** (medium) — Grounding maps "up votes" to posts.Score instead of users.UpVotes. Needs better column disambiguation.
2. **task_243** (medium) — Ratio formula inverted. Similar to task_169's /12 issue.
3. **task_38** (medium) — Returns 1 row instead of 140. Answer shaper is collapsing data.
4. **task_420** (hard) — Returns 0% instead of 100%. Percentage computation inverted.
5. **task_408** (hard) — Returns 0 instead of 0.316. Computation fails entirely.
6. **task_379** (hard) — Returns 17 rows instead of 7. Filter too loose.
7. **task_214** (medium) — Check if NULL→0 coalesce caused regression.
8. **task_344/task_418** (hard/extreme) — Domain threshold guessing. Very hard to fix generically.
