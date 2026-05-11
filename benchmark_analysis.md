# Benchmark Analysis — Latest Run (20260511)

**Score: 36/50 perfect, 4 partial, 10 zero (avg 0.7633)**

Previous run (20260510): 33/50 perfect, avg 0.6788
Original baseline (20260505): 25/50 correct (50%)

---

## Improvements Since Last Analysis

| Task | Before | Now | Fix |
|------|--------|-----|-----|
| task_145 | 0 | 1.0 | computation_type + sanity check |
| task_243 | 0 | 1.0 | computation_type "ratio" guided correct division |
| task_250 | 0 | 1.0 | sanity check caught None result, retried |
| task_352 | 0 | 1.0 | FK lookup + planner field extraction improved |
| task_420 | 0 | 1.0 | sanity check caught 0%, duplicate column fix |

---

## Still Failing: Zero-Score (10 tasks)

### Category 1: Wrong Computation / Thresholds (4 tasks)

| Task | Difficulty | Question | Predicted | Gold | Root Cause |
|------|-----------|----------|-----------|------|------------|
| task_169 | medium | Average monthly consumption of SME 2013 | 82027220.30 | 459.96 | SUM vs AVG ambiguity — "Total Annual Consumption" interpreted as SUM. Known unsolvable. |
| task_249 | medium | Avg upvotes + age for users with >10 posts | 177.07, 34.09 | 182.28, 34.08 | Wrong GROUP BY — counting posts differently (COUNT(*) vs COUNT(DISTINCT)) |
| task_344 | hard | Male patients normal WBC + abnormal fibrinogen | 2 | 4 | Missing domain knowledge — WBC/FG normal ranges not in knowledge.md |
| task_418 | extreme | Patients with abnormal creatinine, not 70 yet | 0 | 1 | Wrong threshold or filter — returns 0 instead of 1 |

### Category 2: Wrong Column / Filter (3 tasks)

| Task | Difficulty | Question | Predicted | Gold | Root Cause |
|------|-----------|----------|-----------|------|------------|
| task_25 | easy | Event with lowest cost | February Speaker | Nov/Oct/Sep Speaker | Non-deterministic: grounding picks budget.spent instead of expense.cost ~40% of runs |
| task_75 | easy | Best lap time in race 19 Q2 | Nakajima | Räikkönen | Wrong qualifying column — uses q3 or wrong race filter |
| task_257 | medium | Total views + user for 'Computer Game Datasets' | 1708, (empty) | 1708, mbq | DisplayName is NULL in joined result — wrong join or user lookup |

### Category 3: Wrong Aggregation/Grouping (2 tasks)

| Task | Difficulty | Question | Predicted | Gold | Root Cause |
|------|-----------|----------|-----------|------|------------|
| task_163 | medium | Expense types + total for October Meeting | Advertisement 54.25, Food 121.14 | Meeting, 175.39 | Wrong interpretation: grouped by expense category instead of event type. Gold wants event.type + SUM(cost) |
| task_396 | hard | Percentage of Marvel heroes height 150-180 | 53.125 | ~53.1 (different calc) | Likely rounding or filter difference |

### Category 4: Complex Computation (1 task)

| Task | Difficulty | Question | Predicted | Gold | Root Cause |
|------|-----------|----------|-----------|------|------------|
| task_408 | hard | How much faster % champion vs last in 2008 Australian GP | 1.47 | different formula | Time string parsing + percentage formula mismatch |

---

## Partial Scores (4 tasks)

| Task | Difficulty | Score | Issue |
|------|-----------|-------|-------|
| task_86 | easy | 0.94 | Column name `race_name` vs `name` — values are correct |
| task_379 | hard | 0.67 | Doc extraction: some molecules incorrectly classified as carcinogenic |
| task_22 | easy | 0.50 | Missing one date (only returned 1 of 2 dues payments) |
| task_180 | medium | 0.06 | Wrong customer filter — "paid >29 per unit" misinterpreted |

---

## Fixes Implemented This Session

1. **computation_type in grounding** — Grounding now outputs explicit type (ratio/percentage/count_distinct/etc.) injected as hard constraint into SQL prompt
2. **Sanity check (_sanity_check_result)** — LLM verifies result makes sense (catches 0%, None, ratio=1.0)
3. **SQL validation (_validate_sql_against_grounding)** — LLM checks SQL uses correct columns per grounding
4. **FK lookup for doc extraction workers** — Workers see PK→name mappings from existing tables
5. **Post-extraction FK resolution** — Fuzzy matches text values to actual PKs after extraction
6. **Better worker extraction hints** — Category/type and numeric field extraction rules in prompt
7. **Duplicate column name fix** — Case-insensitive dedup prevents SQLite crashes (task_420)
8. **Planner fallback improvement** — Uses knowledge columns when planner times out
9. **consolidated.db in output** — DB copied to artifacts for inspection

---

## Priority for Next Fixes

1. **task_25 non-determinism** — Grounding sometimes picks budget.spent over expense.cost. Need stronger column selection or deterministic override when column name matches question word.
2. **task_163 aggregation grain** — "type of expenses" misinterpreted as expense category vs event type. Needs better question parsing.
3. **task_249 COUNT(DISTINCT)** — Need to enforce DISTINCT when grounding says count_distinct.
4. **task_257 NULL DisplayName** — Join issue, user lookup returns NULL.
5. **task_396 percentage** — Close to correct, may be rounding/filter issue.
6. **task_86 column name** — Simple rename issue, nearly free fix.
