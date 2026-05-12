# Benchmark Analysis — Latest Run (20260511T173411Z)

**Score: 34/50 perfect, 1 partial, 15 zero (avg 0.6880)**

Previous runs:

- 20260511 (latest_run): 30/50 perfect, avg 0.608
- 20260510: 33/50 perfect, avg 0.6788
- Original baseline (20260505): 25/50 correct (50%)

---

## Improvements Since Last Analysis

| Task     | Before | Now | Fix                        |
| -------- | ------ | --- | -------------------------- |
| task_199 | 0      | 1.0 | Fixed join/filter          |
| task_249 | 0      | 1.0 | COUNT DISTINCT fixed       |
| task_250 | 0      | 1.0 | Sanity check + retry       |
| task_303 | 0      | 1.0 | Fixed                      |
| task_418 | 0      | 1.0 | Threshold/filter corrected |
| task_420 | 0      | 1.0 | Duplicate column fix       |
| task_89  | 0      | 1.0 | Fixed                      |

---

## Still Failing: Zero-Score (15 tasks)

### Category 1: Wrong Computation / Thresholds (5 tasks)

| Task     | Difficulty | Question                                                       | Predicted | Gold   | Root Cause                                                                                                                                                                                                                              |
| -------- | ---------- | -------------------------------------------------------------- | --------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| task_169 | medium     | Average monthly consumption of SME 2013                        | 5519.48   | 459.96 | Grounding contradicts itself: MANDATORY CORRECTION says "don't divide by 12" but REFERENCE FORMULA says `AVG(Consumption)/12`. SQL does plain AVG which is wrong — gold IS total/12. Grounding self-contradiction confused the planner. |
| task_344 | hard       | Male patients normal WBC + abnormal fibrinogen                 | 2         | 4      | Grounding guesses standard clinical ranges (WBC 4-10, FG 2-4) since domain knowledge doesn't specify. Gold uses different thresholds → different count.                                                                                 |
| task_350 | hard       | Students at Women's Soccer wanting T-shirt                     | 0         | 7      | SQL uses `LIKE '%Women''s Soccer%'` which fails to match `"The Women's Soccer event"` (apostrophe escaping issue). Also incorrectly filters `t_shirt_size = 'medium'` — question may ask for any size.                                  |
| task_352 | hard       | Budget ratio Yearly Kickoff vs October Meeting (Advertisement) | 0         | 2.727  | Grounding correct but SQL uses `budget.event_id` join key which doesn't exist — actual FK is `budget.link_to_event`. Join fails silently → 0.                                                                                           |
| task_396 | hard       | % Marvel heroes height 150-180                                 | 50.0      | 54.84  | SQL logic looks correct (COUNT Marvel / COUNT \* WHERE height BETWEEN 150 AND 180). Discrepancy likely from `height_cm` stored as TEXT — some NULL/0 values included or excluded differently during CAST.                               |

### Category 2: Wrong Column / Filter (5 tasks)

| Task     | Difficulty | Question                                         | Predicted                  | Gold                | Root Cause                                                                                                                                                                                                 |
| -------- | ---------- | ------------------------------------------------ | -------------------------- | ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| task_25  | easy       | Event with lowest cost                           | February Speaker           | Nov/Oct/Sep Speaker | Grounding picks `budget.spent` as "cost" (CONSTRAINT says "Cost is defined by spent column"). But gold uses `expense.cost` table. Both columns listed in REQUIRED COLUMNS but wrong one chosen as primary. |
| task_75  | easy       | Best lap time in race 19 Q2                      | Fisichella                 | Räikkönen           | SQL correct (`ORDER BY q2 ASC LIMIT 1`). Likely `q2` contains time strings (e.g., "1:30.5") sorted lexicographically instead of as duration — string sort gives wrong MIN.                                 |
| task_86  | easy       | Races for Alex Yoong with track < 20             | race_name (correct values) | name (same values)  | Column alias mismatch: agent aliases races.name as `race_name` but gold expects literal column name `name`.                                                                                                |
| task_173 | medium     | Countries of gas stations June 2013              | "Country" (single value)   | CZE, SVK            | SQL is correct but post-processing collapses result. `pre_answer` shows `cols=['result'], rows=1` — answer formatting renamed column to 'result' and returned only 1 row. Bug in answer extraction.        |
| task_180 | medium     | Consumption for people paying >29/unit product 5 | 153 rows (wrong)           | 9 rows (correct)    | JOIN creates duplicates: multiple transactions per customer × yearmonth row. Needs to first get DISTINCT CustomerIDs from transactions, THEN join to yearmonth. Missing subquery/dedup step.               |

### Category 3: Wrong Aggregation/Grouping (2 tasks)

| Task     | Difficulty | Question                                  | Predicted                 | Gold                   | Root Cause                                                                                                                                                                                                        |
| -------- | ---------- | ----------------------------------------- | ------------------------- | ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| task_163 | medium     | Expense types + total for October Meeting | Advert 54.25, Food 121.14 | Meeting, 175.39        | Question asks "type of expenses" — grounding interprets as expense `category` (Advertisement, Food). Gold wants `event.type` column + total SUM(cost). Semantic ambiguity: "type of expenses" ≠ expense category. |
| task_22  | easy       | Date Connor Hilton paid dues              | 2019-10-02 (1 date)       | 2019-10-02, 2019-09-12 | Grounding says `EXPECTED ROWS: single` but there are 2 dues payments. SQL adds `LIMIT 1` because grounding says single row. Grounding incorrectly assumed single result. (fixed)                                  |

### Category 4: Complex Computation / Doc Extraction (3 tasks)

| Task     | Difficulty | Question                                                 | Predicted                 | Gold                 | Root Cause                                         |
| -------- | ---------- | -------------------------------------------------------- | ------------------------- | -------------------- | -------------------------------------------------- |
| task_379 | hard       | Toxicology elements of 4th atom, carcinogenic            | 8 elements (has 'na','h') | 7 elements (has 'f') | Wrong molecule classification or atom numbering    |
| task_408 | hard       | % faster champion vs last in 2008 Australian GP          | None                      | 0.316%               | Time parsing failed — returned None                |
| task_415 | hard       | Constructor ref + website for 2009 Singapore GP champion | (empty)                   | mclaren, wiki URL    | No prediction produced — likely timeout or failure |

---

## Partial Scores (1 task)

| Task     | Difficulty | Score | Issue                                                                |
| -------- | ---------- | ----- | -------------------------------------------------------------------- |
| task_257 | medium     | 0.40  | Total views correct (1708) but user DisplayName is NULL — wrong join |

---

## Grounding Root Cause Summary

| Issue Type                                               | Tasks Affected    | Fix Approach                                                                                   |
| -------------------------------------------------------- | ----------------- | ---------------------------------------------------------------------------------------------- |
| Grounding picks wrong column when question is ambiguous  | task_25, task_163 | Need tiebreaker: prefer `expense.cost` over `budget.spent` when question says "cost" literally |
| Grounding says EXPECTED ROWS: single when multiple exist | task_22           | Don't trust row count prediction — remove LIMIT unless grounding is very confident             |
| Grounding self-contradiction (CORRECTION vs FORMULA)     | task_169          | MANDATORY CORRECTION and REFERENCE FORMULA must be consistent                                  |
| SQL uses wrong FK column name                            | task_352          | Schema slice should show actual FK columns clearly                                             |
| Post-processing bug collapses results                    | task_173          | Answer extraction renames/collapses multi-row results                                          |
| String sorting instead of numeric for time columns       | task_75           | Need time-string-to-seconds conversion before ORDER BY                                         |
| Column alias vs actual column name                       | task_86           | Don't alias columns — use SELECT name directly                                                 |
| Missing dedup step in multi-join                         | task_180          | Need subquery pattern for "get customers THEN get their data"                                  |
| Apostrophe escaping in LIKE patterns                     | task_350          | Fix string escaping or use different match strategy                                            |

---

## Priority for Next Fixes (by estimated ease)

1. **task_86** (easy) — Don't alias: `SELECT name` not `SELECT name AS race_name`. Nearly free.
2. **task_22** (easy) — Remove LIMIT 1 when grounding says single but data has multiple. Or change grounding to not predict row count.
3. **task_173** (medium) — Bug in answer post-processing. Result is correct but gets collapsed.
4. **task_25** (easy) — Grounding literal match: question says "cost" → use `expense.cost` not `budget.spent`.
5. **task_352** (hard) — FK column name mismatch. Schema slice should make `link_to_event` obvious.
6. **task_350** (hard) — Apostrophe escaping fix + remove incorrect t_shirt_size filter.
7. **task_180** (medium) — Add dedup subquery pattern.
8. **task_75** (easy) — Time string comparison needs conversion.
9. **task_163** (medium) — Semantic: "type of expenses" → event.type. Hard to fix without special-casing.
10. **task_169** (medium) — Grounding self-contradiction. Needs formula consistency check.

---

## Fixes Implemented This Session

1. **computation_type in grounding** — Grounding now outputs explicit type (ratio/percentage/count_distinct/etc.) injected as hard constraint into SQL prompt
2. **Sanity check (\_sanity_check_result)** — LLM verifies result makes sense (catches 0%, None, ratio=1.0)
3. **SQL validation (\_validate_sql_against_grounding)** — LLM checks SQL uses correct columns per grounding
4. **FK lookup for doc extraction workers** — Workers see PK→name mappings from existing tables
5. **Post-extraction FK resolution** — Fuzzy matches text values to actual PKs after extraction
6. **Better worker extraction hints** — Category/type and numeric field extraction rules in prompt
7. **Duplicate column name fix** — Case-insensitive dedup prevents SQLite crashes (task_420)
8. **Planner fallback improvement** — Uses knowledge columns when planner times out
9. **consolidated.db in output** — DB copied to artifacts for inspection
10. **progress_callback in docker_entrypoint** — Copies predictions immediately as each task finishes (fixes score=0 on eval)
11. **\_strip_thinking_tokens** — Removes <think> blocks from vLLM responses
12. **extra_body for Qwen** — Disables thinking mode via chat_template_kwargs
