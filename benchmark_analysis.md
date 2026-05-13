# Benchmark Analysis — Latest Run (20260512T093443Z)

**Score: 37/50 perfect, 1 partial, 12 zero (avg 0.7571)**

Previous runs:

- 20260512 (latest): 37/50 perfect, avg 0.757
- 20260511T173411Z: 34/50 perfect, avg 0.688
- 20260511 (earlier): 30/50 perfect, avg 0.608
- 20260510: 33/50 perfect, avg 0.679
- Original baseline (20260505): 25/50 correct (50%)

---

## Improvements Since Last Analysis (34 → 37)

| Task     | Before | Now | Fix                                       |
| -------- | ------ | --- | ----------------------------------------- |
| task_169 | 0      | 1.0 | Formula check + clear domain_rules        |
| task_75  | 0      | 1.0 | Domain SQL extraction + null guard        |
| task_25  | 0      | 1.0 | LLM column name priority + re-grounding   |
| task_173 | 0      | 1.0 | Filter validation + re-grounding feedback |
| task_22  | 0      | 1.0 | Removed EXPECTED ROWS / LIMIT 1 bias      |
| task_180 | 0      | 1.0 | Fixed (dedup/join issue resolved)         |
| task_415 | 0      | 1.0 | Fixed (constructor ref + website)         |

---

## Still Failing: Zero-Score (12 tasks)

### Category 1: Wrong Computation / Aggregation (5 tasks)

| Task     | Difficulty | Question                                                   | Predicted | Gold    | Root Cause                                                                 |
| -------- | ---------- | ---------------------------------------------------------- | --------- | ------- | -------------------------------------------------------------------------- |
| task_243 | medium     | How many times is user 24 posts compared to avg            | 1.0       | 0.375   | Ratio inverted or wrong denominator                                        |
| task_249 | medium     | Average up votes and average age for users with 100+ views | 3.29, 35  | 182, 34 | Wrong filter or wrong column for up votes                                  |
| task_283 | medium     | Percentage of superheroes with blue eyes                   | 100.0     | 31.2    | WHERE filter too restrictive (only blue-eyed heroes counted as population) |
| task_396 | hard       | % Marvel heroes height 150-180                             | 53.125    | 54.84   | Slight value mismatch — likely NULL/text handling in height column         |
| task_408 | hard       | % faster champion vs last in 2008 Australian GP            | 0.32      | 0.316   | Rounding: pred=0.32 vs gold=0.3156 (1.4% rel error)                        |

### Category 2: Wrong Column / Filter / Join (4 tasks)

| Task     | Difficulty | Question                                                       | Predicted           | Gold            | Root Cause                                                                |
| -------- | ---------- | -------------------------------------------------------------- | ------------------- | --------------- | ------------------------------------------------------------------------- |
| task_163 | medium     | Expense types + total for October Meeting                      | Advert 54, Food 121 | Meeting, 175.39 | "type of expenses" → event.type not expense category. Semantic ambiguity. |
| task_250 | medium     | Post by slashnick with most answers count                      | "id,answercount"    | 351             | Parsing error — returned column names as value instead of actual result   |
| task_350 | hard       | Students at Women's Soccer wanting T-shirt                     | 0                   | 7               | Apostrophe escaping in LIKE or wrong filter logic                         |
| task_352 | hard       | Budget ratio Yearly Kickoff vs October Meeting (Advertisement) | 0                   | 2.727           | Wrong FK join (budget.event_id doesn't exist, need budget.link_to_event)  |

### Category 3: Domain Knowledge / Threshold Guessing (2 tasks)

| Task     | Difficulty | Question                                          | Predicted | Gold | Root Cause                                                        |
| -------- | ---------- | ------------------------------------------------- | --------- | ---- | ----------------------------------------------------------------- |
| task_344 | hard       | Male patients normal WBC + abnormal fibrinogen    | 2         | 4    | Grounding guesses clinical ranges not defined in domain knowledge |
| task_418 | extreme    | Patients abnormal creatinine + specific condition | 0         | 1    | Threshold guessing — domain knowledge doesn't specify ranges      |

### Category 4: Complex Multi-step / Doc Extraction (1 task)

| Task     | Difficulty | Question                                                  | Predicted | Gold | Root Cause                                 |
| -------- | ---------- | --------------------------------------------------------- | --------- | ---- | ------------------------------------------ |
| task_214 | medium     | Brazilian Portuguese translated sets in Commander Legends | 0         | 7    | Wrong table or filter — likely MTG dataset |

---

## Partial Scores (1 task)

| Task     | Difficulty | Score | Issue                                              |
| -------- | ---------- | ----- | -------------------------------------------------- |
| task_379 | hard       | 0.86  | 7/8 elements correct — has 'h','na' instead of 'f' |

---

## Key Fixes This Session

1. **Formula validation check** — Fast LLM compares grounding formula against domain knowledge, corrects dropped operations (e.g. AVG(X)/12 → was being reduced to just AVG(X))
2. **Clear domain_rules on formula correction** — When formula check fires, contradicting constraints are removed so SQL planner follows the corrected formula
3. **SQL planner literal formula rule** — "implement REFERENCE FORMULA LITERALLY — do NOT rewrite or substitute any function or operator"
4. **Anchors above knowledge_guidance** — Precise SQL-aggregate formulas from anchors take priority over ambiguous LaTeX/English in knowledge_guidance
5. **Parallel LLM checks** — ThreadPoolExecutor for (filter_validation + column_priority) and (semantic_feedback + formula_check) — saves ~1 LLM call latency per group
6. **Domain SQL extraction** — Inline `- SQL: \`...\`` patterns extracted and injected as reference
7. **Null guard improvement** — `_apply_null_guard` adds both `IS NOT NULL AND col != ''` for ORDER BY queries

---

## Priority for Next Fixes

1. **task_283** (medium) — Percentage logic wrong: counts only blue-eyed heroes as population instead of all heroes. Population vs metric confusion.
2. **task_250** (medium) — Parsing bug: returned column names as value. Answer extraction issue.
3. **task_249** (medium) — Wrong up votes value (3.29 vs 182). Likely wrong column.
4. **task_163** (medium) — "type of expenses" semantic ambiguity. Hard without special-casing.
5. **task_352** (hard) — FK mismatch: need `link_to_event` not `event_id`.
6. **task_350** (hard) — Apostrophe escaping in LIKE patterns.
7. **task_344/task_418** (hard/extreme) — Domain threshold guessing. Very hard to fix generically.
8. **task_396** (hard) — Slight value mismatch from NULL/text handling.
9. **task_408** (hard) — Rounding (1.4% error). Might pass with looser tolerance.
