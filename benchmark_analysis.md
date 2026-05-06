# Benchmark Analysis — Run 20260505T070453Z

**Score: 25/50 correct (50%)**

- Exact match: 12
- Value match (correct values, column names differ): 13
- Wrong: 21
- Timeout: 4

---

## Category 1: Timeouts (4 tasks)

| Task | Difficulty | Question |
|------|-----------|----------|
| task_349 | hard | (timed out before producing answer) |
| task_396 | hard | (timed out before producing answer) |
| task_408 | hard | (timed out before producing answer) |
| task_418 | hard | (timed out before producing answer) |

**Root cause:** Graph extraction (discovery + per-entity extraction) takes too long on large documents. All are hard tasks with doc files.

---

## Category 2: Wrong Computation / Logic (7 tasks)

| Task | Difficulty | Question | Predicted | Gold |
|------|-----------|----------|-----------|------|
| task_169 | medium | Average monthly consumption of SME customers in 2013 | 82027220.30 | 459.96 |
| task_196 | medium | Average number of bonds for iodine atoms | 2.0 | 1.0 |
| task_249 | medium | Average upvotes and age for users with >10 posts | 340.0, 34.08 | 182.28, 34.08 |
| task_214 | medium | Brazilian Portuguese sets in Commander block | 0 | 7 |
| task_344 | hard | Male patients with normal WBC + abnormal fibrinogen | 2 | 4 |
| task_350 | hard | Students at Women's Soccer wanting medium T-shirt | 2 | 7 |
| task_352 | hard | Budget ratio Yearly Kickoff vs October Meeting (Advertisement) | 0 | 2.727 |

**Root cause:** SQL logic errors — wrong aggregation, wrong filters, wrong join conditions, or misinterpreted knowledge.md thresholds.

---

## Category 3: Empty / Missing Results (4 tasks)

| Task | Difficulty | Question | Predicted | Gold |
|------|-----------|----------|-----------|------|
| task_173 | medium | List countries of gas stations with transactions in June 2013 | (empty — header only) | CZE, SVK |
| task_330 | hard | Final score Sep 24 2008 Belgian Jupiler League match | (empty — header only) | 1, 1 |
| task_415 | hard | Constructor ref + website for 2009 Singapore GP champion | (empty — header only) | mclaren, url |
| task_86 | easy | Which race was Alex Yoong in with track number < 20 | (returned reasoning text) | Australian GP, Malaysian GP, ... |

**Root cause:** Agent couldn't find the data — either graph extraction failed on doc files, the SQL returned no rows due to wrong filters, or required tables were missing from context.

---

## Category 4: Wrong Entity Selection (4 tasks)

| Task | Difficulty | Question | Predicted | Gold |
|------|-----------|----------|-----------|------|
| task_19 | easy | Members who grew up in Illinois | Annabella Warren, Tyler Hewitt, Trent Smith | Trent Smith, Tyler Hewitt, Annabella Warren |
| task_25 | easy | Which event has the lowest cost | Officers meeting - November ($20.2) | November Speaker, October Speaker, September Speaker |
| task_199 | medium | Schools from Riverside districts with avg SAT math > 400 | River Springs Charter, La Sierra High, ... | Arlington High, John W. North High, ... |
| task_75 | easy | Best lap time in race 19 Q2 | Fisichella | Räikkönen |

**Root cause:** Wrong sort order, wrong filter interpretation, or wrong subset of records.

---

## Category 5: Wrong Output Format (5 tasks)

| Task | Difficulty | Question | Issue |
|------|-----------|----------|-------|
| task_38 | easy | Cash withdrawals for client 3356 | Returned all columns (7), gold wants only trans_id. Also only 2 rows vs 16. |
| task_180 | medium | Consumption for customers paying >29/unit for product 5 | Wrong customers + wrong consumption values |
| task_379 | hard | Tally toxicology element of 4th atom per carcinogenic molecule | Returned element+count, gold wants just element list |
| task_163 | medium | Expense types and total for October Meeting | Returned itemized expenses, gold wants single "Meeting,175.39" |
| task_355 | hard | Member who spent on water+veggie tray+supplies with cost | Returned multiple members, gold=single member "Elijah Allen,28.15" |

**Root cause:** Misinterprets what to return — too many columns, wrong interpretation of grouping, or wrong AND vs OR logic for filters.

---

## Category 6: Column Format Only (1 task — actually correct)

| Task | Difficulty | Predicted | Gold |
|------|-----------|-----------|------|
| task_27 | easy | "Sacha Harrison", 866.25 | "Sacha", "Harrison", 866.25 |

**Root cause:** full_name vs first_name + last_name split. Values are correct.

---

## Priority for Fixes

1. **Timeouts (4)** — optimize graph extraction speed or increase timeout
2. **Wrong computation (7)** — improve SQL generation and knowledge.md rule extraction
3. **Wrong output format (5)** — better interpretation of what columns/rows to return
4. **Empty results (4)** — improve graph extraction reliability
5. **Wrong entity selection (4)** — better filter/sort logic
