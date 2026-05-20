"""Pure SQL text manipulation utilities for QuestionDrivenAgent."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path


def _fix_unescaped_apostrophes(sql: str) -> str:
    """Fix unescaped apostrophes inside single-quoted SQL string literals.

    'Women's Soccer' → 'Women''s Soccer'
    Handles multiple literals in one statement.
    """
    result = []
    i = 0
    while i < len(sql):
        if sql[i] == "'":
            # Find the end of this string literal
            # Walk forward collecting chars; an apostrophe followed by a letter
            # (not another apostrophe and not end-of-token) is unescaped
            result.append("'")
            i += 1
            while i < len(sql):
                if sql[i] == "'" and i + 1 < len(sql) and sql[i + 1] == "'":
                    # Already escaped — keep both
                    result.append("''")
                    i += 2
                elif sql[i] == "'":
                    # Could be end of literal or unescaped apostrophe
                    # Heuristic: if next char is a word char (letter/digit) and prev char is
                    # also a word char, it's an unescaped apostrophe mid-word (e.g. Women's)
                    prev_is_word = i > 0 and (sql[i - 1].isalnum() or sql[i - 1] == ' ')
                    next_is_word = i + 1 < len(sql) and sql[i + 1].isalpha()
                    if prev_is_word and next_is_word:
                        result.append("''")
                        i += 1
                    else:
                        # End of literal
                        result.append("'")
                        i += 1
                        break
                else:
                    result.append(sql[i])
                    i += 1
        else:
            result.append(sql[i])
            i += 1
    return "".join(result)


def _sanitize_sql(sql: str, db_path: Path) -> str:
    """Fix common LLM SQL formatting issues: trailing junk, unquoted multi-word columns, unescaped apostrophes."""
    # Strip trailing braces/brackets that leak from JSON
    sql = sql.rstrip().rstrip("}").rstrip("]").rstrip()
    # Remove trailing semicolons (SQLite doesn't need them and they can cause issues with multiple statements)
    sql = sql.rstrip(";").strip()

    # Strip trailing column alias when the SELECT has no top-level FROM (scalar expression only).
    # Avoids CAST(...AS REAL) / column-alias AS ambiguity in SQLite.
    stripped_upper = sql.upper().strip()
    if stripped_upper.startswith("SELECT") and "CAST(" in stripped_upper:
        # Check if there's a top-level FROM (not inside a subquery)
        depth = 0
        has_top_from = False
        for token in re.finditer(r'\(|\)|FROM', sql, re.IGNORECASE):
            if token.group() == '(':
                depth += 1
            elif token.group() == ')':
                depth -= 1
            elif depth == 0 and token.group().upper() == 'FROM':
                has_top_from = True
                break
        if not has_top_from:
            sql = re.sub(r'\s+AS\s+"[^"]*"\s*$', '', sql, flags=re.IGNORECASE)
            sql = re.sub(r'\s+AS\s+[a-z_]\w*\s*$', '', sql, flags=re.IGNORECASE)

    # Fix unbalanced parentheses (common LLM mistake in nested scalar subqueries)
    open_count = sql.count('(')
    close_count = sql.count(')')
    if open_count > close_count:
        sql = sql + ')' * (open_count - close_count)
    elif close_count > open_count:
        excess = close_count - open_count
        for _ in range(excess):
            if sql.endswith(')'):
                sql = sql[:-1]

    # Fix unescaped apostrophes inside single-quoted string literals
    # e.g. 'Women's Soccer' → 'Women''s Soccer'
    sql = _fix_unescaped_apostrophes(sql)

    # Quote unquoted multi-word column names using actual schema
    try:
        conn = sqlite3.connect(str(db_path))
        all_columns: set[str] = set()
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
            for col in conn.execute(f'PRAGMA table_info("{row[0]}")').fetchall():
                col_name = col[1]
                if " " in col_name or "(" in col_name or "%" in col_name or "-" in col_name:
                    all_columns.add(col_name)
        conn.close()

        # For each multi-word column, find unquoted references and quote them
        for col in sorted(all_columns, key=len, reverse=True):
            # Match the column name not already inside quotes
            # CamelCase collapsed version (e.g. SchoolName for "School Name")
            no_space = col.replace(" ", "").replace("(", "").replace(")", "").replace("-", "")
            if no_space in sql:
                sql = sql.replace(no_space, f'"{col}"')
            # Also fix dot-prefixed versions (e.g. frpm.SchoolName)
            for prefix in ("frpm.", "satscores.", "T1.", "T2."):
                if f"{prefix}{no_space}" in sql:
                    sql = sql.replace(f"{prefix}{no_space}", f'{prefix}"{col}"')
    except Exception:
        pass

    return sql


def _enforce_grounding_filters(sql: str, grounding_context: str, db_path: Path) -> str:
    """Replace incorrect filter columns/values with those specified in CONDITIONS.

    Handles two cases:
    1. LLM used a different column from the same table (original behavior)
    2. LLM used the semantic value (e.g. 'carcinogenic') instead of the coded value
       (e.g. '+') that CONDITIONS specifies for the correct column
    """
    # Parse CONDITIONS section (the authoritative grounding filters)
    grounding_filters: list[tuple[str, str, str, str]] = []
    # Match both CONDITIONS: and FILTER VALUES: sections
    for m in re.finditer(
        r'"(\w+)"\."(\w+)":\s*(=|>=|<=|>|<|LIKE|IS NOT NULL)\s*(.*)',
        grounding_context,
    ):
        val = m.group(4).strip()
        # Strip COLLATE NOCASE suffix
        val = re.sub(r'\s+COLLATE\s+NOCASE\s*$', '', val, flags=re.IGNORECASE).strip()
        # Strip surrounding quotes
        val = val.strip("'\"")
        grounding_filters.append((m.group(1), m.group(2), m.group(3), val))
    if not grounding_filters:
        return sql

    try:
        conn = sqlite3.connect(str(db_path), timeout=5)
        tables_cols: dict[str, list[str]] = {}
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
            tname = row[0]
            tables_cols[tname] = [
                r[1] for r in conn.execute(f'PRAGMA table_info("{tname}")').fetchall()
            ]
        conn.close()
    except Exception:
        return sql

    for tbl, col, op, val in grounding_filters:
        if tbl not in tables_cols or col not in tables_cols[tbl]:
            continue
        if op != "=":
            continue

        # Case 2: Grounding says table.col = 'coded_value' but LLM used a DIFFERENT
        # column from the same table with a human-readable value (semantic equivalent).
        # e.g. CONDITIONS says molecule.label = '+' but LLM wrote molecule.storage_label = 'carcinogenic'
        # Detect: grounding column IS NOT in SQL, but another col from same table IS used with = 'something'
        col_in_sql = f'"{col}"' in sql or f'.{col}' in sql or f'."{col}"' in sql
        if not col_in_sql:
            other_cols = [c for c in tables_cols[tbl] if c != col and c.lower() != "_id" and c.lower() != "id"]
            for other_col in other_cols:
                patterns = [
                    rf'"{re.escape(tbl)}"\."{re.escape(other_col)}"\s*=\s*\'[^\']*\'',
                    rf'"{re.escape(other_col)}"\s*=\s*\'[^\']*\'',
                    rf'(?:\w+\.)?"{re.escape(other_col)}"\s*=\s*\'[^\']*\'',
                ]
                for pat in patterns:
                    match = re.search(pat, sql)
                    if match:
                        old_cond = match.group(0)
                        new_cond = f'"{tbl}"."{col}" = \'{val}\''
                        sql = sql.replace(old_cond, new_cond)
                        break
        else:
            # Case 2b: Grounding column IS in SQL but with WRONG value
            # e.g. CONDITIONS says molecule.label = '+' but LLM wrote molecule.label = 'carcinogenic'
            wrong_val_patterns = [
                rf'("{re.escape(tbl)}"\.)?"{re.escape(col)}"\s*=\s*\'([^\']*)\'',
            ]
            for pat in wrong_val_patterns:
                match = re.search(pat, sql)
                if match:
                    used_val = match.group(2) if match.lastindex >= 2 else ""
                    if used_val and used_val != val and used_val.lower() != val.lower():
                        old_cond = match.group(0)
                        new_cond = f'"{tbl}"."{col}" = \'{val}\''
                        sql = sql.replace(old_cond, new_cond)
                    break
    return sql


def _apply_null_guard(sql: str) -> str:
    """Add IS NOT NULL + != '' for columns used in ORDER BY LIMIT 1 or subquery MIN/MAX."""
    guarded = sql

    # Pattern 1: ORDER BY <col> ASC/DESC LIMIT 1
    order_match = re.search(
        r'ORDER\s+BY\s+(\w+(?:\.\w+)?)\s+(ASC|DESC)\s+LIMIT\s+1',
        guarded, re.IGNORECASE
    )
    if order_match:
        col = order_match.group(1)
        guarded = _inject_null_check(guarded, col, order_match.start())

    # Pattern 2: WHERE col = (SELECT MIN/MAX(col) ...) — guard the subquery
    # Handles both quoted ("table"."col") and unquoted (table.col) identifiers
    minmax_match = re.search(
        r'\(\s*SELECT\s+(MIN|MAX)\s*\(\s*("?\w+"?(?:\."?\w+"?)?)\s*\)\s+FROM\s+("?\w+"?)',
        guarded, re.IGNORECASE
    )
    if minmax_match:
        col = minmax_match.group(2)
        minmax_match.group(3)  # table (unused but parsed)
        # Add WHERE col != '' inside the subquery if not already there
        subq_start = minmax_match.start()
        subq_end = guarded.find(")", subq_start + 1)
        if subq_end == -1:
            subq_end = len(guarded)
        # Find the closing paren of the subquery
        depth = 0
        for i in range(subq_start, len(guarded)):
            if guarded[i] == '(':
                depth += 1
            elif guarded[i] == ')':
                depth -= 1
                if depth == 0:
                    subq_end = i
                    break
        subquery = guarded[subq_start:subq_end + 1]
        bare_col = col.split(".")[-1].strip('"') if "." in col else col.strip('"')
        subq_check = subquery.lower().replace('"', '')
        has_not_null = bare_col.lower() + ' is not null' in subq_check
        has_not_empty = bare_col.lower() + " != ''" in subq_check or bare_col.lower() + " <> ''" in subq_check

        if not has_not_null and not has_not_empty:
            if re.search(r'\bWHERE\b', subquery, re.IGNORECASE):
                new_subq = subquery[:-1] + f' AND {col} IS NOT NULL AND {col} != \'\')'
            else:
                new_subq = subquery[:-1] + f' WHERE {col} IS NOT NULL AND {col} != \'\')'
            guarded = guarded[:subq_start] + new_subq + guarded[subq_end + 1:]
        elif has_not_null and not has_not_empty:
            new_subq = subquery[:-1] + f' AND {col} != \'\')'
            guarded = guarded[:subq_start] + new_subq + guarded[subq_end + 1:]
        elif not has_not_null and has_not_empty:
            new_subq = subquery[:-1] + f' AND {col} IS NOT NULL)'
            guarded = guarded[:subq_start] + new_subq + guarded[subq_end + 1:]

    return guarded


def _strip_agg_null_filter(sql: str) -> str:
    """Remove IS NOT NULL filters on columns that are AVG/SUM targets.

    AVG and SUM already skip NULLs. Adding IS NOT NULL on a target column
    restricts the population for OTHER columns in the same query (wrong).
    E.g. SELECT AVG(UpVotes), AVG(Age) ... WHERE Age IS NOT NULL
    incorrectly excludes users without age from the UpVotes average.
    """
    upper = sql.upper()
    if "AVG(" not in upper and "SUM(" not in upper:
        return sql
    # Find columns used in AVG()/SUM()
    agg_cols: set[str] = set()
    for m in re.finditer(r'(?:AVG|SUM)\s*\(\s*(?:\w+\.)?"?(\w+)"?\s*\)', sql, re.IGNORECASE):
        agg_cols.add(m.group(1).lower())
    if not agg_cols:
        return sql
    # Remove "AND col IS NOT NULL" for those columns (only when there are multiple agg targets)
    # Multiple targets = population restriction on one affects the others
    if len(agg_cols) < 2:
        return sql
    result = sql
    for col in agg_cols:
        # Match: AND [alias.]"col" IS NOT NULL (with optional table alias/quotes)
        result = re.sub(
            rf'\s+AND\s+(?:\w+\.)?"?{col}"?\s+IS\s+NOT\s+NULL\b',
            '', result, flags=re.IGNORECASE,
        )
    return result


def _inject_null_check(sql: str, col: str, insert_before: int) -> str:
    """Inject IS NOT NULL AND != '' for col before the given position."""
    if re.search(rf'{re.escape(col)}\s+IS\s+NOT\s+NULL', sql, re.IGNORECASE):
        if re.search(rf"{re.escape(col)}\s*!=\s*''", sql, re.IGNORECASE):
            return sql
        prefix = sql[:insert_before].rstrip()
        suffix = sql[insert_before:]
        return f"{prefix} AND {col} != '' {suffix}"
    prefix = sql[:insert_before].rstrip()
    suffix = sql[insert_before:]
    if re.search(r'\bWHERE\b', prefix, re.IGNORECASE):
        return f"{prefix} AND {col} IS NOT NULL AND {col} != '' {suffix}"
    else:
        return f"{prefix} WHERE {col} IS NOT NULL AND {col} != '' {suffix}"


