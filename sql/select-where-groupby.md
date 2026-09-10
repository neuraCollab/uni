# SELECT / WHERE / GROUP BY / HAVING / ORDER BY

## What

The core "read data" clauses of SQL. See [`query-execution-order.md`](./query-execution-order.md) for the logical order they run in — `FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY` — which explains most of the gotchas below.

## SELECT / WHERE basics

```sql
SELECT id, name, salary
FROM employees
WHERE department = 'Engineering'
  AND salary > 80000
ORDER BY salary DESC
LIMIT 10;
```

- `WHERE` filters **individual rows** before any grouping happens.
- Boolean operators: `AND`, `OR`, `NOT`, with standard precedence (`AND` binds tighter than `OR` — parenthesize when mixing them).
- Pattern matching: `LIKE '%foo%'` (case-sensitive in Postgres, case-insensitive in MySQL/SQL Server by default), `ILIKE` for case-insensitive (Postgres-specific).
- Range/set filters: `BETWEEN a AND b` (inclusive both ends), `IN (1,2,3)`, `NOT IN (...)`.

## NULL handling — three-valued logic

SQL uses **three-valued logic**: `TRUE`, `FALSE`, and `UNKNOWN`. Any comparison involving `NULL` evaluates to `UNKNOWN`, not `TRUE` or `FALSE`. Rows where the `WHERE`/`HAVING` predicate evaluates to `UNKNOWN` are **excluded** (treated like `FALSE` for filtering purposes, but not for `NOT`).

```sql
NULL = NULL        -- UNKNOWN (not TRUE!)
NULL <> NULL        -- UNKNOWN
NULL = 5            -- UNKNOWN
NOT UNKNOWN         -- still UNKNOWN
```

**`IS NULL` vs `= NULL`** — a top interview gotcha:

```sql
-- WRONG: always returns zero rows, because col = NULL is UNKNOWN, never TRUE
SELECT * FROM t WHERE col = NULL;

-- RIGHT
SELECT * FROM t WHERE col IS NULL;
SELECT * FROM t WHERE col IS NOT NULL;
```

**`COALESCE(a, b, c, ...)`** — returns the first non-NULL argument. Common for defaulting:

```sql
SELECT COALESCE(nickname, first_name, 'Unknown') AS display_name FROM users;
```

**`NULLIF(a, b)`** — returns `NULL` if `a = b`, else returns `a`. Common for avoiding division-by-zero:

```sql
SELECT total / NULLIF(count, 0) AS avg_per_item FROM stats;  -- NULL instead of a divide-by-zero error
```

**`NOT IN` + NULL trap** (also covered in [`ctes-subqueries.md`](./ctes-subqueries.md)):

```sql
-- If subquery returns ANY NULL, this returns ZERO rows — silently!
SELECT * FROM employees
WHERE dept_id NOT IN (SELECT dept_id FROM departments WHERE budget IS NULL);
```
`NOT IN (1, 2, NULL)` becomes `col <> 1 AND col <> 2 AND col <> NULL` — the last comparison is `UNKNOWN`, which poisons the whole `AND` chain to `UNKNOWN`/`FALSE`. Prefer `NOT EXISTS` for NULL-safety.

## Aggregation functions

| Function | Notes |
|---|---|
| `COUNT(*)` | Counts all rows, including ones with NULL columns |
| `COUNT(col)` | Counts only rows where `col IS NOT NULL` |
| `COUNT(DISTINCT col)` | Counts distinct non-NULL values |
| `SUM(col)` | Ignores NULLs; `SUM` of all-NULL group is `NULL`, not 0 |
| `AVG(col)` | Ignores NULLs in both numerator and denominator (i.e., averages over non-NULL rows only) |
| `MIN` / `MAX` | Ignore NULLs |

```sql
-- Table `t`: values (1, NULL, 3)
SELECT COUNT(*)        FROM t;  -- 3
SELECT COUNT(val)      FROM t;  -- 2  (NULL not counted)
SELECT SUM(val)        FROM t;  -- 4  (NULL skipped, not treated as 0)
SELECT AVG(val)        FROM t;  -- 2  (= 4 / 2, NOT 4 / 3)
```

This AVG-ignores-NULL behavior is a classic interview trap: people expect `AVG` to divide by total row count, but it divides only by non-NULL count.

## GROUP BY

Buckets rows sharing the same value(s) into groups, so aggregate functions compute per-group instead of over the whole table.

```sql
SELECT department, COUNT(*) AS headcount, AVG(salary) AS avg_salary
FROM employees
GROUP BY department;
```

**Rule**: every non-aggregated column in `SELECT` must appear in `GROUP BY` (strict in Postgres/SQL Server; MySQL historically allowed "loose" grouping — avoid relying on that).

Grouping by multiple columns:

```sql
SELECT department, job_title, COUNT(*)
FROM employees
GROUP BY department, job_title;
```

## WHERE vs HAVING — the key distinction

- `WHERE` filters **rows** *before* aggregation.
- `HAVING` filters **groups** *after* aggregation — it can reference aggregate functions; `WHERE` cannot (see [`query-execution-order.md`](./query-execution-order.md)).

Concrete example — "departments with more than 5 highly-paid (>\$100k) employees":

```sql
SELECT department, COUNT(*) AS high_earners
FROM employees
WHERE salary > 100000        -- filter rows first: only high earners considered
GROUP BY department
HAVING COUNT(*) > 5;         -- then filter groups: only depts with >5 such employees
```

If you swapped the logic (`HAVING salary > 100000`) it would error — `salary` isn't an aggregate and isn't in `GROUP BY`. If you tried `WHERE COUNT(*) > 5` it would error — aggregates aren't available yet at `WHERE`-time.

You can combine both: `WHERE` cheaply discards irrelevant rows before the (more expensive) grouping/aggregation work, `HAVING` then filters the resulting groups.

## ORDER BY

```sql
SELECT name, salary FROM employees ORDER BY salary DESC, name ASC;
```

- Default is `ASC`. NULLs sort last by default in Postgres for `ASC` (`NULLS LAST`), but **first** in MySQL for `ASC`. Use explicit `NULLS FIRST`/`NULLS LAST` (Postgres/SQL standard) when it matters.
- Can order by column position (`ORDER BY 2`), alias, or expression not in `SELECT`.

## DISTINCT

```sql
SELECT DISTINCT department FROM employees;
SELECT DISTINCT department, job_title FROM employees;  -- distinct combinations
```

`DISTINCT` applies to the *entire row* of selected columns, not a single one — `SELECT DISTINCT a, b` doesn't mean "distinct a, paired with any b."

`COUNT(DISTINCT col)` counts distinct non-NULL values of `col`.

## Common mistakes

- Using `= NULL` instead of `IS NULL`.
- Forgetting `SUM`/`AVG` skip NULLs (can silently skew results if NULLs mean "0" in your domain — use `COALESCE(col, 0)`).
- Trying to filter an aggregate in `WHERE` instead of `HAVING`.
- Using `NOT IN` against a column/subquery that can contain NULLs.
- Assuming `GROUP BY` output order is guaranteed without an explicit `ORDER BY` (it isn't, in any engine).

## Interview questions

1. What's the difference between `COUNT(*)` and `COUNT(column)`?
2. Why does `WHERE col = NULL` never match anything?
3. Write a query for "departments with average salary above 90000, considering only active employees."
4. What does `AVG(col)` return if `col` is NULL for every row in the group?
5. Explain why `WHERE` can't use aggregate functions but `HAVING` can.

```sql
-- Answer to #3
SELECT department, AVG(salary) AS avg_salary
FROM employees
WHERE is_active = TRUE
GROUP BY department
HAVING AVG(salary) > 90000;
```

## See also

- [`query-execution-order.md`](./query-execution-order.md)
- [`joins.md`](./joins.md)
- [`ctes-subqueries.md`](./ctes-subqueries.md) — NOT IN / NULL trap in subqueries
