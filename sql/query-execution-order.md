# Query Execution Order

The single most commonly tested SQL concept. SQL is declarative — you write clauses in one order, but the engine *logically* processes them in a different, fixed order. Understanding this order explains almost every "why doesn't this work" SQL question.

## Written order vs. logical execution order

```sql
SELECT   ...
FROM     ...
JOIN     ...
WHERE    ...
GROUP BY ...
HAVING   ...
ORDER BY ...
LIMIT    ...
```

Logical (conceptual) execution order:

| # | Clause | What happens |
|---|--------|---------------|
| 1 | `FROM` | Identify source table(s) |
| 2 | `JOIN` (+ `ON`) | Combine rows from multiple tables |
| 3 | `WHERE` | Filter individual rows (pre-aggregation) |
| 4 | `GROUP BY` | Bucket remaining rows into groups |
| 5 | `HAVING` | Filter groups (post-aggregation) |
| 6 | `SELECT` | Compute output expressions / aliases |
| 7 | `DISTINCT` | Remove duplicate result rows |
| 8 | `WINDOW functions` | Evaluated using `SELECT`-list expressions, after `HAVING`, before `ORDER BY`/`DISTINCT` in most engines (conceptually right after `SELECT`) |
| 9 | `ORDER BY` | Sort the result set |
| 10 | `LIMIT` / `OFFSET` | Cut down to N rows |

> Mnemonic: **F**rom **J**oin **W**here **G**roup **H**aving **S**elect **D**istinct **O**rder **L**imit — "**F**at **J**oe **W**ants **G**reat **H**am **S**andwiches, **D**efinitely **O**rdering **L**ots".

This is why, engine-internally, `FROM`/`JOIN` build the working row set first, and `SELECT` — despite being written first — is evaluated almost last.

## Why this matters: concrete consequences

### 1. You can't reference a `SELECT` alias in `WHERE`, but you can in `ORDER BY`

```sql
-- FAILS: alias `total` doesn't exist yet when WHERE runs (step 3, before SELECT at step 6)
SELECT price * qty AS total
FROM orders
WHERE total > 100;          -- ERROR: column "total" does not exist

-- WORKS: repeat the expression instead
SELECT price * qty AS total
FROM orders
WHERE price * qty > 100;

-- WORKS: ORDER BY runs after SELECT (step 9 > step 6), so the alias exists
SELECT price * qty AS total
FROM orders
ORDER BY total;
```

(MySQL is a notable exception — it lets you use a `SELECT` alias in `HAVING`/`GROUP BY` as a convenience, deviating from the standard. Don't rely on this if you need portable SQL.)

### 2. Why `HAVING` can use aggregates but `WHERE` can't

`WHERE` (step 3) runs *before* `GROUP BY` (step 4) even exists — there's no aggregate to filter on yet, only raw rows. `HAVING` (step 5) runs *after* grouping/aggregation, so `SUM()`, `COUNT()`, `AVG()`, etc. are already computed and available to filter on.

```sql
-- WRONG: aggregate not yet computed at WHERE-time
SELECT dept, COUNT(*) FROM employees WHERE COUNT(*) > 5 GROUP BY dept;  -- ERROR

-- RIGHT
SELECT dept, COUNT(*) AS cnt
FROM employees
GROUP BY dept
HAVING COUNT(*) > 5;
```

### 3. Window functions see `WHERE`-filtered but not `HAVING`-or-`LIMIT`-filtered rows

Window functions conceptually run after `WHERE`/`GROUP BY`/`HAVING`, but before `ORDER BY`/`LIMIT`. That's why you **cannot filter on a window function's result in the same `SELECT`'s `WHERE`** — you must wrap it in a subquery/CTE (see [`window-functions.md`](./window-functions.md)).

```sql
-- FAILS: ROW_NUMBER() isn't computed yet when WHERE runs
SELECT *, ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC) AS rn
FROM employees
WHERE rn = 1;   -- ERROR: column "rn" does not exist

-- RIGHT: filter in an outer query
SELECT * FROM (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC) AS rn
  FROM employees
) t
WHERE rn = 1;
```

### 4. `DISTINCT` runs after `SELECT`, so it dedupes on the *output* expressions, not the source columns

```sql
SELECT DISTINCT dept FROM employees;  -- dedupes on the computed dept column, post-SELECT
```

## Compact diagram

```
FROM  →  JOIN/ON  →  WHERE  →  GROUP BY  →  HAVING  →  SELECT  →  DISTINCT  →  ORDER BY  →  LIMIT/OFFSET
 |          |           |          |           |          |          |            |            |
build     combine    filter      bucket     filter     compute    dedupe        sort       page/cap
rows      rows       rows        rows       groups     output      rows         result      result
```

## Interview questions

- **Q: Why can't you put an aggregate function in `WHERE`?**
  A: `WHERE` executes before `GROUP BY`, so aggregates haven't been computed yet. Use `HAVING`.

- **Q: In what order does the database actually execute a query with a `JOIN`, `WHERE`, `GROUP BY`, and `ORDER BY`?**
  A: `FROM`/`JOIN` → `WHERE` → `GROUP BY` → `HAVING` → `SELECT` → `DISTINCT` → `ORDER BY` → `LIMIT`.

- **Q: Can you use a column alias defined in `SELECT` inside a `GROUP BY`?**
  A: Standard SQL: no (same reasoning as `WHERE`). PostgreSQL/MySQL/SQLite allow it as a convenience; SQL Server does not. Don't rely on this cross-dialect — repeat the expression or reference the ordinal position instead.

- **Q: Does `LIMIT` run before or after `ORDER BY`?**
  A: After — otherwise "top 10 by revenue" would be meaningless (it would just be an arbitrary 10 rows, then sorted).

## See also

- [`select-where-groupby.md`](./select-where-groupby.md) — WHERE vs HAVING in depth
- [`window-functions.md`](./window-functions.md) — the ROW_NUMBER-in-WHERE trap
- [`ctes-subqueries.md`](./ctes-subqueries.md) — using a CTE/subquery to work around execution-order limits
