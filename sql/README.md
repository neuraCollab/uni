# SQL — Interview Prep

Concise, cram-friendly SQL notes for interviews. Standard SQL, PostgreSQL-flavored where dialect matters (most common in interviews), with MySQL/SQL Server/SQLite differences called out where they commonly trip people up.

## Cheat sheet: query execution order

The single most commonly tested SQL concept. SQL is written top-down but **executed** in a fixed logical order:

```
FROM  →  JOIN/ON  →  WHERE  →  GROUP BY  →  HAVING  →  SELECT  →  DISTINCT  →  ORDER BY  →  LIMIT
```

| Step | Clause | What happens |
|---|---|---|
| 1 | `FROM` / `JOIN` | Build the working row set from source table(s) |
| 2 | `WHERE` | Filter individual rows (before aggregation) |
| 3 | `GROUP BY` | Bucket rows into groups |
| 4 | `HAVING` | Filter groups (after aggregation — can use `COUNT()`, `SUM()`, etc.) |
| 5 | `SELECT` | Compute output columns/aliases |
| 6 | `DISTINCT` | Remove duplicate output rows |
| 7 | `ORDER BY` | Sort the result (can reference `SELECT` aliases) |
| 8 | `LIMIT`/`OFFSET` | Cap/page the final result |

**Why this matters**: it's why `WHERE` can't reference a `SELECT` alias or an aggregate (`WHERE` runs before both exist), why `HAVING` can use aggregates but `WHERE` can't, and why you can't filter a window function's result in the same query's `WHERE` (must wrap in a subquery/CTE). Full detail: [`query-execution-order.md`](./query-execution-order.md).

## Contents

| File | Covers |
|---|---|
| [`select-where-groupby.md`](./select-where-groupby.md) | SELECT/WHERE/GROUP BY/HAVING/ORDER BY, NULL three-valued logic, `COALESCE`/`NULLIF`, aggregation functions, `COUNT(*)` vs `COUNT(col)`, WHERE vs HAVING, DISTINCT |
| [`joins.md`](./joins.md) | INNER/LEFT/RIGHT/FULL OUTER/CROSS/SELF joins with a worked example table, join vs subquery tradeoffs, "no manager" pattern, NULLs breaking equality joins |
| [`window-functions.md`](./window-functions.md) | `OVER (PARTITION BY ... ORDER BY ...)`, ROW_NUMBER/RANK/DENSE_RANK with ties, LAG/LEAD, running totals, moving averages, top-N-per-group pattern, NTILE |
| [`ctes-subqueries.md`](./ctes-subqueries.md) | CTEs vs subqueries vs temp tables, recursive CTEs (org-chart traversal), correlated vs uncorrelated subqueries, EXISTS vs IN vs JOIN, the `NOT IN` + NULL trap |
| [`query-execution-order.md`](./query-execution-order.md) | The logical execution order in full detail, why aliases/aggregates are/aren't visible where they are |
| [`indexes-optimization.md`](./indexes-optimization.md) | B-tree indexes, when they help/don't (low cardinality, leading wildcards, functions on columns), composite index order, covering indexes, EXPLAIN ANALYZE, N+1 problem, over-indexing |
| [`interview-problems.md`](./interview-problems.md) | 8 classic problems with pattern + solution: second-highest salary, highest-avg-salary department, login streaks, running totals, duplicate detection, employees > manager salary, median, pivot |

## Suggested read order

1. [`select-where-groupby.md`](./select-where-groupby.md) — fundamentals
2. [`joins.md`](./joins.md)
3. [`query-execution-order.md`](./query-execution-order.md) — ties everything together
4. [`window-functions.md`](./window-functions.md)
5. [`ctes-subqueries.md`](./ctes-subqueries.md)
6. [`indexes-optimization.md`](./indexes-optimization.md)
7. [`interview-problems.md`](./interview-problems.md) — practice applying all of the above

## Top gotchas (quick recall before an interview)

- `WHERE col = NULL` never matches — use `IS NULL`. (three-valued logic)
- `NOT IN (subquery)` silently returns zero rows if the subquery can produce a `NULL`. Use `NOT EXISTS`.
- Can't filter a window function alias in the same `SELECT`'s `WHERE` — wrap in a CTE/subquery.
- `AVG()`/`SUM()` ignore NULLs — `AVG` divides by the non-NULL count, not the total row count.
- Filtering a `LEFT JOIN`'s right-table column in `WHERE` (instead of `ON`) silently turns it back into an `INNER JOIN`.
- `RANK()` leaves gaps after ties, `DENSE_RANK()` doesn't, `ROW_NUMBER()` never ties.
- A function applied to an indexed column (`LOWER(col)`) disables plain index usage — needs an expression index.
