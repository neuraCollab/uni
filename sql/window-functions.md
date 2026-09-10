# Window Functions

## What

A window function computes a value across a *set of rows related to the current row* (the "window") without collapsing rows into groups, unlike `GROUP BY`. Every input row stays in the output, each with an extra computed column.

## Syntax

```sql
<function>() OVER (
  [PARTITION BY col1, col2, ...]
  [ORDER BY col3, ...]
  [ROWS|RANGE BETWEEN ... AND ...]
)
```

- `PARTITION BY` — splits rows into groups (like `GROUP BY`, but doesn't collapse rows). Omit it to treat the whole result set as one partition.
- `ORDER BY` (inside `OVER`) — defines row order within each partition, required for ranking/offset functions and for running-total framing.
- Frame clause (`ROWS BETWEEN ...`) — narrows the window further (e.g., "3 rows before current"). Defaults to `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` when `ORDER BY` is present.

## Sample data — `sales`

| id | rep   | region | amount | sale_date  |
|----|-------|--------|--------|------------|
| 1  | Alice | East   | 100    | 2024-01-01 |
| 2  | Bob   | East   | 100    | 2024-01-02 |
| 3  | Carol | East   | 80     | 2024-01-03 |
| 4  | Dave  | West   | 200    | 2024-01-01 |
| 5  | Eve   | West   | 150    | 2024-01-02 |

## ROW_NUMBER vs RANK vs DENSE_RANK — ties matter

```sql
SELECT rep, region, amount,
  ROW_NUMBER() OVER (PARTITION BY region ORDER BY amount DESC) AS rn,
  RANK()       OVER (PARTITION BY region ORDER BY amount DESC) AS rnk,
  DENSE_RANK() OVER (PARTITION BY region ORDER BY amount DESC) AS drnk
FROM sales;
```

East partition (Alice=100, Bob=100, Carol=80 — Alice and Bob **tie**):

| rep   | amount | rn | rnk | drnk |
|-------|--------|----|-----|------|
| Alice | 100    | 1  | 1   | 1    |
| Bob   | 100    | 2  | 1   | 1    |
| Carol | 80     | 3  | 3   | 2    |

- **`ROW_NUMBER`**: always unique, arbitrary tiebreak among ties (1, 2, 3, ...) — order between Alice/Bob is not deterministic unless `ORDER BY` fully disambiguates.
- **`RANK`**: ties get the same rank, but the *next* rank skips ahead by the number of tied rows (1, 1, 3 — "3" is skipped nowhere, it's the count so far).
- **`DENSE_RANK`**: ties get the same rank, next rank is always +1 with no gaps (1, 1, 2).

Interview one-liner: `RANK` leaves gaps after ties, `DENSE_RANK` doesn't, `ROW_NUMBER` never ties at all.

## LAG / LEAD

Access a prior/following row's value within the same partition without a self-join.

```sql
SELECT rep, sale_date, amount,
  LAG(amount) OVER (PARTITION BY rep ORDER BY sale_date) AS prev_amount,
  LEAD(amount) OVER (PARTITION BY rep ORDER BY sale_date) AS next_amount,
  amount - LAG(amount) OVER (PARTITION BY rep ORDER BY sale_date) AS change
FROM sales;
```

`LAG(col, n, default)` / `LEAD(col, n, default)` — `n` (default 1) rows back/forward, `default` value if out of range (else `NULL`).

## Running totals

```sql
SELECT rep, sale_date, amount,
  SUM(amount) OVER (PARTITION BY rep ORDER BY sale_date
                     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_total
FROM sales;
```

With `ORDER BY` present and no explicit frame, most engines default to exactly this frame (`RANGE UNBOUNDED PRECEDING TO CURRENT ROW`), so `SUM(amount) OVER (PARTITION BY rep ORDER BY sale_date)` alone often gives the same result — but **be explicit with `ROWS BETWEEN`** to avoid `RANGE` vs `ROWS` surprises when there are duplicate `ORDER BY` values (RANGE treats peer rows — same order-by value — as a single step, inflating running totals across ties; ROWS treats each physical row individually).

## Moving average

```sql
SELECT rep, sale_date, amount,
  AVG(amount) OVER (PARTITION BY rep ORDER BY sale_date
                     ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS moving_avg_3
FROM sales;
```
3-row moving average (current + 2 preceding rows).

## Top-N per group — the classic pattern

**Goal**: top 1 sale per region.

```sql
-- WRONG — cannot filter on a window function in the same SELECT's WHERE
SELECT rep, region, amount,
  ROW_NUMBER() OVER (PARTITION BY region ORDER BY amount DESC) AS rn
FROM sales
WHERE rn = 1;   -- ERROR: column "rn" does not exist
```

This fails because of **query execution order**: `WHERE` runs before window functions are computed (see [`query-execution-order.md`](./query-execution-order.md)). Window functions are logically evaluated alongside/after `SELECT`, so their aliases aren't visible to `WHERE` in the same query level.

```sql
-- RIGHT — wrap in a subquery/CTE, filter in the outer query
WITH ranked AS (
  SELECT rep, region, amount,
    ROW_NUMBER() OVER (PARTITION BY region ORDER BY amount DESC) AS rn
  FROM sales
)
SELECT rep, region, amount
FROM ranked
WHERE rn = 1;
```

Postgres shortcut for this exact "top 1 per group" case: `DISTINCT ON`:
```sql
SELECT DISTINCT ON (region) region, rep, amount
FROM sales
ORDER BY region, amount DESC;
```
(Postgres-specific, not standard SQL.)

This top-N pattern (ROW_NUMBER + filter in outer query) is one of the single most-asked SQL interview questions — memorize it.

## NTILE — bucketing into N groups

```sql
SELECT rep, amount,
  NTILE(4) OVER (ORDER BY amount DESC) AS quartile
FROM sales;
```
Splits rows as evenly as possible into `N` buckets (e.g., quartiles/deciles). If rows don't divide evenly, earlier buckets get the extra rows.

## Common mistakes

- Trying to filter on a window function alias in the same `SELECT`'s `WHERE`/`HAVING` — must use an outer query/CTE.
- Using `RANK` when you actually want `ROW_NUMBER` (or vice versa) — ties change the semantics of "top N."
- Forgetting `PARTITION BY` — omitting it treats the *entire result set* as one window, which is sometimes intentional (e.g., overall running total) but often a bug.
- Not specifying an explicit frame (`ROWS BETWEEN ...`) when duplicate `ORDER BY` values exist and the default `RANGE` framing isn't what you want.
- Confusing window functions with `GROUP BY` — window functions don't reduce row count; `GROUP BY` does.

## Interview questions

1. What's the difference between `RANK()` and `DENSE_RANK()` when there are ties?
2. Why can't you write `WHERE row_number() = 1` directly? How do you work around it?
3. Write a query to find the 2nd highest sale per region.
4. How would you compute a 7-day moving average of daily sales?
5. What does `LAG(amount, 1, 0)` do differently from `LAG(amount)`?

```sql
-- Answer to #3: 2nd highest sale per region
WITH ranked AS (
  SELECT rep, region, amount,
    DENSE_RANK() OVER (PARTITION BY region ORDER BY amount DESC) AS rnk
  FROM sales
)
SELECT rep, region, amount FROM ranked WHERE rnk = 2;
```

## See also

- [`query-execution-order.md`](./query-execution-order.md) — why the top-N-per-group trap happens
- [`ctes-subqueries.md`](./ctes-subqueries.md)
- [`interview-problems.md`](./interview-problems.md) — running totals, second-highest salary, streaks
