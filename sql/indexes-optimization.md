# Indexes & Query Optimization

## What

An index is an auxiliary data structure that lets the database find rows without scanning the whole table — trading extra storage and slower writes for faster reads.

## How indexes work (B-tree default)

Most general-purpose indexes (Postgres/MySQL InnoDB/SQL Server default) are **B-trees**: a balanced tree structure keeping keys sorted, giving $O(\log n)$ lookups, range scans, and ordered traversal.

```sql
CREATE INDEX idx_employees_dept ON employees(department);
```

This builds a sorted structure of `department` values, each pointing to the corresponding row(s) (via row pointer / clustered key, depending on engine).

## Where indexes help

- **`WHERE` equality/range lookups**: `WHERE department = 'Sales'`, `WHERE salary BETWEEN 50000 AND 90000`.
- **`JOIN` conditions**: an index on the join column (especially the foreign-key side) lets the engine avoid a full scan of one side.
- **`ORDER BY`**: an index matching the sort column(s) lets the engine read rows already sorted, avoiding an explicit sort step.
- **`GROUP BY`**: can sometimes use a sorted index to group without a separate sort/hash step.

## Where indexes DON'T help

- **Low-cardinality columns** (few distinct values, e.g. a boolean `is_active` flag, or `gender`): the planner often prefers a sequential scan anyway, since filtering doesn't eliminate enough rows to justify random-access index lookups.
- **Leading wildcard `LIKE`**: `WHERE name LIKE '%smith'` can't use a standard B-tree index (the sorted prefix is unknown). `WHERE name LIKE 'smith%'` *can* use it (known prefix). Full leading-wildcard search needs a different structure (trigram/full-text index — e.g. Postgres `pg_trgm` + GIN index).
- **Functions/expressions applied to the indexed column**: `WHERE LOWER(email) = 'x@y.com'` can't use a plain index on `email` — the stored values aren't lowercased. Fix with a **functional/expression index**: `CREATE INDEX ON users (LOWER(email));`, or store normalized data.
- **Implicit type casts**: comparing a text column to a number (or vice versa) can silently disable index use in some engines.
- **Very small tables**: a full scan is often cheaper than the overhead of an index lookup — the optimizer will (correctly) ignore the index.
- **High-cardinality writes with poor selectivity in the query**: if a query is expected to return a large fraction of the table, a sequential scan beats many random index lookups.

## Composite (multi-column) index order matters

```sql
CREATE INDEX idx_orders_cust_date ON orders(customer_id, order_date);
```

- Usable for: `WHERE customer_id = X`, and `WHERE customer_id = X AND order_date > Y` (leftmost-prefix rule).
- **Not** usable (as a direct index scan) for `WHERE order_date > Y` alone — the leading column (`customer_id`) isn't constrained, so the sorted structure doesn't help skip to relevant `order_date` ranges.
- Rule of thumb: put the column used in equality filters first, range-filtered/sort columns after; put higher-selectivity (more distinct values) columns earlier, generally.

## Covering indexes

An index that contains **all** columns a query needs — the engine can answer the query from the index alone, never touching the actual table (heap) rows.

```sql
CREATE INDEX idx_orders_covering ON orders(customer_id, order_date) INCLUDE (total_amount);  -- Postgres/SQL Server INCLUDE
SELECT order_date, total_amount FROM orders WHERE customer_id = 5;  -- can be an index-only scan
```

`INCLUDE` (Postgres 11+/SQL Server) adds columns to the index leaf pages without making them part of the sort key — cheaper than including them in the key itself.

## EXPLAIN / EXPLAIN ANALYZE

```sql
EXPLAIN ANALYZE SELECT * FROM employees WHERE department = 'Sales';
```

- `EXPLAIN` — shows the planned execution strategy (estimated costs), without running the query.
- `EXPLAIN ANALYZE` — actually **runs** the query and shows real timing/row counts alongside the plan (use with care on writes/expensive queries — it executes them).

Key plan node types to recognize:

| Plan node | Meaning |
|---|---|
| **Seq Scan** (Postgres) / **Table Scan** | Reads every row in the table — no index used |
| **Index Scan** | Uses an index to find matching rows, then fetches each row from the table (heap) |
| **Index-Only Scan** | Answers entirely from the index (covering index) — no heap access needed, fastest |
| **Bitmap Heap/Index Scan** (Postgres) | Builds a bitmap of matching pages from the index, then fetches in physical order — good for moderately selective queries |
| **Nested Loop / Hash Join / Merge Join** | Join strategies — hash join for large unsorted sets, merge join for pre-sorted inputs, nested loop for small outer sets or index-assisted lookups |

A `Seq Scan` on a large table where you expected an `Index Scan` is the #1 thing to investigate — usually means: missing index, index not usable due to a function/cast, or the optimizer decided the scan would return too large a fraction of rows to bother with the index.

## N+1 query problem (ORM/application context)

Fetching a list, then issuing one additional query *per row* to fetch related data — instead of one batched query (or a join).

```python
# N+1: 1 query for orders, then N queries (one per order) for its customer
orders = db.query("SELECT * FROM orders")
for o in orders:
    customer = db.query(f"SELECT * FROM customers WHERE id = {o.customer_id}")  # N queries!
```

Fix: batch-fetch with a single query (`WHERE id IN (...)`), or use a `JOIN`, or use the ORM's eager-loading feature (`select_related`/`prefetch_related` in Django, `joinedload` in SQLAlchemy).

```sql
-- One query instead of N
SELECT * FROM customers WHERE id IN (1, 2, 3, ...);
```

This is a very common interview/system-design question because it's a real, frequent production performance bug.

## Common mistake: over-indexing

Every index:
- Speeds up reads matching its columns.
- **Slows down every `INSERT`/`UPDATE`/`DELETE`** on that table, since each index must also be updated.
- Consumes disk space and cache/buffer pool memory (competing with actual table data for cache).

Rule of thumb: index columns that are frequently filtered/joined/sorted on and selective enough to matter; avoid indexing every column "just in case," especially on write-heavy tables. Periodically audit for unused indexes (`pg_stat_user_indexes` in Postgres) and drop them.

## Common mistakes summary

- Expecting an index to help with a low-selectivity `WHERE` clause.
- Wrapping an indexed column in a function (`LOWER()`, date truncation, etc.) without a matching expression index.
- Leading-wildcard `LIKE '%x'` searches expecting index usage.
- Composite index column order not matching query patterns (not leading with the equality-filtered column).
- Over-indexing a write-heavy table, tanking `INSERT`/`UPDATE` throughput.
- Not running `EXPLAIN ANALYZE` before assuming a query is "slow because SQL is slow" — usually it's a missing/misused index.

## Interview questions

1. Why doesn't an index help with `WHERE is_deleted = false` on a mostly-active table?
2. What's the difference between an index scan and an index-only scan?
3. Why does column order matter in a composite index?
4. What is the N+1 query problem, and how do you fix it?
5. Why would adding more indexes ever make a system *slower*?
6. Why can't `WHERE UPPER(name) = 'ALICE'` use a plain index on `name`?

## See also

- [`query-execution-order.md`](./query-execution-order.md)
- [`joins.md`](./joins.md) — indexes on join columns
- [`interview-problems.md`](./interview-problems.md)
