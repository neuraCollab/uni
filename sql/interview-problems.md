# Classic SQL Interview Problems

Concise, pattern-first. Each problem: statement, technique, solution. See linked files for deeper explanation of each technique.

## 1. Second-highest salary

**Problem**: Find the second-highest salary in `employees(id, name, salary)`. Handle ties/missing gracefully (return NULL if it doesn't exist).

**Pattern**: `DENSE_RANK()` (handles ties correctly) or `LIMIT/OFFSET` with `DISTINCT`, or a correlated subquery for portability.

```sql
-- Window function approach (best: handles ties as "one" rank)
SELECT salary AS second_highest
FROM (
  SELECT salary, DENSE_RANK() OVER (ORDER BY salary DESC) AS rnk
  FROM employees
) t
WHERE rnk = 2;

-- Portable subquery approach (no window functions needed)
SELECT MAX(salary) AS second_highest
FROM employees
WHERE salary < (SELECT MAX(salary) FROM employees);

-- OFFSET approach (careful: doesn't dedupe ties without DISTINCT)
SELECT DISTINCT salary FROM employees ORDER BY salary DESC LIMIT 1 OFFSET 1;
```

See: [`window-functions.md`](./window-functions.md)

## 2. Department with the highest average salary

**Problem**: Given `employees(id, name, dept_id, salary)` and `departments(id, name)`, find the department with the highest average salary.

**Pattern**: `GROUP BY` + `HAVING`/`ORDER BY LIMIT`, or window function to handle ties.

```sql
SELECT d.name, AVG(e.salary) AS avg_salary
FROM employees e
JOIN departments d ON d.id = e.dept_id
GROUP BY d.name
ORDER BY avg_salary DESC
LIMIT 1;

-- Handling ties (multiple departments sharing the max average)
WITH dept_avg AS (
  SELECT d.name, AVG(e.salary) AS avg_salary
  FROM employees e JOIN departments d ON d.id = e.dept_id
  GROUP BY d.name
)
SELECT name, avg_salary FROM dept_avg
WHERE avg_salary = (SELECT MAX(avg_salary) FROM dept_avg);
```

See: [`select-where-groupby.md`](./select-where-groupby.md), [`joins.md`](./joins.md)

## 3. Consecutive-days login streak

**Problem**: Given `logins(user_id, login_date)`, find the longest run of consecutive-day logins per user.

**Pattern**: The "gaps and islands" trick — subtract a `ROW_NUMBER()` from the date; consecutive dates produce the same "group key" (since both advance by 1 together).

```sql
WITH numbered AS (
  SELECT user_id, login_date,
    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY login_date) AS rn
  FROM (SELECT DISTINCT user_id, login_date FROM logins) d
),
grouped AS (
  SELECT user_id, login_date,
    login_date - (rn * INTERVAL '1 day') AS grp   -- constant within each consecutive run
  FROM numbered
)
SELECT user_id, COUNT(*) AS streak_length, MIN(login_date) AS streak_start, MAX(login_date) AS streak_end
FROM grouped
GROUP BY user_id, grp
ORDER BY user_id, streak_length DESC;
```
(`login_date - rn` is constant for a consecutive run because both the date and the row number increase by exactly 1 per day within the run — MySQL: use `DATE_SUB(login_date, INTERVAL rn DAY)`.)

See: [`window-functions.md`](./window-functions.md)

## 4. Running total per customer

**Problem**: Given `orders(id, customer_id, order_date, amount)`, compute a running total of `amount` per customer ordered by date.

**Pattern**: `SUM() OVER (PARTITION BY ... ORDER BY ... ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)`.

```sql
SELECT customer_id, order_date, amount,
  SUM(amount) OVER (
    PARTITION BY customer_id ORDER BY order_date
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  ) AS running_total
FROM orders;
```

See: [`window-functions.md`](./window-functions.md)

## 5. Duplicate row detection / deletion

**Problem**: Given `users(id, email, ...)` with duplicate `email` values (different `id`s), find and delete duplicates, keeping the earliest (lowest `id`) row per email.

**Pattern**: `GROUP BY ... HAVING COUNT(*) > 1` to detect; `ROW_NUMBER()` partition to delete.

```sql
-- Detect
SELECT email, COUNT(*) AS cnt
FROM users
GROUP BY email
HAVING COUNT(*) > 1;

-- Delete duplicates, keeping the lowest id per email
DELETE FROM users
WHERE id NOT IN (
  SELECT MIN(id) FROM users GROUP BY email
);

-- Equivalent, window-function version (often needed when there's no single unique key,
-- e.g. duplicates defined by multiple columns)
WITH ranked AS (
  SELECT id, ROW_NUMBER() OVER (PARTITION BY email ORDER BY id) AS rn
  FROM users
)
DELETE FROM users WHERE id IN (SELECT id FROM ranked WHERE rn > 1);
```

See: [`select-where-groupby.md`](./select-where-groupby.md), [`window-functions.md`](./window-functions.md)

## 6. Employees earning more than their manager

**Problem**: Given `employees(id, name, salary, manager_id)`, find employees who earn more than their direct manager.

**Pattern**: Self-join on `manager_id = id`, compare salaries.

```sql
SELECT e.name AS employee, e.salary, m.name AS manager, m.salary AS manager_salary
FROM employees e
JOIN employees m ON e.manager_id = m.id
WHERE e.salary > m.salary;
```

See: [`joins.md`](./joins.md)

## 7. Median without a MEDIAN function

**Problem**: Compute the median `salary` from `employees` in an engine without a built-in `MEDIAN()` (e.g. standard MySQL).

**Pattern**: `PERCENTILE_CONT` where available (Postgres/SQL Server); otherwise a manual row-counting approach with `ROW_NUMBER`/`COUNT`.

```sql
-- Postgres / SQL Server (2019+ has PERCENTILE_CONT too)
SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY salary) AS median_salary
FROM employees;

-- Portable manual approach (works anywhere, including MySQL without window-function percentile support)
WITH ordered AS (
  SELECT salary,
    ROW_NUMBER() OVER (ORDER BY salary) AS rn,
    COUNT(*) OVER () AS total
  FROM employees
)
SELECT AVG(salary) AS median_salary
FROM ordered
WHERE rn IN ( (total + 1) / 2, (total + 2) / 2 );   -- averages the middle 1 or 2 rows
```

See: [`window-functions.md`](./window-functions.md)

## 8. Pivot rows to columns

**Problem**: Given `sales(rep, quarter, amount)` (long format), produce one row per `rep` with a column per `quarter` (wide format).

**Pattern**: Conditional aggregation (`CASE WHEN` + `SUM`/`MAX`) — portable across all engines; `CROSSTAB`/`PIVOT` where the dialect supports it natively.

```sql
-- Portable: conditional aggregation
SELECT rep,
  SUM(CASE WHEN quarter = 'Q1' THEN amount ELSE 0 END) AS q1,
  SUM(CASE WHEN quarter = 'Q2' THEN amount ELSE 0 END) AS q2,
  SUM(CASE WHEN quarter = 'Q3' THEN amount ELSE 0 END) AS q3,
  SUM(CASE WHEN quarter = 'Q4' THEN amount ELSE 0 END) AS q4
FROM sales
GROUP BY rep;

-- SQL Server native PIVOT
SELECT rep, [Q1], [Q2], [Q3], [Q4]
FROM sales
PIVOT (SUM(amount) FOR quarter IN ([Q1], [Q2], [Q3], [Q4])) AS p;
```

See: [`select-where-groupby.md`](./select-where-groupby.md)

## Quick reference: which technique for which problem shape

| Problem shape | Technique |
|---|---|
| "Nth highest/lowest X" | `DENSE_RANK()` or `LIMIT/OFFSET` with `DISTINCT` |
| "Top N per group" | `ROW_NUMBER() PARTITION BY group` + filter in outer query |
| "Running total / moving average" | `SUM()`/`AVG() OVER (... ROWS BETWEEN ...)` |
| "Consecutive streaks / gaps & islands" | `date - ROW_NUMBER()` trick |
| "Compare row to previous/next row" | `LAG`/`LEAD` |
| "Rows in A without matching B" | `LEFT JOIN ... WHERE b.id IS NULL` or `NOT EXISTS` |
| "Self-referencing hierarchy" | Self-join (1 level) or recursive CTE (N levels) |
| "Long-to-wide reshape" | Conditional aggregation (`CASE WHEN` + `SUM`) |
| "Duplicate detection/removal" | `GROUP BY HAVING COUNT(*) > 1`, or `ROW_NUMBER()` + delete `rn > 1` |

## See also

- [`select-where-groupby.md`](./select-where-groupby.md)
- [`joins.md`](./joins.md)
- [`window-functions.md`](./window-functions.md)
- [`ctes-subqueries.md`](./ctes-subqueries.md)
- [`query-execution-order.md`](./query-execution-order.md)
- [`indexes-optimization.md`](./indexes-optimization.md)
