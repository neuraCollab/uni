# Concurrency & Async: threading vs multiprocessing vs asyncio

## What / why

The interview question behind this whole topic: *"why doesn't threading
speed up CPU-bound Python code?"* The answer is the **GIL**, and picking the
right concurrency model (threading / multiprocessing / asyncio) for a given
workload is one of the most common practical Python interview exercises.

## The GIL, in one paragraph

CPython's Global Interpreter Lock allows only **one thread to execute Python
bytecode at a time**, even on a multi-core machine. Threads still provide
real concurrency for **I/O-bound** work, because a thread releases the GIL
while waiting on I/O (network, disk, `time.sleep`) — other threads run
during that wait. But for **CPU-bound** work (pure computation), threads
take turns on the same core; adding more threads doesn't add more compute
throughput, it can even make things slower (context-switch overhead).

- **CPU-bound** (number crunching, image processing, parsing large data) →
  GIL blocks parallel speedup from threads → use **multiprocessing**
  (separate processes, separate interpreters, separate GILs, real parallelism).
- **I/O-bound** (network requests, file/DB I/O, waiting on other services) →
  GIL isn't the bottleneck, waiting is → use **threading** or **asyncio**.

## Decision table

| | Best for | Parallelism model | Overhead | Shares memory? |
|---|---|---|---|---|
| **threading** | I/O-bound, blocking libraries you can't rewrite as async | Concurrent, not parallel (GIL) | Low (OS threads) | Yes — needs locks for shared mutable state |
| **multiprocessing** | CPU-bound | True parallel (separate processes) | High (process spawn, IPC serialization/pickling) | No — must explicitly pass/pickle data |
| **asyncio** | I/O-bound, especially many concurrent connections | Concurrent, single-threaded cooperative multitasking | Very low (no OS thread/process per task) | Yes — but single-threaded, so no data races (still need care around `await` points) |

Rule of thumb: **asyncio for I/O when you control the whole stack** (async
libraries throughout); **threading for I/O when using blocking
libraries/legacy code**; **multiprocessing for CPU-bound**.

## asyncio — anchor example

[code/async_chat_server.py](code/async_chat_server.py) is a self-contained
async TCP chat server+client (adapted from a university networking lab,
Tkinter GUI stripped out). It demonstrates the core asyncio vocabulary:

- **`asyncio.start_server` / `asyncio.open_connection`** — event-driven
  socket I/O; the event loop resumes a coroutine when its socket is ready,
  instead of blocking a thread on `recv()`.
- **`asyncio.Queue`** as a producer/consumer channel: every connection
  handler *publishes* messages into the queue; a single `broadcaster()` task
  *consumes* the queue and does all the actual writes — this is what avoids
  interleaved writes to sockets shared across many "publishers":

  ```python
  async def broadcaster(self) -> None:
      while True:
          message, room = await self.queue.get()
          targets = [c.writer for c in self.clients.values() if c.room == room]
          await asyncio.gather(*(self._send(w, message) for w in targets))
  ```

- **`asyncio.create_task`** — schedules a coroutine to run concurrently
  ("fire and forget", tracked so it can be cancelled later), used both for
  the long-lived `broadcaster` task and per-connection handlers.
- **`asyncio.gather`** — runs multiple awaitables concurrently and waits for
  all of them, used to fan a message out to every client in a room at once
  instead of writing to sockets one at a time.

Key mental model: an `async def` function returns a **coroutine object**
when called — nothing runs until it's `await`ed or scheduled with
`create_task`. `await` yields control back to the event loop, which can run
other ready coroutines while this one waits on I/O.

## multiprocessing — anchor example

[code/multiprocessing_pipeline.py](code/multiprocessing_pipeline.py) is
adapted from a university image-processing lab that split large images into
blocks and processed each block in a separate process
(OpenCV/Tkinter/Excel-export scaffolding stripped; the payload replaced with
a dependency-free "scan a block of numbers" task). The pattern —
**chunk → `Pool.map` → merge** — is the canonical shape for CPU-bound
parallelism in Python:

```python
def process_chunk(args):          # must be top-level: pickled and sent to workers
    chunk_id, chunk, threshold = args
    count = sum(1 for v in chunk if v > threshold)
    return ChunkResult(chunk_id, count, sum(chunk))

def run_pipeline(data, threshold, chunk_size=1000, processes=None):
    chunks = split_into_chunks(data, chunk_size)
    tasks = [(i, c, threshold) for i, c in enumerate(chunks)]
    with mp.Pool(processes=processes) as pool:
        results = pool.map(process_chunk, tasks)          # runs in parallel
    return merge(results)                                  # back in parent process
```

Each worker process has its own Python interpreter (own GIL), so this is
**real** parallel execution across CPU cores — unlike threading. The cost:
data passed to/from workers must be **pickled**, which adds serialization
overhead and means workers don't share memory with the parent (no implicit
shared mutable state — a feature, not a bug, since it sidesteps race
conditions entirely).

## threading — canonical example (written from scratch)

No clean threading-only example existed in the source repo, so this is a
standard, hand-written worker-pool pattern using `queue.Queue` (thread-safe
by design) — the typical shape for an I/O-bound task queue (e.g. downloading
many URLs with a blocking HTTP client):

```python
import queue
import threading
import time

def worker(task_queue: "queue.Queue[int]", results: list, lock: threading.Lock) -> None:
    while True:
        item = task_queue.get()
        if item is None:               # sentinel: no more work
            task_queue.task_done()
            break
        time.sleep(0.1)                # stand-in for a blocking I/O call
        with lock:                     # protect shared mutable state
            results.append(item * 2)
        task_queue.task_done()

def run(n_items: int, n_workers: int = 4) -> list:
    task_queue: "queue.Queue[int]" = queue.Queue()
    results: list = []
    lock = threading.Lock()

    threads = [
        threading.Thread(target=worker, args=(task_queue, results, lock))
        for _ in range(n_workers)
    ]
    for t in threads:
        t.start()

    for item in range(n_items):
        task_queue.put(item)
    for _ in threads:
        task_queue.put(None)           # one sentinel per worker to stop it

    for t in threads:
        t.join()
    return results
```

Notice the `lock` around `results.append` — unlike asyncio (single-threaded)
or multiprocessing (separate memory), threads share memory, so mutable
shared state needs explicit synchronization (`threading.Lock`,
`threading.RLock`) to avoid race conditions.

## Interview questions

- **Why doesn't threading speed up CPU-bound code?** The GIL lets only one
  thread run Python bytecode at a time; CPU-bound threads just take turns on
  one core instead of running in parallel.
- **When does threading actually help, then?** I/O-bound work — a thread
  blocked on `socket.recv()` or file I/O releases the GIL, letting other
  threads run during the wait.
- **multiprocessing vs asyncio for a web scraper hitting 1000 URLs?**
  asyncio (if using an async HTTP client) — I/O-bound, and asyncio avoids
  the overhead of spawning/serializing across processes; multiprocessing
  would work but wastes resources on I/O-bound work.
- **Why must the function passed to `Pool.map` be top-level, not a lambda
  or closure?** It has to be picklable to be sent to worker processes;
  lambdas and closures aren't picklable.
- **Does asyncio need locks like threading does?** Generally no for pure
  Python code — only one coroutine runs at a time, switches happen only at
  `await` points, so as long as you don't `await` in the middle of a
  "critical section", there's no race. (Still relevant if mixing with
  threads, or reasoning carefully about state that spans multiple `await`s.)
- **What's the cost of multiprocessing that threading doesn't have?**
  Process startup time and IPC (pickling data across the process boundary)
  — much higher overhead per task than a thread or an asyncio task.

## Common mistakes

- Reaching for `threading` to speed up a CPU-bound loop and being confused
  when it doesn't help (or gets slower).
- Using `multiprocessing` for I/O-bound work — pays serialization/process
  overhead for no parallelism benefit, since the bottleneck was waiting, not
  computing.
- Forgetting that objects passed into `Pool.map`/`Process` must be
  picklable — closures, lambdas, open file handles, and locks generally
  aren't.
- Blocking the asyncio event loop with a synchronous call (e.g.
  `time.sleep()` instead of `await asyncio.sleep()`, or a CPU-heavy
  computation) — freezes *every* coroutine, not just the current one, since
  there's only one thread.
- Sharing mutable state across threads without a lock (race conditions),
  or assuming multiprocessing shares memory the way threading does (it
  doesn't — each process gets its own copy).

## Related

[iterators-generators.md](iterators-generators.md) (generators are the
conceptual ancestor of coroutines/`async def`) ·
[context-managers.md](context-managers.md) (`async with`) ·
[decorators.md](decorators.md) (`asyncio.timeout` and similar utilities are
often used as decorators/context managers)
