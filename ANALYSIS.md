# ANALYSIS.md — Cache Line & False Sharing Mechanics

Host: AMD64 Family 25 Model 80 (AuthenticAMD, Zen 3), 12 logical cores, cache line 64 B, Windows, Python 3.11.
Core affinity pinning was **not available** (`affinity_supported = False`), so the OS scheduler placed the four threads; context-switch counters are unavailable on Windows (`resource` module missing → `None`).
INSTANCE_SEED = 12256014976366901763. All runs: 4 threads, 60,000,000 updates per thread, median of 3 repeats.
Final counter value in every slot of every layout: 150,000,000 (bit-exact, matches the closed form).

## Question 1 — MESI state transitions (stride = 1)

With stride 1, `buffer[0]` and `buffer[1]` are 8 B apart, so they live in the same 64-byte line.

1. Core 0 stores to `buffer[0]`. Its L1 misses (state I), it issues a Read-For-Ownership, and the line arrives in **M** (Modified) after the store. Core 1 holds the line **I**.
2. Core 1 stores to `buffer[1]`. Its L1 misses (state I), so it broadcasts an RFO. Core 0 must supply the dirty line (write-back or cache-to-cache transfer) and goes **M → I**. Core 1 receives the line and goes **I → M**.
3. Core 0's next store to `buffer[0]` now misses (I), sends an RFO, forces Core 1 **M → I**, and the line moves back (**I → M** in Core 0).
4. This repeats on every iteration: the line ping-pongs M → I → M between the cores (with E/S appearing only transiently, since every access is a write and requires exclusive ownership). Different bytes are being written, but coherence works on the whole line.

The store cannot commit to L1 until ownership arrives. Stores wait in the store buffer and commit in program order (x86-TSO), so the store at the head of the buffer, waiting for its RFO, blocks all younger stores behind it. Each core's next store targets the same contended line, the store buffer fills, and the core stalls for a cross-core transfer (tens of ns) instead of an L1 hit (about 1 ns).

## Question 2 — Quantitative slowdown

Per-update numbers are `median_elapsed_ns / 60,000,000` (per thread, threads run concurrently). The aggregate column divides by 4 threads × 60M updates.

| Layout | stride | bytes | median elapsed (ns) | ns / update / thread | ns / update (aggregate) |
|---|---|---|---|---|---|
| A contiguous | 1 | 8 | 569,638,400 | 9.4940 | 2.3735 |
| B half-line | 4 | 32 | 197,973,900 | 3.2996 | 0.8249 |
| C line-padded | 8 | 64 | 43,032,800 | 0.7172 | 0.1793 |
| D prefetch-safe | 16 | 128 | 35,106,800 | 0.5851 | 0.1463 |
| Private register | — | — | 3,784,700 | 0.0631 | 0.0158 |

- Slowdown stride 1 vs stride 8 = 569,638,400 / 43,032,800 = **13.24×** (stride 1 vs stride 16 = 16.2×).
- Stride 4 is 4.60× slower than stride 8, and 2.88× faster than stride 1. Stride 4 is a 2-way sharing case (threads 0,1 on line 0, threads 2,3 on line 1), so it sits between the 4-way ping-pong and the no-sharing case.
- Stride 8 vs private register: 43,032,800 / 3,784,700 = 11.4×. So A → C (13.2×) is pure cross-core coherence cost, and C → private (11.4×) is the cost of a real L1 load + store chain per iteration (store-to-load forwarding latency, about 0.7 ns ≈ 3 cycles at ~4 GHz) compared with a register add.
- Stride 1 vs private: 150×.

## Question 3 — L2 spatial prefetcher and 128-byte pairing

Stride 8 (64 B spacing) took 43,032,800 ns (0.7172 ns/update) and stride 16 (128 B spacing) took 35,106,800 ns (0.5851 ns/update) in this run, a 22.6% difference. The three repeats of stride 8 were 35,668,900 / 43,032,800 / 46,669,700 ns and of stride 16 were 34,890,600 / 35,106,800 / 41,904,700 ns; the ranges overlap (best stride 8 is 35.67 ms vs best stride 16 34.89 ms, a 2.2% difference), so the median gap is run-to-run noise (unpinned threads, turbo/boost state, background load), not a consistent prefetcher effect.

Mechanism: some Intel cores have an "adjacent cache line" L2 prefetcher that treats each aligned 128-byte pair (even line + odd line) as a sector and fetches the buddy line along with the one that missed. With 64-byte padding, thread 0's slot (line 0) and thread 1's slot (line 1) share one 128-byte sector, so a prefetch of the buddy line can pull in a line another core is writing, forcing an extra ownership transfer or invalidation (a secondary invalidation). 128-byte spacing puts each slot in its own sector and removes this. 64-byte padding can still suffer when the prefetcher is enabled and threads write adjacent lines of the same pair at the same time, or when the buffer base is not 128-byte aligned.

My CPU is an AMD Zen 3, and I did not observe a consistent stride-8 vs stride-16 difference, which fits that the Intel-style paired-line prefetch is not the dominant effect on this core. Stride 16 is the safest choice when the target microarchitecture is unknown.

## Question 4 — Register promotion vs real stores

An optimizing compiler's loop-invariant scalar promotion sees that `buffer[slot]` has the same address on every iteration. If nothing else in the loop can alias it, it can load it once before the loop, keep the running value in a register, and store it once after the loop (it may even turn the loop into closed form). That is legal because LLVM assumes data-race-free code and does not model other threads. It would completely hide false sharing, and it is what the private baseline deliberately does.

My kernel prevents that: each iteration loads `old = buffer[slot]`, computes the store address as `slot + (old >> 63)` and stores `old + inc` to `buffer[j]`. The store address depends on the value just loaded from memory, and the compiler cannot prove `old >> 63 == 0`, so it cannot prove the store always goes to the same address. Promotion is blocked, and each iteration issues a real load and a real store. The accumulated value is carried through memory from one iteration to the next (a true data dependency). The shift is always 0 at run time, so the result is identical to `buffer[slot] += inc`.

From `inspect_llvm()` in my run: nopython signatures compiled = True (so no object-mode fallback), `store i64` count = 5, `load i64` count = 7, atomic/volatile stores = 0 (they are plain stores, not volatile), `PyEval_SaveThread` count = 2 (the wrapper releases the GIL because of `nogil=True`), and 4 `Py*`/NRT symbols that are part of the CPython call wrapper (argument unpacking and error paths), not the loop. The private kernel has `store i64` count = 1: the single store of the accumulator after the loop, while its loop body only adds to an SSA register.

## Question 5 — CPython GIL masking vs nogil=True

CPython bytecode threads hold the Global Interpreter Lock while executing bytecode, so only one thread runs Python code at a time. Two threads never store to the same cache line at the same instant from different cores, so there is no RFO ping-pong on the data. The only contention is lock hand-off between threads (the GIL is requested every ~5 ms switch interval), so adding threads adds waiting and context switching but no parallel speedup. Stride 1 and stride 8 would take about the same time under the GIL, which is why false sharing is invisible there.

My measurements:
- Python bytecode loop (1,000,000 iterations per thread): 1 thread = 58,010,500 ns, 4 threads = 237,431,400 ns, so wall time rose **4.09×** for the same per-thread work and the aggregate throughput (4M updates in 237 ms) is no better than one thread (1M in 58 ms): fully serialized.
- Numba **without** nogil (GIL held for the whole native call, 20M iterations per thread, adjacent slots): 1 thread = 13,904,200 ns, 4 threads = 50,655,000 ns, i.e. **3.64×**, again almost perfect serialization.
- Numba **with** `nogil=True` (same code, adjacent slots on one cache line): 1 thread = 12,712,400 ns, 4 threads = 132,175,000 ns, i.e. **10.4×** slower even though the threads run truly in parallel on separate cores. That extra slowdown beyond the serialized 4× comes from hardware cache-line contention (RFO ping-pong), not from a software lock.

So GIL contention costs about 4× (waiting for a lock, one thread at a time), whereas false sharing under nogil costs about 10–13× (all four threads running but stalled on cache-line ownership transfers), and only the nogil run exposes the coherence mechanism.
