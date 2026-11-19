import os
import re
import sys
import json
import time
import hashlib
import platform
import threading
import subprocess
import statistics
import numpy as np
from numba import njit

try:
    import resource
except ImportError:
    resource = None

# Replace with your exact Git handle before executing
GIT_USERNAME = "AliOrynbay"

NUM_THREADS = 4
ITERATIONS_PER_THREAD = 60_000_000
REPEATS = 3
STRIDES = [1, 4, 8, 16]
SMOKE = "--smoke" in sys.argv
if SMOKE:
    ITERATIONS_PER_THREAD = 2_000_000
    REPEATS = 1


def detect_cache_line_size() -> int:
    """Query L1 data cache coherency line size in bytes from OS sysfs/sysctl."""
    linux_path = "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
    if os.path.exists(linux_path):
        with open(linux_path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    if sys.platform == "darwin":
        out = subprocess.check_output(["sysctl", "-n", "hw.cachelinesize"], encoding="utf-8")
        return int(out.strip())
    return 64


def read_cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    if sys.platform == "darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], encoding="utf-8").strip()
        except Exception:
            pass
    return platform.processor() or platform.machine()


def pick_cores(n: int):
    """Choose n logical CPUs, preferring distinct physical cores (no SMT siblings),
    because SMT siblings share an L1 and would hide false sharing."""
    if not hasattr(os, "sched_getaffinity"):
        return []
    avail = sorted(os.sched_getaffinity(0))
    chosen, seen = [], set()
    for c in avail:
        p = f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list"
        try:
            with open(p, "r", encoding="utf-8") as f:
                key = f.read().strip()
        except OSError:
            key = str(c)
        if key not in seen:
            seen.add(key)
            chosen.append(c)
        if len(chosen) == n:
            break
    for c in avail:
        if len(chosen) == n:
            break
        if c not in chosen:
            chosen.append(c)
    return chosen if len(chosen) == n else []


# ==============================================================================
# TASK 1: Hardware Topology & Student Identity Context
# ==============================================================================
def init_system_context(git_user: str) -> dict:
    assert git_user != "REPLACE_WITH_YOUR_GIT_USERNAME" and len(git_user.strip()) > 0, (
        "Set GIT_USERNAME to your real Git handle before running."
    )
    logical_cores = os.cpu_count() or 1
    if logical_cores < NUM_THREADS:
        raise RuntimeError(f"Requires at least {NUM_THREADS} logical cores; detected {logical_cores}.")

    line_size = detect_cache_line_size()
    cpu_model = read_cpu_model()
    cpu_sig = f"{platform.machine()}::{platform.processor()}::{logical_cores}::{cpu_model}"

    digest = hashlib.sha256(f"{git_user.strip()}::{cpu_sig}".encode("utf-8")).hexdigest()
    instance_seed = int(digest[:16], 16)

    cores = pick_cores(NUM_THREADS)
    affinity_supported = bool(cores) and hasattr(os, "sched_setaffinity")

    return {
        "git_username": git_user.strip(),
        "logical_cores": logical_cores,
        "cache_line_size": line_size,
        "cpu_model": cpu_model,
        "cpu_signature": cpu_sig,
        "instance_seed": instance_seed,
        "pinned_cores": cores if affinity_supported else [],
        "affinity_supported": affinity_supported,
        "os": sys.platform,
        "python": platform.python_version(),
    }


# ==============================================================================
# TASK 2: GIL-free native counter kernel
# ==============================================================================
def _strided_accumulate_py(buffer, slot, iters, seed):
    one = np.uint64(1)
    three = np.uint64(3)
    sh = np.uint64(63)
    for k in range(iters):
        old = buffer[slot]                              # real L1 load every iteration
        j = slot + np.int64(old >> sh)                  # address depends on the loaded value (always +0)
        buffer[j] = old + ((seed ^ np.uint64(k)) & three) + one   # real L1 store every iteration


worker_strided_accumulate = njit(nogil=True)(_strided_accumulate_py)
# Same body, but the GIL stays held during the call (used only for the Q5 experiment)
worker_strided_gil_held = njit(_strided_accumulate_py)


# ==============================================================================
# TASK 4: thread-local register privatization baseline
# ==============================================================================
@njit(nogil=True)
def worker_private_accumulate(buffer, slot, iters, seed):
    one = np.uint64(1)
    three = np.uint64(3)
    acc = np.uint64(0)
    for k in range(iters):
        acc += ((seed ^ np.uint64(k)) & three) + one    # accumulator lives in a register
    buffer[slot] = acc                                   # exactly one shared-memory write


def expected_final(iters: int, seed: int) -> int:
    """Closed form of sum_{k<iters} (((seed ^ k) & 3) + 1), independent of the kernel."""
    s2 = seed & 3
    vals = [(s2 ^ r) + 1 for r in range(4)]
    return (iters // 4) * sum(vals) + sum(vals[: iters % 4])


def aligned_u64(n_elems: int) -> np.ndarray:
    raw = np.zeros(n_elems * 8 + 64, dtype=np.uint8)
    off = (-raw.ctypes.data) % 64
    buf = raw[off: off + n_elems * 8].view(np.uint64)
    assert buf.ctypes.data % 64 == 0
    return buf


def thread_seeds(instance_seed: int):
    mask = (1 << 64) - 1
    return [np.uint64((instance_seed ^ ((t + 1) * 0x9E3779B97F4A7C15)) & mask)
            for t in range(NUM_THREADS)]


def run_threads(kernel, buf, slots, iters, seeds, cores):
    n = len(slots)
    barrier = threading.Barrier(n)
    starts = [0] * n
    ends = [0] * n

    def target(t):
        if cores and hasattr(os, "sched_setaffinity"):
            try:
                os.sched_setaffinity(0, {cores[t]})   # applies to the calling thread on Linux
            except OSError:
                pass
        barrier.wait()
        starts[t] = time.perf_counter_ns()
        kernel(buf, slots[t], iters, seeds[t])
        ends[t] = time.perf_counter_ns()

    threads = [threading.Thread(target=target, args=(t,)) for t in range(n)]
    ru0 = resource.getrusage(resource.RUSAGE_SELF) if resource else None
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    ru1 = resource.getrusage(resource.RUSAGE_SELF) if resource else None
    elapsed = max(ends) - min(starts)
    vol = (ru1.ru_nvcsw - ru0.ru_nvcsw) if resource else None
    invol = (ru1.ru_nivcsw - ru0.ru_nivcsw) if resource else None
    return elapsed, vol, invol


# ==============================================================================
# TASK 3: multi-threaded stride sweep (+ Task 4 baseline)
# ==============================================================================
def run_stride_benchmark(ctx: dict) -> dict:
    iters = ITERATIONS_PER_THREAD
    seeds = thread_seeds(ctx["instance_seed"])
    expected = [expected_final(iters, int(s)) for s in seeds]
    cores = ctx["pinned_cores"]

    # JIT warm-up so compilation time is never measured
    warm = aligned_u64(64)
    worker_strided_accumulate(warm, 0, 16, seeds[0])
    worker_private_accumulate(warm, 0, 16, seeds[0])

    results = {}
    finals_by_layout = {}

    def bench(name, kernel, stride):
        buf_elems = max(NUM_THREADS * stride, 16) + 32
        runs = []
        for _ in range(REPEATS):
            buf = aligned_u64(buf_elems)
            slots = [t * stride for t in range(NUM_THREADS)]
            el, vol, invol = run_threads(kernel, buf, slots, iters, seeds, cores)
            finals = [int(buf[s]) for s in slots]
            runs.append({"elapsed_ns": el, "vol_cs": vol, "invol_cs": invol, "finals": finals})
        runs.sort(key=lambda r: r["elapsed_ns"])
        med = runs[len(runs) // 2]
        entry = {
            "stride_elements": stride,
            "stride_bytes": stride * 8,
            "all_elapsed_ns": [r["elapsed_ns"] for r in runs],
            "median_elapsed_ns": med["elapsed_ns"],
            "min_elapsed_ns": runs[0]["elapsed_ns"],
            "ns_per_update_per_thread": med["elapsed_ns"] / iters,
            "ns_per_update_aggregate": med["elapsed_ns"] / (iters * NUM_THREADS),
            "voluntary_ctx_switches": med["vol_cs"],
            "involuntary_ctx_switches": med["invol_cs"],
            "final_counters": med["finals"],
            "parity_with_expected": all(r["finals"] == expected for r in runs),
        }
        finals_by_layout[name] = [r["finals"] for r in runs]
        return entry

    for stride in STRIDES:
        results[f"stride_{stride}"] = bench(f"stride_{stride}", worker_strided_accumulate, stride)
    results["private"] = bench("private", worker_private_accumulate, 1)
    results["_expected_final"] = expected
    results["_finals_by_layout"] = finals_by_layout
    return results


# ==============================================================================
# Q5 experiment: GIL held vs released
# ==============================================================================
def py_bytecode_loop(buf, slot, iters):
    for _ in range(iters):
        buf[slot] += 1


def run_gil_experiment(ctx: dict) -> dict:
    out = {}
    seeds = thread_seeds(ctx["instance_seed"])
    n_iters = ITERATIONS_PER_THREAD // 3
    warm = aligned_u64(64)
    worker_strided_gil_held(warm, 0, 16, seeds[0])

    def timed(kernel, n_threads, buf_factory):
        buf = buf_factory()
        barrier = threading.Barrier(n_threads)
        starts, ends = [0] * n_threads, [0] * n_threads

        def target(t):
            barrier.wait()
            starts[t] = time.perf_counter_ns()
            kernel(buf, t, n_iters, seeds[t])
            ends[t] = time.perf_counter_ns()

        ths = [threading.Thread(target=target, args=(t,)) for t in range(n_threads)]
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        return max(ends) - min(starts)

    mk = lambda: aligned_u64(64)
    out["numba_nogil_1_thread_ns"] = timed(worker_strided_accumulate, 1, mk)
    out["numba_nogil_4_threads_ns"] = timed(worker_strided_accumulate, NUM_THREADS, mk)
    out["numba_gil_held_1_thread_ns"] = timed(worker_strided_gil_held, 1, mk)
    out["numba_gil_held_4_threads_ns"] = timed(worker_strided_gil_held, NUM_THREADS, mk)

    py_iters = 1_000_000 if not SMOKE else 200_000
    lst = [0] * 64

    def py_timed(n_threads):
        barrier = threading.Barrier(n_threads)
        starts, ends = [0] * n_threads, [0] * n_threads

        def target(t):
            barrier.wait()
            starts[t] = time.perf_counter_ns()
            py_bytecode_loop(lst, t, py_iters)
            ends[t] = time.perf_counter_ns()

        ths = [threading.Thread(target=target, args=(t,)) for t in range(n_threads)]
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        return max(ends) - min(starts)

    out["python_bytecode_1_thread_ns"] = py_timed(1)
    out["python_bytecode_4_threads_ns"] = py_timed(NUM_THREADS)
    out["numba_iters_per_thread"] = n_iters
    out["python_iters_per_thread"] = py_iters
    return out


# ==============================================================================
# TASK 5: exact parity and LLVM IR verification
# ==============================================================================
def audit_llvm_and_parity(results: dict, ctx: dict, gil_exp: dict) -> dict:
    expected = results["_expected_final"]
    layouts = [f"stride_{s}" for s in STRIDES] + ["private"]

    parity_expected = all(results[l]["parity_with_expected"] for l in layouts)
    ref = results["stride_1"]["final_counters"]
    parity_across = all(results[l]["final_counters"] == ref for l in layouts)
    parity_closed_form = (ref == expected)

    ir_all = worker_strided_accumulate.inspect_llvm()
    ir = "\n".join(ir_all.values())
    store_i64 = len(re.findall(r"\bstore i64\b", ir))
    load_i64 = len(re.findall(r"\bload i64\b", ir))
    atomic_or_volatile = len(re.findall(r"\bstore (?:atomic|volatile)\b", ir))
    pyobj_markers = len(re.findall(r"PyObject_|PyNumber_|NRT_MemInfo_alloc", ir))
    nopython = len(worker_strided_accumulate.nopython_signatures) > 0
    # Py* symbols may legitimately appear in the cpython call wrapper (arg unpacking, errors);
    # object-mode fallback is excluded by nopython_signatures being non-empty.
    gil_release_calls = len(re.findall(r"PyEval_SaveThread", ir))
    llvm_ok = nopython and store_i64 > 0 and load_i64 > 0

    private_ir = "\n".join(worker_private_accumulate.inspect_llvm().values())
    audit = {
        "nopython_compiled": nopython,
        "store_i64_count": store_i64,
        "load_i64_count": load_i64,
        "atomic_or_volatile_store_count": atomic_or_volatile,
        "python_object_marker_count": pyobj_markers,
        "gil_release_calls_in_wrapper": gil_release_calls,
        "private_kernel_store_i64_count": len(re.findall(r"\bstore i64\b", private_ir)),
        "llvm_ok": llvm_ok,
        "parity_all_layouts_equal": parity_across,
        "parity_matches_closed_form": parity_closed_form,
        "parity_all_repeats_match_expected": parity_expected,
    }

    r1 = results["stride_1"]["median_elapsed_ns"]
    r8 = results["stride_8"]["median_elapsed_ns"]
    slowdown = r1 / r8
    manifest = {
        "git_username": ctx["git_username"],
        "instance_seed": ctx["instance_seed"],
        "cpu_model": ctx["cpu_model"],
        "logical_cores": ctx["logical_cores"],
        "cache_line_size": ctx["cache_line_size"],
        "pinned_cores": ctx["pinned_cores"],
        "affinity_supported": ctx["affinity_supported"],
        "num_threads": NUM_THREADS,
        "iterations_per_thread": ITERATIONS_PER_THREAD,
        "repeats": REPEATS,
        "expected_final_counter_per_thread": expected,
        "results": {k: v for k, v in results.items() if not k.startswith("_")},
        "slowdown_stride1_vs_stride8": slowdown,
        "false_sharing_observed": slowdown > 1.5,
        "gil_experiment": gil_exp,
        "llvm_audit": audit,
        "smoke_run": SMOKE,
        "verification_passed": bool(parity_across and parity_closed_form
                                    and parity_expected and llvm_ok),
    }
    path = "cache_lab_manifest.smoke.json" if SMOKE else "cache_lab_manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main():
    ctx = init_system_context(GIT_USERNAME)
    print(f"GIT_USERNAME={ctx['git_username']}")
    print(f"CPU={ctx['cpu_model']}  logical_cores={ctx['logical_cores']}")
    print(f"cache_line_size={ctx['cache_line_size']} B  INSTANCE_SEED={ctx['instance_seed']}")
    print(f"affinity_supported={ctx['affinity_supported']} pinned_cores={ctx['pinned_cores']}")
    print(f"threads={NUM_THREADS} iters/thread={ITERATIONS_PER_THREAD:,} repeats={REPEATS}\n")

    results = run_stride_benchmark(ctx)
    print(f"{'layout':<12}{'bytes':>6}{'median ns':>18}{'ns/update/thread':>20}{'inv.cs':>8}")
    for key in [f"stride_{s}" for s in STRIDES] + ["private"]:
        r = results[key]
        print(f"{key:<12}{r['stride_bytes']:>6}{r['median_elapsed_ns']:>18,}"
              f"{r['ns_per_update_per_thread']:>20.4f}{str(r['involuntary_ctx_switches']):>8}")

    gil_exp = run_gil_experiment(ctx)
    print("\nGIL experiment:")
    for k, v in gil_exp.items():
        print(f"  {k} = {v:,}")

    m = audit_llvm_and_parity(results, ctx, gil_exp)
    print(f"\nslowdown stride1/stride8 = {m['slowdown_stride1_vs_stride8']:.3f}x")
    print(f"LLVM audit: {m['llvm_audit']}")
    print(f"verification_passed={m['verification_passed']}")
    print("Wrote", "cache_lab_manifest.smoke.json" if SMOKE else "cache_lab_manifest.json")


if __name__ == "__main__":
    main()
