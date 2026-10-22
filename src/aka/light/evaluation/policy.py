"""Light's measurement defaults, with Legacy-aligned timing and warmup counts.

Keep these local to Light: selecting another AKA composition does not change
its evaluator or acceptance policy. Explicit request/operator overrides still
participate in the measurement identity.
"""

# Preserve the legacy candidate gate: one base seed plus five additional seeds.
CORRECTNESS_CASES = 6
BENCH_ITERS = 20
WARMUP_ITERS = 5
# Some production shapes spend more than two minutes in the first candidate
# touch (input materialization plus JIT compilation).  Keep this aligned with
# the outer sandbox budget so a valid slow-to-initialize kernel is not rejected
# before Agate can finish the evaluation.
CANDIDATE_TIMEOUT = 600
PERFORMANCE_TIMEOUT = 120
SHAPE_BATCH_SIZE = 1
BATCH_WORKERS = 16
MEASUREMENT_CONTRACT = "runtime-native-eval-v1"
ACCEPTANCE_METRIC = "geomean-latency-reduction-v1"


def validate_abba_repeats(repeats: int) -> None:
    if type(repeats) is not int or not 2 <= repeats <= 16 or repeats % 2:
        raise ValueError("--comparison-repeats must be even, in 2..16 (measurements per side)")


def runner_overrides(environment=None) -> dict:
    import os
    environment = os.environ if environment is None else environment
    values = {"warmup_iters": WARMUP_ITERS}
    for field, default in (("candidate_timeout_s", CANDIDATE_TIMEOUT), ("perf_timeout_s", PERFORMANCE_TIMEOUT)):
        value = int(environment.get("ATREX_SANDBOX_" + field.upper(), default))
        if not 1 <= value <= 86400:
            raise ValueError(f"{field} must be between 1 and 86400 seconds")
        values[field] = value
    return values


def measurement_defaults(environment=None) -> dict:
    return {"contract": MEASUREMENT_CONTRACT, "bench_iters": BENCH_ITERS,
            "correctness_cases": CORRECTNESS_CASES, "lock_clocks": True,
            "runner_overrides": runner_overrides(environment)}
