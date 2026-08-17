# This file was created with the assistance of Generative AI.
"""recsys package init.

Single-thread the BLAS pool as early as importing this package. implicit does its own
parallelism and its CPU ALS explicitly warns that a multi-threaded BLAS pool "can lead to severe
performance issues here" (and our own baseline confirmed 1 thread is as fast or faster for both
fit and recommend on this data).

Two mechanisms, because either can be the effective one depending on import order:
  - OPENBLAS_NUM_THREADS / OMP_NUM_THREADS: read by the BLAS libraries at load time (import numpy).
    Effective only if this package is imported before numpy -- otherwise a no-op (too late).
  - threadpoolctl.threadpool_limits: reconfigures the ALREADY-LOADED pool at runtime. Effective
    when numpy is imported first (the usual case). This is what actually silences implicit's
    RuntimeWarning; the env-var setdefault above is the belt to its suspenders.

This only changes threading/performance, never numerical output. On Apple Accelerate-backed numpy
(the Mac) it likewise caps BLAS threads, which is fine -- implicit prefers 1 either way. Hold the
limiter in a module global so it isn't garbage-collected (which would restore the original limit).
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

try:
    from threadpoolctl import threadpool_limits as _threadpool_limits
    _BLAS_LIMITER = _threadpool_limits(limits=1, user_api="blas")
except Exception:
    _BLAS_LIMITER = None
