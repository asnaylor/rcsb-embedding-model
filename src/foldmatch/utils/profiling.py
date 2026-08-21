import logging
from contextlib import contextmanager, nullcontext
from typing import Iterator

import torch


def _gib(value: int) -> float:
    return value / (1024 ** 3)


@contextmanager
def cuda_profile_stage(
        name: str,
        *,
        nvtx: bool = False,
        memory: bool = False,
        logger: logging.Logger | None = None,
) -> Iterator[None]:
    """Optionally annotate a CUDA stage and report its allocator peak.

    Memory reporting synchronizes CUDA before and after the stage so that the
    reported range has an unambiguous boundary. This is intentionally opt-in
    because synchronization changes execution timing.
    """
    if not (nvtx or memory) or not torch.cuda.is_available():
        yield
        return

    log = logger or logging.getLogger(__name__)
    baseline_allocated = 0
    baseline_reserved = 0

    if memory:
        torch.cuda.synchronize()
        baseline_allocated = torch.cuda.memory_allocated()
        baseline_reserved = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()

    nvtx_range = torch.cuda.nvtx.range(name) if nvtx else nullcontext()
    status = "ok"
    try:
        with nvtx_range:
            yield
    except BaseException:
        status = "error"
        raise
    finally:
        if memory:
            # An OOM can also make synchronization fail. Preserve the original
            # exception and still report every allocator counter that is safe to
            # query from the host.
            try:
                torch.cuda.synchronize()
            except Exception:
                status = "error"

            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            peak_allocated = torch.cuda.max_memory_allocated()
            peak_reserved = torch.cuda.max_memory_reserved()
            log.info(
                "[cuda-memory] stage=%s status=%s "
                "allocated_start=%.3f_GiB allocated_end=%.3f_GiB "
                "peak_allocated=%.3f_GiB peak_allocated_delta=%.3f_GiB "
                "reserved_start=%.3f_GiB reserved_end=%.3f_GiB "
                "peak_reserved=%.3f_GiB peak_reserved_delta=%.3f_GiB",
                name,
                status,
                _gib(baseline_allocated),
                _gib(allocated),
                _gib(peak_allocated),
                _gib(max(0, peak_allocated - baseline_allocated)),
                _gib(baseline_reserved),
                _gib(reserved),
                _gib(peak_reserved),
                _gib(max(0, peak_reserved - baseline_reserved)),
            )
