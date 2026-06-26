import time
from contextlib import contextmanager
from dataclasses import dataclass, field

@dataclass
class TimerReport:
    items: dict[str, float] = field(default_factory=dict)

    def add(self, name: str, seconds: float):
        self.items[name] = self.items.get(name, 0.0) + seconds

    def summary(self, topk: int = 100) -> str:
        pairs = sorted(self.items.items(), key=lambda x: -x[1])[:topk]
        lines = ["\n[Timing Summary]"]
        for k, v in pairs:
            lines.append(f"{k:45s} {v:10.3f} s")
        return "\n".join(lines)

@contextmanager
def timed(name: str, report: TimerReport | None = None, print_each: bool = True):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        if report is not None:
            report.add(name, dt)
        if print_each:
            print(f"[time] {name}: {dt:.3f}s")