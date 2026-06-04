"""NDJSON event emitter — the studio's shared pipeline vocabulary."""
from __future__ import annotations
import json, sys, time
from dataclasses import dataclass, field
from typing import Any, Callable, TextIO

STEP_START = "step_start"; STEP_COMPLETE = "step_complete"
TOKEN = "token"; DONE = "done"; ERROR = "error"
GATE_PASS = "gate_pass"; GATE_FAIL = "gate_fail"
RETRY = "retry"; FALLBACK = "fallback"; SKIP = "skip"

@dataclass
class EventEmitter:
    out: TextIO | None = field(default_factory=lambda: sys.stdout)
    sink: Callable[[dict], None] | None = None
    collected: list[dict] = field(default_factory=list)

    def emit(self, event: str, stage: str = "", **data: Any) -> dict:
        rec = {"event": event, "stage": stage, "ts": round(time.time(), 3), **data}
        self.collected.append(rec)
        if self.out:
            self.out.write(json.dumps(rec) + "\n"); self.out.flush()
        if self.sink:
            self.sink(rec)
        return rec

    def step_start(self, s, **d):    return self.emit(STEP_START, s, **d)
    def step_complete(self, s, **d): return self.emit(STEP_COMPLETE, s, **d)
    def gate_pass(self, s, **d):     return self.emit(GATE_PASS, s, **d)
    def gate_fail(self, s, **d):     return self.emit(GATE_FAIL, s, **d)
    def retry(self, s, **d):         return self.emit(RETRY, s, **d)
    def fallback(self, s, **d):      return self.emit(FALLBACK, s, **d)
    def skip(self, s, **d):          return self.emit(SKIP, s, **d)
    def error(self, s, **d):         return self.emit(ERROR, s, **d)
    def done(self, **d):             return self.emit(DONE, "pipeline", **d)
