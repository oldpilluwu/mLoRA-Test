import json
import os
import threading
from typing import Any, Dict

__trace_file = None
__trace_lock = threading.Lock()


def trace_json_init(path: str | None):
    global __trace_file

    if path is None or path == "":
        return

    trace_dir = os.path.dirname(path)
    if trace_dir != "":
        os.makedirs(trace_dir, exist_ok=True)

    __trace_file = open(path, "a", encoding="utf-8")


def trace_json_log(event: str, payload: Dict[str, Any]):
    global __trace_file

    if __trace_file is None:
        return

    record = {"event": event, **payload}
    with __trace_lock:
        __trace_file.write(json.dumps(record, sort_keys=True) + "\n")
        __trace_file.flush()


def trace_json_close():
    global __trace_file

    if __trace_file is None:
        return

    with __trace_lock:
        __trace_file.close()
        __trace_file = None
