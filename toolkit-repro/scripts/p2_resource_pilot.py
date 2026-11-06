import json
import os
import statistics
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import psutil

root = Path(__file__).resolve().parents[1]
record = root / "records" / (
    "p2-resource-pilot-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    + "-" + uuid.uuid4().hex[:6]
)
record.mkdir(parents=True, exist_ok=False)
report = {
    "status": "STARTED",
    "purpose": "Resource pilot; no formal retrieval evaluation",
    "ram_sampling_interval_seconds": 0.1,
    "ram_measurement": "Child-process sampled RSS; peak is a lower bound",
}
report_path = record / "report.json"


def save():
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


save()
stop = threading.Event()
sampler = None
child = None
samples = {"peak_rss_bytes": 0, "count": 0}
child_record = None

try:
    report["available_ram_before_bytes"] = psutil.virtual_memory().available
    command = [
        sys.executable,
        str(root / "scripts/p1_real_cache.py"),
        "prepare", "--pages", "8",
    ]
    report["command"] = command
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"

    started = time.perf_counter()
    child = subprocess.Popen(
        command,
        cwd=str(root),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        bufsize=1,
    )
    process = psutil.Process(child.pid)

    def sample_ram():
        while not stop.is_set():
            try:
                rss = process.memory_info().rss
                samples["peak_rss_bytes"] = max(samples["peak_rss_bytes"], rss)
                samples["count"] += 1
            except psutil.NoSuchProcess:
                break
            stop.wait(0.1)

    sampler = threading.Thread(target=sample_ram, daemon=True)
    sampler.start()

    with (record / "console.log").open("x", encoding="utf-8") as log:
        for line in child.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
            if line.startswith("Record: "):
                child_record = Path(line[len("Record: "):].strip())

    code = child.wait()
    report["total_wall_seconds"] = time.perf_counter() - started
    report["child_exit_code"] = code
    assert code == 0, "Cache preparation failed; see console.log"
    assert child_record is not None

    child_report = json.loads(
        (child_record / "report.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (child_record / "cache-manifest.json").read_text(encoding="utf-8")
    )
    assert child_report["status"] == "PASS"
    report["cache_preparation_record"] = str(child_record)
    report["new_page_ids"] = child_report["new_page_ids"]
    report["model_loaded"] = child_report["model_loaded"]
    report["peak_allocated_bytes"] = child_report.get("peak_allocated_bytes")
    report["peak_reserved_bytes"] = child_report.get("peak_reserved_bytes")

    page_times = list(child_report["new_page_seconds"].values())
    completed_pages = len(manifest["completed"]["pages"])
    completed_queries = len(manifest["completed"]["queries"])
    report["cache_counts"] = {
        "pages": completed_pages, "expected_pages": 452,
        "queries": completed_queries, "expected_queries": 232,
    }

    if page_times:
        mean = statistics.mean(page_times)
        report["page_encoding"] = {
            "samples_seconds": page_times,
            "mean_seconds": mean,
            "median_seconds": statistics.median(page_times),
            "remaining_pages": 452 - completed_pages,
            "remaining_embed_images_seconds_estimate": (
                mean * (452 - completed_pages)
            ),
            "timing_boundary": (
                "Toolkit embed_images wall time, including its preprocessing "
                "and cleanup; excludes separate mask preparation, pooling "
                "and cache writing"
            ),
        }
    else:
        report["page_encoding"] = {
            "note": "No new pages encoded; no new throughput estimate"
        }

    report["status"] = "PASS"

except Exception as exc:
    report["status"] = "FAILED"
    report["error"] = repr(exc)
    raise

finally:
    stop.set()
    if sampler is not None:
        sampler.join(timeout=2)
    report["ram_samples"] = samples
    save()

print("Pilot record:", record)
print("New pages:", report["new_page_ids"])
print("Cache counts:", report["cache_counts"])
print("Sampled peak process RSS GiB:", samples["peak_rss_bytes"] / 1024**3)
if report["peak_allocated_bytes"] is not None:
    print("PyTorch peak allocated GiB:",
          report["peak_allocated_bytes"] / 1024**3)
print("Page encoding:", report["page_encoding"])
print("P2 resource pilot: PASS")