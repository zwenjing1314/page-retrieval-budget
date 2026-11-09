import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
import zipfile
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

from prb_cache import digest, json_bytes


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
ENV = "page-retrieval-budget"

SOURCES = {
    "P1": ROOT / "records/p1-closeout-20261107-004222-e54417",
    "encoding": ROOT / "records/p2-full-cache-20261107-005355-ff459c",
    "A": ROOT / "records/p2-formal-a-20261107-012250-612a76",
    "B": ROOT / "records/p2-formal-b-20261107-013537-34e55b",
    "C": ROOT / "records/p2-formal-c-20261107-014210-68a2ce",
    "timing": ROOT / "records/p3-search-timing-20261107-014826-ca6af3",
    "storage": ROOT / "records/p3-storage-20261107-015532-76c94e",
}

record = ROOT / "records" / (
    "p3-freeze-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    + "-" + uuid.uuid4().hex[:6]
)
record.mkdir(parents=True, exist_ok=False)

report = {"status": "STARTED", "environment": ENV}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save():
    (record / "report.json").write_bytes(json_bytes(report))


def write_json(name, value):
    (record / name).write_bytes(json_bytes(value))


def reject_credentials(text):
    # Do not save exports containing credentials embedded in URLs.
    for value in re.findall(r"""https?://[^\s<>"']+""", text):
        parsed = urlsplit(value)
        assert parsed.username is None and parsed.password is None, (
            "Credential-bearing URL found; export not saved"
        )
        assert not re.search(
            r"/(?:t|token)/[^/]+", parsed.path, flags=re.I
        ), "Token-bearing URL found; export not saved"
        sensitive = {
            "token", "access_token", "api_key", "apikey",
            "password", "secret", "signature",
        }
        assert not any(
            key.lower() in sensitive
            for key, _ in parse_qsl(parsed.query)
        ), "Credential query parameter found; export not saved"


def capture(args, filename):
    result = subprocess.run(
        args,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Export failed: {args[0]} {args[1:3]}; "
            f"exit={result.returncode}"
        )
    reject_credentials(result.stdout)
    (record / filename).write_bytes(result.stdout.encode("utf-8"))
    return result.stdout


save()
print("Record:", record, flush=True)

try:
    assert Path(sys.prefix).name.lower() == ENV
    report["python_executable"] = sys.executable
    report["python_version"] = sys.version

    reports = {}
    evidence = {}
    for label, source in SOURCES.items():
        previous = load(source / "report.json")
        assert previous["status"] == "PASS", label
        if label in {"A", "B", "C"}:
            assert previous["completed_queries"] == 232
            assert previous["counts"] == {
                "pages": 452, "queries": 232, "qrels": 3628
            }
        if label in {"A", "B", "C", "timing", "storage"}:
            for name, expected in previous["artifact_sha256"].items():
                assert digest(source / name) == expected, (label, name)
        reports[label] = previous
        evidence[label] = {
            "path": str(source),
            "report_sha256": digest(source / "report.json"),
            "status": previous["status"],
        }
    write_json("evidence-index.json", evidence)

    profile = load(ROOT / "cache/encoded-economics-v1/profile.json")
    for name, expected in profile["versions"].items():
        assert version(name) == expected, f"Version changed: {name}"
    for relative, expected in profile["encoding_code_sha256"].items():
        assert digest(ROOT / relative.replace("\\", "/")) == expected

    report["core_versions"] = {
        name: version(name)
        for name in (
            "torch", "torchvision", "transformers", "peft",
            "colpali-engine", "accelerate", "datasets",
            "qdrant-client", "numpy", "pydantic",
        )
    }

    # Locate Conda itself without inspecting any other environment.
    conda = os.environ.get("CONDA_EXE")
    if not conda:
        conda = str(Path(sys.prefix).parents[1] / "Scripts/conda.exe")
    conda = Path(conda)
    assert conda.is_file(), "Conda executable not found; report this error"

    capture(
        [str(conda), "list", "--name", ENV, "--json"],
        "conda-packages.json",
    )
    capture(
        [str(conda), "list", "--name", ENV, "--explicit"],
        "conda-explicit-win-64.txt",
    )
    capture(
        [str(conda), "env", "export", "--name", ENV],
        "environment.yml",
    )
    capture(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        "pip-freeze.txt",
    )
    capture(
        [sys.executable, "-m", "pip", "list", "--format=json"],
        "pip-packages.json",
    )
    print("Conda/pip environment exports: PASS", flush=True)

    capture(
        ["git", "-C", str(PROJECT), "rev-parse", "HEAD"],
        "outer-git-head.txt",
    )
    capture(
        ["git", "-C", str(PROJECT), "status", "--short",
         "--untracked-files=all", "--",
         "README.md", "docs", "toolkit-repro/scripts",
         "toolkit-repro/visual-rag-toolkit"],
        "outer-git-status.txt",
    )
    capture(
        ["git", "-C", str(PROJECT), "diff", "--binary", "HEAD", "--",
         "toolkit-repro/scripts", "toolkit-repro/visual-rag-toolkit"],
        "working-changes.patch",
    )

    # Snapshot source files, including scripts not yet tracked by Git.
    # This makes the snapshot independent of git diff's untracked-file limit.
    source_manifest = []
    archive = record / "source-snapshot.zip"
    with zipfile.ZipFile(
        archive, "x", compression=zipfile.ZIP_DEFLATED
    ) as output:
        for folder in (
            ROOT / "scripts",
            ROOT / "visual-rag-toolkit",
        ):
            assert folder.is_dir()
            for directory, dirs, files in os.walk(folder, followlinks=False):
                parent = Path(directory)
                for name in list(dirs):
                    path = parent / name
                    if (
                        name in {".git", "__pycache__", ".pytest_cache"}
                        or path.is_symlink()
                        or bool(
    getattr(path.lstat(), "st_file_attributes", 0) & 0x400
)
                    ):
                        dirs.remove(name)

                for name in sorted(files):
                    path = parent / name
                    if path.is_symlink():
                        continue
                    if (
                        name.endswith((".pyc", ".pyo"))
                        or name == ".env"
                        or name.startswith(".env.")
                    ):
                        continue
                    relative = path.relative_to(ROOT).as_posix()
                    content = path.read_bytes()
                    output.writestr(relative, content)
                    source_manifest.append({
                        "path": relative,
                        "bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    })

        for name in (
            "toolkit-upstream.txt",
            "core-constraints-v1.txt",
            "core-constraints-v2.txt",
        ):
            path = ROOT / name
            assert path.is_file(), name
            content = path.read_bytes()
            output.writestr(name, content)
            source_manifest.append({
                "path": name,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            })

    with zipfile.ZipFile(archive) as saved:
        assert saved.testzip() is None
        for item in source_manifest:
            assert hashlib.sha256(saved.read(item["path"])).hexdigest() == (
                item["sha256"]
            )
    write_json("source-manifest.json", source_manifest)

    # Preserve existing documentation before the manual README update.
    shutil.copyfile(
        PROJECT / "README.md",
        record / "README.before-closeout.md",
    )
    shutil.copyfile(
        PROJECT / "复现执行计划与验收标准.md",
        record / "execution-plan-reference.md",
    )

    storage = reports["storage"]["collection_storage"]
    timing = reports["timing"]["timing_summary"]["combined_three_rounds"]
    encoding = reports["encoding"]

    summary = {
        group: {
            "metrics": reports[group]["aggregate_metrics"],
            "cached_search_timing": timing[group],
            "collection_storage": storage[group],
            "evaluation_record": str(SOURCES[group]),
        }
        for group in ("A", "B", "C")
    }
    write_json("results-evidence-summary.json", summary)

    ledger = {
        "encoding_record": str(SOURCES["encoding"]),
        "supplemental_encoding_scope": {
            "new_pages": encoding["new_pages"],
            "new_queries": encoding["new_queries"],
            "reused_counts": encoding["reused_counts"],
        },
        "supplemental_encoding_timings": encoding["timing_totals_seconds"],
        "encoding_peak_allocated_bytes": encoding["peak_allocated_bytes"],
        "encoding_peak_reserved_bytes": encoding["peak_reserved_bytes"],
        "encoding_sampled_python_rss": encoding["ram_sampling"],
        "A_recovery_index_prepare_and_readback_seconds": (
            reports["A"]["index_prepare_and_readback_seconds"]
        ),
        "B_build_upload_and_readback_seconds": (
            reports["B"]["build_upload_and_readback_seconds"]
        ),
        "C_index_prepare_and_readback_seconds": (
            reports["C"]["index_prepare_and_readback_seconds"]
        ),
        "measurement_limits": [
            "444-page/230-query encoding timings are supplemental-run timings.",
            "Earlier 8 pages and 2 queries have separate diagnostic provenance.",
            "A failed upload and recovery are separate records; "
            "there is no single uninterrupted from-zero A build measurement.",
            "Cached-query search is measured; true request-level "
            "query-encoding-plus-search E2E is not measured.",
            "Docker memory snapshots are not sampled peak Qdrant memory.",
            "Segment storage mixes vectors, payload and index/mapping files; "
            "do not label all segment bytes as vector-only bytes.",
            "Runtime file accounting excludes outer docs/Git, Conda and "
            "Docker images; it is not whole-machine physical disk usage.",
        ],
        "candidate_provenance": reports["B"]["candidate_note"],
    }
    write_json("resource-ledger.json", ledger)

    report["evidence_records"] = len(evidence)
    report["source_snapshot_files"] = len(source_manifest)
    report["source_snapshot_sha256"] = digest(archive)
    report["known_measurement_limits"] = ledger["measurement_limits"]
    report["artifact_sha256"] = {
        path.name: digest(path)
        for path in record.iterdir()
        if path.is_file() and path.name != "report.json"
    }
    report["status"] = "PASS"

except BaseException as exc:
    report["status"] = (
        "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED"
    )
    report["error"] = repr(exc)
    raise

finally:
    save()

print("Evidence records:", report["evidence_records"])
print("Source snapshot files:", report["source_snapshot_files"])
print("Core versions:", report["core_versions"])
print("Resource measurement limits: recorded")
print("Record:", record)
print("P3 environment/source/evidence freeze: PASS")