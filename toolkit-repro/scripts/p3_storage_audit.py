import json
import os
import re
import stat
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path, PurePosixPath

from qdrant_client import QdrantClient

from prb_cache import digest, json_bytes


ROOT = Path(__file__).resolve().parents[1]
TIMING = ROOT / "records/p3-search-timing-20261107-014826-ca6af3"
SOURCES = {
    "A": ROOT / "records/p2-formal-a-20261107-012250-612a76",
    "B": ROOT / "records/p2-formal-b-20261107-013537-34e55b",
    "C": ROOT / "records/p2-formal-c-20261107-014210-68a2ce",
}
CONTAINER = "prb-qdrant-1141"
VOLUME = "prb-qdrant-1141-data"

record = ROOT / "records" / (
    "p3-storage-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    + "-" + uuid.uuid4().hex[:6]
)
record.mkdir(parents=True, exist_ok=False)

report = {
    "status": "STARTED",
    "container": CONTAINER,
    "volume": VOLUME,
    "measurement_state": "Running container; read-only inspection",
    "measurement_notes": [
        "du allocated bytes include filesystem blocks and directory overhead.",
        "du apparent bytes describe file lengths, including sparse-file holes.",
        "Linux filesystem allocation is not Windows Docker VHDX file size.",
        "Whole volume includes historical diagnostic collections.",
        "Collection totals are already included in whole-volume totals.",
        "Segment files may mix vectors, payload, mappings and indexes.",
        "Windows runtime inventory uses logical file lengths, not NTFS allocation.",
    ],
}
client = None


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(name, value):
    (record / name).write_bytes(json_bytes(value))


def save():
    write_json("report.json", report)


def command(args, output_name=None):
    result = subprocess.run(
        args,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    if output_name:
        (record / output_name).write_bytes(
            result.stdout.encode("utf-8")
        )
    if result.returncode != 0:
        raise RuntimeError({
            "command": args,
            "exit_code": result.returncode,
            "stderr": result.stderr,
            "stdout": result.stdout,
        })
    return result.stdout.strip()


def docker_format(template):
    return command([
        "docker", "inspect", "--format", template, CONTAINER
    ])


def exec_container(*args, output_name=None):
    return command(
        ["docker", "exec", CONTAINER, *args],
        output_name=output_name,
    )


def du_bytes(path, apparent=False):
    args = ["du", "-B1", "-s"]
    if apparent:
        args.append("--apparent-size")
    args.extend(["--", str(path)])
    output = exec_container(*args)
    assert len(output.splitlines()) == 1
    return int(output.split()[0])


def file_inventory(path, output_name):
    output = exec_container(
        "find", str(path), "-xdev", "-type", "f",
        "-printf", r"%P\t%s\t%b\t%D\t%i\t%n\t%T@\n",
        output_name=output_name,
    )
    entries = {}
    for line in output.splitlines():
        fields = line.split("\t")
        assert len(fields) == 7, ("Unexpected filename/output", line)
        relative, size, blocks, device, inode, links, mtime = fields
        assert relative not in entries
        entries[relative] = {
            "file_length_bytes": int(size),
            "allocated_file_bytes": int(blocks) * 512,
            "device": device,
            "inode": inode,
            "hardlink_count": int(links),
            "mtime": mtime,
        }
    assert entries, ("No regular files found", str(path))
    return entries


def classify_files(entries):
    # These are path-based categories, not inferred pure vector sizes.
    sections = {}
    seen_inodes = {}
    cross_category_links = []

    for relative, item in sorted(entries.items()):
        parts = PurePosixPath(relative).parts
        if "wal" in parts:
            section = "wal_files"
        elif "segments" in parts:
            section = "segment_files_mixed"
        else:
            section = "metadata_and_other_files"

        key = (item["device"], item["inode"])
        if key in seen_inodes:
            if seen_inodes[key] != section:
                cross_category_links.append(relative)
            continue
        seen_inodes[key] = section

        bucket = sections.setdefault(section, {
            "unique_regular_files": 0,
            "file_length_bytes": 0,
            "allocated_file_bytes": 0,
        })
        bucket["unique_regular_files"] += 1
        bucket["file_length_bytes"] += item["file_length_bytes"]
        bucket["allocated_file_bytes"] += item["allocated_file_bytes"]

    return {
        "sections": sections,
        "cross_category_hardlinks": cross_category_links,
        "note": (
            "Regular files only; directory overhead is excluded here. "
            "Use collection du totals for overall filesystem allocation. "
            "segment_files_mixed is not a vector-only or payload-only size."
        ),
    }


def collection_check(group):
    name = collections[group]
    info = client.get_collection(name)
    actual = json.loads(info.model_dump_json())
    baseline = load(SOURCES[group] / "collection-after.json")
    assert actual["config"] == baseline["config"], group
    assert info.status.value == "green", (group, info.status)
    assert info.indexed_vectors_count == 0
    assert client.count(name, exact=True).count == 452

    expected_ids = {
        item["point_id"]
        for item in load(SOURCES[group] / "page-id-mapping.json")
    }
    seen = set()
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=name,
            offset=offset,
            limit=128,
            with_payload=False,
            with_vectors=False,
        )
        for point in batch:
            pid = str(point.id)
            assert pid not in seen
            seen.add(pid)
        if offset is None:
            break
    assert seen == expected_ids
    return actual


def runtime_inventory():
    totals = {}
    excluded = []

    # Only the new, isolated toolkit-repro tree is traversed.
    for directory, dirs, files in os.walk(ROOT, followlinks=False):
        parent = Path(directory)

        for name in list(dirs):
            path = parent / name
            relative = path.relative_to(ROOT).as_posix()
            attributes = path.lstat()
            reparse = bool(
                getattr(attributes, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            )
            if (
                path == record
                or path.is_symlink()
                or reparse
                or "layoutpatch" in name.lower()
            ):
                dirs.remove(name)
                excluded.append(relative)

        for name in files:
            path = parent / name
            relative = path.relative_to(ROOT)
            attributes = path.lstat()
            reparse = bool(
                getattr(attributes, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            )
            if (
                path.is_symlink()
                or reparse
                or not stat.S_ISREG(attributes.st_mode)
            ):
                excluded.append(relative.as_posix())
                continue

            first = relative.parts[0]
            category = {
                "models": "models",
                "data": "data",
                "cache": "research_cache",
                "records": "historical_records",
                "visual-rag-toolkit": "author_source",
                "scripts": "scripts",
            }.get(first, "other_runtime_files")
            bucket = totals.setdefault(category, {
                "files": 0,
                "logical_file_bytes": 0,
            })
            bucket["files"] += 1
            bucket["logical_file_bytes"] += attributes.st_size

    return {
        "root": str(ROOT),
        "categories": totals,
        "total_logical_file_bytes": sum(
            value["logical_file_bytes"] for value in totals.values()
        ),
        "excluded_paths": excluded,
        "note": (
            "Current storage-audit record excluded to avoid self-growth. "
            "Does not include outer project docs/Git, Conda environment, "
            "Docker images or the Qdrant volume. "
            "Not Windows physical allocated disk usage."
        ),
    }


save()
print("Record:", record, flush=True)

try:
    assert Path(sys.prefix).name.lower() == "page-retrieval-budget"

    timing_report = load(TIMING / "report.json")
    assert timing_report["status"] == "PASS"
    assert timing_report["completed_measured_blocks"] == 9
    for name, expected_hash in timing_report["artifact_sha256"].items():
        assert digest(TIMING / name) == expected_hash, name

    collections = timing_report["collections"]
    report["source_timing_record"] = str(TIMING)
    report["source_timing_report_sha256"] = digest(TIMING / "report.json")
    report["collections"] = collections

    accepted = {}
    for group, source in SOURCES.items():
        previous = load(source / "report.json")
        assert previous["status"] == "PASS"
        assert previous["collection"] == collections[group]
        for name, expected_hash in previous["artifact_sha256"].items():
            assert digest(source / name) == expected_hash, (group, name)
        accepted[group] = previous

    assert accepted["C"]["points_with_initial_field"] == 0
    for name in collections.values():
        assert re.fullmatch(r"prb_p2_[abc]_[0-9a-f]+", name)

    state_before = json.loads(docker_format("{{json .State}}"))
    assert state_before["Running"] is True
    assert state_before["OOMKilled"] is False
    mounts = json.loads(docker_format("{{json .Mounts}}"))
    write_json("docker-mounts.json", mounts)
    write_json("docker-state-before.json", state_before)
    report["container_image_id"] = docker_format("{{.Image}}")

    matches = [
        mount for mount in mounts
        if mount["Type"] == "volume" and mount.get("Name") == VOLUME
    ]
    assert len(matches) == 1, "Expected dedicated named volume not found"
    destination = PurePosixPath(matches[0]["Destination"])
    assert destination.is_absolute()
    assert str(destination) != "/"
    assert ".." not in destination.parts
    report["volume_mount_destination"] = str(destination)

    du_version = exec_container(
        "du", "--version", output_name="du-version.txt"
    )
    find_version = exec_container(
        "find", "--version", output_name="find-version.txt"
    )
    assert "GNU" in du_version and "GNU" in find_version, (
        "This collector requires GNU du/find; do not install or substitute "
        "tools automatically. Report the saved version output."
    )

    client = QdrantClient(
        url="http://127.0.0.1:16333",
        timeout=60,
        trust_env=False,
    )
    assert client.info().version == "1.14.1"

    volume_before = {
        "allocated_bytes": du_bytes(destination),
        "apparent_bytes": du_bytes(destination, apparent=True),
    }
    before_inventories = {}
    measurements = {}
    inode_groups = {}

    for group in ("A", "B", "C"):
        report["current_group"] = group
        save()
        info = collection_check(group)
        write_json(group + "-collection-before.json", info)

        path = destination / "collections" / collections[group]
        exec_container("test", "-d", str(path))
        entries = file_inventory(path, group + "-files-before.tsv")
        before_inventories[group] = entries

        for item in entries.values():
            key = (item["device"], item["inode"])
            inode_groups.setdefault(key, set()).add(group)

        measurements[group] = {
            "collection": collections[group],
            "container_path": str(path),
            "points": 452,
            "actual_fields": sorted(info["config"]["params"]["vectors"]),
            "theoretical_vector_bytes": accepted[group]["theoretical_vector_bytes"],
            "du_allocated_bytes": du_bytes(path),
            "du_apparent_bytes": du_bytes(path, apparent=True),
            "file_breakdown": classify_files(entries),
        }
        write_json(group + "-file-inventory.json", entries)
        exec_container(
            "du", "-B1", "--max-depth=5", "--", str(path),
            output_name=group + "-directory-allocated-bytes.tsv",
        )
        print(
            group,
            "allocated MiB:",
            round(measurements[group]["du_allocated_bytes"] / 1024**2, 3),
            "apparent MiB:",
            round(measurements[group]["du_apparent_bytes"] / 1024**2, 3),
            flush=True,
        )

    shared = [
        {"device": key[0], "inode": key[1], "groups": sorted(groups)}
        for key, groups in inode_groups.items() if len(groups) > 1
    ]
    write_json("cross-collection-shared-inodes.json", shared)
    assert not shared, (
        "Cross-collection hardlinks found; attribution needs review"
    )

    # Recheck after every collection has been inspected.
    for group in ("A", "B", "C"):
        path = PurePosixPath(measurements[group]["container_path"])
        after_entries = file_inventory(path, group + "-files-after.tsv")
        assert after_entries == before_inventories[group], (
            "Collection files changed during measurement", group
        )
        assert du_bytes(path) == measurements[group]["du_allocated_bytes"]
        assert du_bytes(path, apparent=True) == (
            measurements[group]["du_apparent_bytes"]
        )
        write_json(
            group + "-collection-after.json",
            collection_check(group),
        )
        measurements[group]["files_stable_during_measurement"] = True

    volume_after = {
        "allocated_bytes": du_bytes(destination),
        "apparent_bytes": du_bytes(destination, apparent=True),
    }
    report["whole_volume"] = {
        "before": volume_before,
        "after": volume_after,
        "stable_totals": volume_before == volume_after,
        "note": (
            "Includes A/B/C and all retained historical collections "
            "plus other files under this mounted storage directory."
        ),
    }

    state_after = json.loads(docker_format("{{json .State}}"))
    assert state_after["Running"] is True
    assert state_after["OOMKilled"] is False
    assert state_before["StartedAt"] == state_after["StartedAt"]
    write_json("docker-state-after.json", state_after)

    runtime = runtime_inventory()
    write_json("runtime-file-accounting.json", runtime)
    write_json("collection-storage.json", measurements)

    report["collection_storage"] = measurements
    report["runtime_file_accounting"] = runtime
    report["actual_storage_ratios"] = {
        "A_over_C_allocated": (
            measurements["A"]["du_allocated_bytes"]
            / measurements["C"]["du_allocated_bytes"]
        ),
        "A_over_C_apparent": (
            measurements["A"]["du_apparent_bytes"]
            / measurements["C"]["du_apparent_bytes"]
        ),
    }
    report["code_sha256"] = {
        "scripts/p3_storage_audit.py": digest(Path(__file__)),
        "scripts/prb_cache.py": digest(ROOT / "scripts/prb_cache.py"),
    }
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
    if client is not None:
        client.close()
    save()

print("Collection files stable: PASS")
print("Cross-collection shared inodes:", len(shared))
print("Actual storage ratios:", report["actual_storage_ratios"])
print("Whole Qdrant volume:", report["whole_volume"])
print("Runtime file categories:", runtime["categories"])
print("Runtime total logical bytes:", runtime["total_logical_file_bytes"])
print("Record:", record)
print("P3 collection storage audit: PASS")