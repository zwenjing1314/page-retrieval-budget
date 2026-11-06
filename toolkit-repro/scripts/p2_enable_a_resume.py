import json
import uuid
from datetime import datetime
from pathlib import Path

from prb_cache import digest, json_bytes

ROOT = Path(__file__).resolve().parents[1]
driver = ROOT / "scripts/p2_evaluate_a.py"
helper = ROOT / "scripts/prb_resume_a.py"

assert driver.is_file(), "Missing A driver"
assert not helper.exists(), "Resume helper exists; stop without overwriting"

original = driver.read_bytes()
text = original.decode("utf-8-sig").replace("\r\n", "\n")
assert "trust_env=False" in text, "Apply direct-connection fix first"

helper_source = r'''import numpy as np
from qdrant_client import models

from prb_cache import json_bytes


def check_config(client, name):
    info = client.get_collection(name)
    assert set(info.config.params.vectors) == {"initial"}
    vector = info.config.params.vectors["initial"]
    assert vector.size == 128
    assert vector.distance == models.Distance.COSINE
    assert vector.datatype == models.Datatype.FLOAT16
    assert vector.on_disk is False
    assert vector.multivector_config.comparator == (
        models.MultiVectorComparator.MAX_SIM
    )
    assert vector.quantization_config is None
    assert info.config.quantization_config is None
    assert info.config.hnsw_config.m == 0
    assert info.config.hnsw_config.full_scan_threshold == 10
    assert info.config.hnsw_config.on_disk is False
    assert info.config.optimizer_config.indexing_threshold == 0
    assert info.config.params.on_disk_payload is False
    assert info.indexed_vectors_count == 0
    return info


def readback(client, name, expected):
    seen = set()
    offset = None
    max_error = 0.0

    while True:
        batch, offset = client.scroll(
            collection_name=name,
            offset=offset,
            limit=4,
            with_payload=True,
            with_vectors=True,
        )
        for stored in batch:
            key = str(stored.id)
            assert key in expected, ("Unexpected point", key)
            assert key not in seen, ("Duplicate point", key)
            seen.add(key)
            point = expected[key]
            assert stored.payload == point["payload"], (
                "Payload differs", key
            )
            assert set(stored.vector) == {"initial"}
            actual = np.asarray(
                stored.vector["initial"], dtype=np.float32
            )
            assert actual.shape == (1024, 128)
            assert np.isfinite(actual).all()

            reference = point["vectors"]["initial"].astype(np.float32)
            reference /= np.linalg.norm(
                reference, axis=1, keepdims=True
            )
            # Qdrant cosine vectors are normalized, then stored in FP16.
            error = float(np.max(np.abs(actual - reference)))
            assert error <= 1e-3, (
                "Stored vector differs from normalized cache", key, error
            )
            max_error = max(max_error, error)

        if offset is None:
            break

    assert client.count(name, exact=True).count == len(seen)
    return seen, max_error


def resume_collection_a(client, name, points, record):
    assert client.collection_exists(name), "Recovery collection is missing"
    expected = {str(p["id"]): p for p in points}
    assert len(points) == len(expected) == 452

    for point in points:
        array = point["vectors"]["initial"]
        assert array.shape == (1024, 128)
        assert array.dtype == np.float16
        assert np.isfinite(array).all()

    info = check_config(client, name)
    (record / "resume-before-collection.json").write_bytes(
        info.model_dump_json(indent=2).encode("utf-8")
    )

    # Validate every existing point before uploading anything.
    seen, before_error = readback(client, name, expected)
    missing = [p for p in points if str(p["id"]) not in seen]
    recovery = {
        "reused_points": len(seen),
        "missing_points_before": len(missing),
        "uploaded_points": 0,
        "existing_point_ids": sorted(seen),
        "missing_corpus_ids": [
            p["payload"]["corpus_id"] for p in missing
        ],
        "vector_readback_max_abs_error_before": before_error,
        "vector_readback_component_tolerance": 1e-3,
        "note": (
            "Existing vectors checked against normalized cached vectors. "
            "Only missing IDs uploaded; no automatic retries or deletions."
        ),
    }

    def save():
        (record / "resume-index.json").write_bytes(
            json_bytes(recovery)
        )

    save()
    print("Verified existing points:", len(seen), flush=True)
    print("Missing points:", len(missing), flush=True)

    for number, point in enumerate(missing, 1):
        # Stop if another process has inserted this ID meanwhile.
        already = client.retrieve(
            collection_name=name,
            ids=[point["id"]],
            with_payload=False,
            with_vectors=False,
        )
        assert not already, "Point appeared concurrently; stop"

        client.upsert(
            collection_name=name,
            wait=True,
            points=[models.PointStruct(
                id=point["id"],
                payload=point["payload"],
                vector={
                    "initial": point["vectors"]["initial"].tolist()
                },
            )],
        )
        recovery["uploaded_points"] = number
        save()
        if number == 1 or number % 20 == 0 or number == len(missing):
            print(
                f"Recovered total: {len(seen) + number}/452",
                flush=True,
            )

    final_info = check_config(client, name)
    final_ids, final_error = readback(client, name, expected)
    assert final_ids == set(expected)
    recovery["final_points"] = len(final_ids)
    recovery["vector_readback_max_abs_error_after"] = final_error
    recovery["status"] = "PASS"
    save()
    print("Recovered full 452-page index: PASS", flush=True)
    return final_info, recovery
'''


def replace_once(old, new):
    global text
    assert text.count(old) == 1, (
        "Driver differs from expected; no files changed",
        old[:100],
        text.count(old),
    )
    text = text.replace(old, new, 1)


replace_once("import csv\n", "import argparse\nimport csv\n")

replace_once(
    "from prb_index import build_collection\n",
    "from prb_index import build_collection\n"
    "from prb_resume_a import resume_collection_a\n",
)

replace_once(
    '\nrecord = ROOT / "records" / (\n',
    '''
parser = argparse.ArgumentParser()
parser.add_argument("--resume-record", type=Path)
args = parser.parse_args()
resume_source = None
resume_report = None

if args.resume_record is not None:
    resume_source = args.resume_record.resolve()
    assert resume_source.parent == (ROOT / "records").resolve()
    resume_report = json.loads(
        (resume_source / "report.json").read_text(encoding="utf-8")
    )
    assert resume_report["status"] in ("FAILED", "INTERRUPTED")
    assert resume_report["group"] == "A"
    assert resume_report["stage"] == "build_and_readback"
    assert resume_report["completed_queries"] == 0
    assert resume_report["stored_fields"] == ["initial"]
    assert resume_report["retrieval_mode"] == "single_full"

record = ROOT / "records" / (
''',
)

replace_once(
    '    "collection": "prb_p2_a_" + uuid.uuid4().hex[:12],\n',
    '''    "collection": (
        resume_report["collection"] if resume_report
        else "prb_p2_a_" + uuid.uuid4().hex[:12]
    ),
''',
)

replace_once(
    '        ROOT / "scripts/prb_index.py",\n',
    '        ROOT / "scripts/prb_index.py",\n'
    '        ROOT / "scripts/prb_resume_a.py",\n',
)

replace_once(
    '''    info = build_collection(
        checked, report["collection"], "A", points
    )
''',
    '''    if resume_report is not None:
        assert resume_report["cache_profile_sha256"] == store.profile_hash
        assert resume_report["counts"] == report["counts"]
        assert resume_report["source_report_sha256"] == (
            report["source_report_sha256"]
        )
        assert resume_report["versions"] == report["versions"]
        for filename in ("qrels.json", "page-id-mapping.json"):
            assert digest(resume_source / filename) == digest(record / filename)
        for relative, expected in resume_report["code_sha256"].items():
            normalized = relative.replace("\\\\", "/")
            if normalized != "scripts/p2_evaluate_a.py":
                assert digest(ROOT / normalized) == expected, (
                    "Source changed since failed run", relative
                )

        report["resume_source_record"] = str(resume_source)
        report["resume_source_report_sha256"] = digest(
            resume_source / "report.json"
        )
        report["connection_trust_env"] = False
        report["index_preparation_note"] = (
            "This run verifies existing points and uploads missing points. "
            "Earlier upload work is recorded in the original failed run."
        )
        save()
        info, recovery = resume_collection_a(
            checked, report["collection"], points, record
        )
        report["resume"] = recovery
    else:
        info = build_collection(
            checked, report["collection"], "A", points
        )
''',
)

replace_once(
    '    report["build_upload_and_readback_seconds"] = time.perf_counter() - t\n',
    '    report["index_prepare_and_readback_seconds"] = time.perf_counter() - t\n',
)

replace_once(
    "    assert checked.uploads == 452\n",
    '''    assert (
        checked.uploads
        + report.get("resume", {}).get("reused_points", 0)
    ) == 452
''',
)

replace_once(
    '            print(f"Uploaded: {self.uploads}/452", flush=True)\n',
    '            print(f"Uploaded this run: {self.uploads}", flush=True)\n',
)

# Check syntax before changing either file. This does not execute the driver.
compile(helper_source, str(helper), "exec")
compile(text, str(driver), "exec")
assert driver.read_bytes() == original, "Driver changed during preparation"

record = ROOT / "records" / (
    "p2-enable-a-resume-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    + "-" + uuid.uuid4().hex[:6]
)
record.mkdir(parents=True, exist_ok=False)
(record / "p2_evaluate_a.before-resume.py").write_bytes(original)

with helper.open("xb") as output:
    output.write(helper_source.encode("utf-8"))
driver.write_bytes(text.encode("utf-8"))

(record / "report.json").write_bytes(json_bytes({
    "status": "PASS",
    "driver_before_sha256": digest(
        record / "p2_evaluate_a.before-resume.py"
    ),
    "driver_after_sha256": digest(driver),
    "helper_sha256": digest(helper),
    "patch_script_sha256": digest(Path(__file__)),
    "note": "Original driver backed up; explicit A recovery option added",
}))

print("Backup record:", record)
print("Explicit A resume support: PASS")