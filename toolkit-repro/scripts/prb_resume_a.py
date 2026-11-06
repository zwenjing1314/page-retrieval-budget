import numpy as np
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
