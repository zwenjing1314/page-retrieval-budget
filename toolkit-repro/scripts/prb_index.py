import numpy as np
from qdrant_client import models

FIELD_SETS = {
    "A": ("initial",),
    "B": ("initial", "experimental_pooling_3"),
    "C": ("experimental_pooling_3",),
}


def build_collection(client, name, mode, points):
    """Create a new isolated collection from cached vectors.

    points: list of {id, payload, vectors}.
    Never overwrites an existing collection.
    """
    fields = FIELD_SETS[mode]
    ids = [str(p["id"]) for p in points]
    assert points and len(set(ids)) == len(ids)
    if client.collection_exists(name):
        raise RuntimeError(f"Collection already exists: {name}")

    # Validate all inputs before creating the collection.
    for point in points:
        for field in fields:
            array = np.asarray(point["vectors"][field])
            expected_rows = 1024 if field == "initial" else 34
            assert array.shape == (expected_rows, 128), (field, array.shape)
            assert array.dtype == np.float16, (field, array.dtype)
            assert np.isfinite(array).all(), field
            assert np.all(np.linalg.norm(array.astype(np.float32), axis=1) > 0)

    client.create_collection(
        collection_name=name,
        vectors_config={
            field: models.VectorParams(
                size=128,
                distance=models.Distance.COSINE,
                datatype=models.Datatype.FLOAT16,
                on_disk=False,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
            )
            for field in fields
        },
        hnsw_config=models.HnswConfigDiff(
            m=0, full_scan_threshold=10, on_disk=False
        ),
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
        on_disk_payload=False,
    )

    # Upload one point at a time; any exception immediately propagates.
    for point in points:
        client.upsert(
            collection_name=name,
            wait=True,
            points=[
                models.PointStruct(
                    id=point["id"],
                    vector={
                        field: point["vectors"][field].tolist()
                        for field in fields
                    },
                    payload=point["payload"],
                )
            ],
        )

    info = client.get_collection(name)
    assert set(info.config.params.vectors) == set(fields)
    for vector in info.config.params.vectors.values():
        assert vector.size == 128
        assert vector.distance == models.Distance.COSINE
        assert vector.datatype == models.Datatype.FLOAT16
        assert vector.on_disk is False
        assert vector.multivector_config.comparator == models.MultiVectorComparator.MAX_SIM
    assert info.config.hnsw_config.m == 0
    assert info.config.hnsw_config.full_scan_threshold == 10
    assert info.config.hnsw_config.on_disk is False
    assert info.config.optimizer_config.indexing_threshold == 0
    assert info.config.params.on_disk_payload is False
    assert info.indexed_vectors_count == 0
    assert client.count(name, exact=True).count == len(points)

    # Compare complete ID sets, payload and actual vector fields.
    expected = {str(p["id"]): p for p in points}
    seen = set()
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=name,
            offset=offset,
            limit=16,
            with_vectors=True,
            with_payload=True,
        )
        for stored in batch:
            key = str(stored.id)
            assert key in expected and key not in seen
            seen.add(key)
            assert stored.payload == expected[key]["payload"]
            assert set(stored.vector) == set(fields)
            for field in fields:
                rows = 1024 if field == "initial" else 34
                array = np.asarray(stored.vector[field])
                assert array.shape == (rows, 128)
                assert np.isfinite(array).all()
        if offset is None:
            break
    assert seen == set(expected)
    return info