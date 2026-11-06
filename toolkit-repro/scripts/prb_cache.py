import hashlib
import json
import os
import uuid
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2
    ).encode("utf-8")


class CacheStore:
    """Immutable per-item cache. A completed directory is the commit marker."""

    def __init__(self, directory, profile, create=False):
        self.root = Path(directory)
        self.profile = profile
        self.profile_hash = hashlib.sha256(json_bytes(profile)).hexdigest()
        identity = self.root / "profile.json"
        if not identity.exists():
            assert create, "Cache profile is missing"
            self.root.mkdir(parents=True, exist_ok=True)
            with identity.open("xb") as f:
                f.write(json_bytes(profile))
        assert json.loads(identity.read_text(encoding="utf-8")) == profile, (
            "Cache identity differs; do not reuse or overwrite it"
        )

    def path(self, kind, item_id):
        assert kind in ("pages", "queries")
        value = str(item_id)
        assert value.isdigit(), "Expected a non-negative official numeric ID"
        return self.root / kind / value

    def exists(self, kind, item_id):
        return self.path(kind, item_id).exists()

    def validate(self, kind, arrays):
        vectors = arrays["vectors"]
        assert vectors.dtype == np.float16
        assert vectors.ndim == 2 and vectors.shape[1] == 128
        assert len(vectors) > 0 and np.isfinite(vectors).all()
        norms = np.linalg.norm(vectors.astype(np.float32), axis=1)
        assert np.max(np.abs(norms - 1)) <= 0.005

        ids = arrays["input_ids"].reshape(-1)
        attention = arrays["attention_mask"].reshape(-1).astype(bool)
        keep = arrays["keep_mask"].reshape(-1)
        assert keep.dtype == np.bool_
        assert ids.shape == attention.shape == keep.shape

        if kind == "pages":
            expected = ids == self.profile["image_token_id"]
            assert vectors.shape == (1024, 128)
            assert arrays["raw"].ndim == 2
            assert arrays["raw"].shape == (len(ids), 128)
            assert np.isfinite(arrays["raw"]).all()
            assert np.array_equal(vectors, arrays["raw"][keep])
            assert arrays["rows"].shape == (32, 128)
            assert arrays["pooled"].shape == (34, 128)
            for key in ("rows", "pooled"):
                assert arrays[key].dtype == np.float16
                assert np.isfinite(arrays[key]).all()
        else:
            expected = attention.copy()
            if (ids[attention] >= 4).any():
                expected &= ids >= 4

        assert np.array_equal(keep, expected)
        assert int(keep.sum()) == len(vectors)

    def read(self, kind, item_id):
        directory = self.path(kind, item_id)
        meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        assert meta["kind"] == kind and meta["id"] == str(item_id)
        assert meta["profile_sha256"] == self.profile_hash
        file = directory / "vectors.npz"
        assert digest(file) == meta["npz_sha256"], f"Damaged cache: {file}"
        with np.load(file, allow_pickle=False) as saved:
            arrays = {name: saved[name].copy() for name in saved.files}
        self.validate(kind, arrays)
        return arrays, meta

    def put(self, kind, item_id, arrays, identity):
        self.validate(kind, arrays)
        destination = self.path(kind, item_id)
        assert not destination.exists(), f"Refusing overwrite: {destination}"
        destination.parent.mkdir(parents=True, exist_ok=True)

        # Interrupted writes remain in a distinct pending directory.
        pending = destination.parent / (
            "_pending-" + str(item_id) + "-" + uuid.uuid4().hex
        )
        pending.mkdir(exist_ok=False)
        file = pending / "vectors.npz"
        np.savez(file, **arrays)
        meta = {
            "kind": kind,
            "id": str(item_id),
            "profile_sha256": self.profile_hash,
            "npz_sha256": digest(file),
            "identity": identity,
            "arrays": {
                key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for key, value in arrays.items()
            },
        }
        (pending / "metadata.json").write_bytes(json_bytes(meta))
        os.rename(pending, destination)
        self.read(kind, item_id)

    def manifest(self):
        result = {
            "cache_directory": str(self.root),
            "profile_sha256": self.profile_hash,
            "expected_counts": {
                "pages": len(self.profile["expected_page_ids"]),
                "queries": len(self.profile["expected_query_ids"]),
            },
            "completed": {},
            "pending_directories": [],
        }
        for kind in ("pages", "queries"):
            entries = []
            parent = self.root / kind
            if parent.exists():
                for directory in sorted(parent.iterdir()):
                    if directory.name.startswith("_pending-"):
                        result["pending_directories"].append(str(directory))
                    elif directory.is_dir() and directory.name.isdigit():
                        _, meta = self.read(kind, directory.name)
                        entries.append({
                            "id": directory.name,
                            "npz_sha256": meta["npz_sha256"],
                        })
            actual = {entry["id"] for entry in entries}
            expected = set(self.profile[
                "expected_page_ids" if kind == "pages" else "expected_query_ids"
            ])
            assert actual <= expected
            result["completed"][kind] = entries
        result["complete"] = all(
            len(result["completed"][kind]) == result["expected_counts"][kind]
            for kind in ("pages", "queries")
        )
        return result