import ast
import hashlib
import json
import subprocess
import uuid
from datetime import datetime
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = root / "visual-rag-toolkit"
embedder = source / "visual_rag/embedding/visual_embedder.py"
runner = source / "benchmarks/vidore_beir_qdrant/run_qdrant_beir.py"

originals = {p: p.read_bytes() for p in (embedder, runner)}
texts = {
    p: raw.decode("utf-8-sig").replace("\r\n", "\n")
    for p, raw in originals.items()
}

# Require previous patches.
loader = source / "benchmarks/vidore_tatdqa_test/dataset_loader.py"
loader_text = loader.read_text(encoding="utf-8-sig")
assert "PRB_ECONOMICS_SNAPSHOT" in loader_text
assert "def _first_present(" in loader_text
assert "Incomplete index:" in texts[runner]

# Add an explicit fixed-local loading branch before generic discovery.
text = texts[embedder]
assert "PRB_COLPALI_BASE" not in text, "Model patch already exists"
marker = '        """Load ColPali-family model."""\n'
assert text.count(marker) == 1

branch = '''
        local_base = os.getenv("PRB_COLPALI_BASE")
        if local_base:
            from transformers import AutoTokenizer, SiglipImageProcessor
            from colpali_engine.models import ColPali, ColPaliProcessor

            adapter = Path(self.model_name).resolve()
            base = Path(local_base).resolve()
            identities = [
                (adapter, "vidore/colpali-v1.3",
                 "b5c6dd62125326e6f0b540c1f7b36e901cdc0a11"),
                (base, "vidore/colpaligemma-3b-pt-448-base",
                 "30ab955d073de4a91dc5a288e8c97226647e3e5a"),
            ]
            for directory, repo, revision in identities:
                identity = json.loads(
                    (directory / "download-manifest.json").read_text(
                        encoding="utf-8"
                    )
                )
                if identity["repo"] != repo or identity["revision"] != revision:
                    raise ValueError("Unexpected fixed model identity")

            if self.torch_dtype != torch.float16:
                raise ValueError("Fixed ColPali mode requires float16")
            if self.processor_speed != "slow":
                raise ValueError(
                    "Fixed mode requires processor_speed='slow'; "
                    "the tokenizer is explicitly fast"
                )

            tokenizer = AutoTokenizer.from_pretrained(
                str(adapter), local_files_only=True, use_fast=True
            )
            if not tokenizer.is_fast:
                raise ValueError("Expected fast tokenizer")
            image_processor = SiglipImageProcessor.from_pretrained(
                str(adapter), local_files_only=True
            )
            processor = ColPaliProcessor(
                image_processor=image_processor, tokenizer=tokenizer
            )

            model = ColPali.from_pretrained(
                str(base),
                local_files_only=True,
                torch_dtype=torch.float16,
                device_map=self.device,
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
            )
            model.load_adapter(
                str(adapter),
                adapter_name="default",
                is_trainable=False,
                adapter_kwargs={"local_files_only": True},
            )
            model.set_adapter("default")
            model.eval()
            model.requires_grad_(False)

            self._model = model
            self._processor = processor
            self._image_token_id = processor.image_token_id
            logger.info("Loaded fixed local ColPali base and adapter")
            return
'''
texts[embedder] = text.replace(marker, marker + branch, 1)

# Replace only the known failure handlers in the indexing function.
text = texts[runner]
assert "Incomplete upload:" not in text, "Failure patch already exists"

start_marker = (
    "            except Exception:\n"
    "                # Retry per-doc"
)
end_marker = "\n            for doc, emb, token_info, crop_meta"
assert text.count(start_marker) == 1
start = text.index(start_marker)
end = text.index(end_marker, start)
text = text[:start] + '''            except Exception as exc:
                _append_jsonl(
                    failure_log,
                    {
                        "dataset": dataset_name,
                        "collection": collection_name,
                        "stage": "encoding",
                        "source_doc_ids": [
                            str((d.payload or {}).get("source_doc_id"))
                            for d in batch
                        ],
                        "error": repr(exc),
                    },
                )
                raise RuntimeError(
                    "Encoding failed; indexing stopped without dropping pages."
                ) from exc
''' + text[end:]

start_marker = (
    "                except Exception as e_single:\n"
    "                    if str(union_doc_id) not in failed_ids:"
)
end_marker = "\n                num_tiles = int(tile_pooled.shape[0])"
assert text.count(start_marker) == 1
start = text.index(start_marker)
end = text.index(end_marker, start)
text = text[:start] + '''                except Exception as exc:
                    _append_jsonl(
                        failure_log,
                        {
                            "dataset": dataset_name,
                            "collection": collection_name,
                            "stage": "pooling",
                            "source_doc_id": source_doc_id,
                            "union_doc_id": union_doc_id,
                            "error": repr(exc),
                        },
                    )
                    raise RuntimeError(
                        "Pooling failed; indexing stopped without dropping pages."
                    ) from exc
''' + text[end:]

marker = "        return uploaded\n"
assert text.count(marker) == 1
text = text.replace(marker, '''        if uploaded != len(points):
            stop_event.set()
            _append_jsonl(
                failure_log,
                {
                    "dataset": dataset_name,
                    "collection": collection_name,
                    "stage": "upload",
                    "expected": len(points),
                    "uploaded": uploaded,
                    "error": "Incomplete upload",
                },
            )
            raise RuntimeError(
                f"Incomplete upload: expected {len(points)}, got {uploaded}. "
                "Indexing stopped without dropping pages."
            )
        return uploaded
''', 1)

# Do not resume while silently skipping entries from a failure log.
marker = "    previously_failed_ids = set(failed_ids)\n"
assert text.count(marker) == 1
text = text.replace(marker, marker + '''    if failed_ids:
        raise RuntimeError(
            "Existing failure log found. Resolve and document failures "
            "before resuming; do not silently skip pages."
        )
''', 1)

# Clean up queued upload work when an ordinary exception aborts indexing.
marker = (
    "    except KeyboardInterrupt:\n"
    "        stop_event.set()\n"
)
assert text.count(marker) == 1
text = text.replace(
    marker,
    "    except BaseException:\n        stop_event.set()\n",
    1,
)
texts[runner] = text

# Validate every proposed file before writing.
for path, content in texts.items():
    ast.parse(content, filename=str(path))

record = root / "records" / (
    "p1-entry-patch-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    + "-" + uuid.uuid4().hex[:6]
)
record.mkdir(parents=True, exist_ok=False)

for path, raw in originals.items():
    (record / (path.stem + ".before.py")).write_bytes(raw)

for path, content in texts.items():
    path.write_bytes(content.encode("utf-8"))

diff = subprocess.run(
    [
        "git", "-C", str(root.parent), "diff", "--",
        "toolkit-repro/visual-rag-toolkit/visual_rag/embedding/visual_embedder.py",
        "toolkit-repro/visual-rag-toolkit/benchmarks/vidore_beir_qdrant/run_qdrant_beir.py",
    ],
    check=True,
    capture_output=True,
).stdout
(record / "changes.patch").write_bytes(diff)

report = {
    "status": "APPLIED_NOT_YET_ACCEPTED",
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "files": [str(p.relative_to(root)) for p in texts],
    "model_mode_variable": "PRB_COLPALI_BASE",
    "model_configuration": {
        "dtype": "float16",
        "attention": "sdpa",
        "image_processor": "SiglipImageProcessor",
        "tokenizer": "fast",
        "local_files_only": True,
    },
}
(record / "report.json").write_bytes(
    json.dumps(report, indent=2).encode("utf-8")
)

print("Patch record:", record)
print("Source syntax checks: PASS")
print("Entry patches: APPLIED; runtime checks still required")