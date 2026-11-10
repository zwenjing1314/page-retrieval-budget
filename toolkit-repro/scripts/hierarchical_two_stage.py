"""Fixed cache-only Economics experiment; creates a fresh two-field collection.

No encoding, pooling, downloads or mutation of prior collections/records.
The Toolkit's experimental_vector_name argument is only a field-name adapter.
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import csv
import json
import math
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone, timedelta
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from qdrant_client import QdrantClient, models
from prb_cache import CacheStore, digest
from ac_semantic_evaluate import metrics, local_reference
from visual_rag.retrieval.multi_vector import MultiVectorRetriever

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache/encoded-economics-v1"
H_CACHE = ROOT / "records/ac-semantic-full-cache-20261109-124408-721631"
BASES = {
    "single-stage full late-interaction MaxSim": "p2-formal-a-20261107-012250-612a76",
    "two-stage retrieval with experimental pooling prefetch and full MaxSim reranking": "p2-formal-b-20261107-013537-34e55b",
    "single-stage retrieval over experimental pooled vectors": "p2-formal-c-20261107-014210-68a2ce",
    "single-stage retrieval over HierarchicalTokenPooler pooled vectors": "ac-semantic-eval-20261109-172244-5e9662",
}
METHOD = "two-stage retrieval with HierarchicalTokenPooler prefetch and full MaxSim reranking"
FIELD = "semantic_hierarchical_30"
TOL = 1e-3  # Registered before execution; per-query-token FP32 reference tolerance.
FIXED = ("0", "20", "98", "174")

def load(p):
    return json.loads(p.read_text(encoding="utf-8"))

def save(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

def append(p, value):
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")

def tsv(p, rows):
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)

def main():
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S")
    record = ROOT / "records" / ("hierarchical-two-stage-" + stamp + "-" + uuid.uuid4().hex[:6])
    record.mkdir(exist_ok=False)
    collection = "prb_hierarchical_two_stage_" + uuid.uuid4().hex[:12]
    report = dict(status="STARTED", method=METHOD, collection=collection, prefetch_k=256,
                  top_k=100, exact=True, model_loads=0, encoding_calls=0, pooling_calls=0,
                  score_tolerance_per_query_token=TOL, fixed_diagnostic_queries=FIXED,
                  candidate_note="独立重放的候选诊断；不是服务端内部 Prefetch 候选日志",
                  command=[sys.executable, "-B", str(Path(__file__).resolve())],
                  timezone="Asia/Shanghai", started_at=datetime.now(timezone(timedelta(hours=8))).isoformat())
    print("Record:", record, flush=True)
    print("Collection:", collection, flush=True)
    save(record / "report.json", report)
    client = None
    started = time.perf_counter()
    try:
        sources = {name: ROOT / "records" / value for name, value in BASES.items()}
        a = next(iter(sources.values()))
        source_hashes = {}
        for name, path in sources.items():
            r = load(path / "report.json")
            assert r["status"] == "PASS" and r["completed_queries"] == 232
            for fname, sha in r.get("artifact_sha256", {}).items():
                assert digest(path / fname) == sha, (path, fname)
            source_hashes[name] = {p.name: digest(p) for p in path.iterdir()
                                   if p.name in {"report.json", "per-query.tsv", "rankings.json", "rankings.jsonl", "prefetch-candidates.jsonl", "qrels.json"}}
        profile = load(CACHE / "profile.json")
        store = CacheStore(CACHE, profile, create=False)
        ar = load(a / "report.json")
        assert store.profile_hash == ar["cache_profile_sha256"]
        for rel, sha in profile["encoding_code_sha256"].items():
            assert digest(ROOT / rel.replace("\\", "/")) == sha
        for package, ver in profile["versions"].items():
            assert version(package) == ver
        for rel, sha in load(list(sources.values())[1] / "report.json")["code_sha256"].items():
            assert digest(ROOT / rel.replace("\\", "/")) == sha
        manifest = store.manifest()
        assert manifest["complete"] and not manifest["pending_directories"]
        assert manifest == load(a / "cache-manifest.json")
        save(record / "cache-manifest.json", manifest)
        h_report = load(H_CACHE / "report.json")
        assert h_report["status"] == "PASS_FULL_CACHE" and h_report["completed_pages"] == 452
        assert h_report["cache_profile_sha256"] == store.profile_hash
        assert digest(Path(h_report["pooler_source"])) == h_report["pooler_source_sha256"]
        mapping = load(a / "page-id-mapping.json")
        by_corpus = {x["corpus_id"]: x for x in mapping}
        by_point = {str(x["point_id"]): x for x in mapping}
        hm = load(H_CACHE / "manifest.json")
        assert len(hm) == len(by_corpus) == len(by_point) == 452
        assert {x["id"] for x in hm} == set(by_corpus) == set(profile["expected_page_ids"])
        points, norm_errors = {}, []
        for item in hm:
            pid = item["id"]
            arrays, meta = store.read("pages", pid)
            assert item["corpus_id"] == pid and item["doc_id"] == meta["identity"]["doc_id"] == by_corpus[pid]["doc_id"]
            assert item["source_vectors_sha256"] == meta["npz_sha256"]
            hp = H_CACHE / "pages" / pid
            assert item == load(hp / "metadata.json")
            assert item["pooled_sha256"] == digest(hp / "vectors.npz")
            with np.load(hp / "vectors.npz", allow_pickle=False) as z:
                pooled = z["pooled"].copy()
                labels, sizes = z["token_cluster"], z["cluster_sizes"]
                assert labels.shape == (1024,) and np.array_equal(np.bincount(labels, minlength=34), sizes)
                assert sizes.tolist() == item["cluster_sizes"] and len(sizes) == 34 and min(sizes) > 0
            assert pooled.shape == (34, 128) and pooled.dtype == np.float16 and np.isfinite(pooled).all()
            err = float(np.max(np.abs(np.linalg.norm(pooled.astype(np.float32), axis=1) - 1)))
            assert err <= 0.005 and abs(err - item["max_unit_norm_error"]) < 1e-7
            norm_errors.append(err)
            point_id = str(by_corpus[pid]["point_id"])
            points[point_id] = {"vectors": {"initial": arrays["vectors"], FIELD: pooled},
                                "payload": {"corpus_id": pid, "doc_id": item["doc_id"], "diagnostic": False}}
        data = ROOT / ("data/economics_reports_v2-" + profile["dataset_revision"])
        assert digest(data / "download-manifest.json") == profile["dataset_download_manifest_sha256"]
        corpus = pq.read_table(data / "corpus/test-00000-of-00001.parquet", columns=["corpus-id", "doc-id"]).to_pylist()
        assert len(corpus) == 452 and {str(x["corpus-id"]): x["doc-id"] for x in corpus} == {p: x["doc_id"] for p, x in by_corpus.items()}
        query_rows = pq.read_table(data / "queries/test-00000-of-00001.parquet", columns=["query-id", "query", "language"]).to_pylist()
        official = {str(x["query-id"]): x for x in query_rows}
        assert len(official) == len(query_rows) == 232 and "0" in official
        assert set(official) == set(profile["expected_query_ids"])
        queries = {}
        for qid, row in official.items():
            arrays, meta = store.read("queries", qid)
            assert meta["identity"]["text"] == row["query"] and meta["identity"]["language"] == row["language"]
            queries[qid] = arrays["vectors"]
        qrels = load(a / "qrels.json")
        qr = pq.read_table(data / "qrels/test-00000-of-00001.parquet", columns=["query-id", "corpus-id", "score"]).to_pylist()
        mapped = {qid: {} for qid in queries}
        for x in qr:
            mapped[str(x["query-id"])][str(by_corpus[str(x["corpus-id"])]["point_id"])] = x["score"]
        assert len(qr) == sum(map(len, mapped.values())) == 3628 and mapped == qrels
        assert set(qrels) == set(queries) and all(g == 2 for rs in qrels.values() for g in rs.values())
        for path in list(sources.values())[:3]:
            assert load(path / "qrels.json") == qrels
        save(record / "qrels.json", qrels)
        save(record / "page-id-mapping.json", mapping)
        save(record / "query-metadata.json", query_rows)
        code_paths = [Path(__file__), ROOT / "scripts/prb_cache.py", ROOT / "scripts/ac_semantic_evaluate.py",
                      ROOT / "visual-rag-toolkit/visual_rag/retrieval/multi_vector.py",
                      ROOT / "visual-rag-toolkit/visual_rag/retrieval/two_stage.py",
                      ROOT / "visual-rag-toolkit/benchmarks/vidore_tatdqa_test/metrics.py"]
        report.update(counts={"pages":452,"queries":232,"qrels":3628}, source_records={k:str(v) for k,v in sources.items()},
                      source_artifact_sha256=source_hashes, cache_profile_sha256=store.profile_hash,
                      hierarchical_cache=str(H_CACHE), hierarchical_manifest_sha256=digest(H_CACHE / "manifest.json"),
                      hierarchical_report_sha256=digest(H_CACHE / "report.json"),
                      max_hierarchical_unit_norm_error=max(norm_errors),
                      versions={k:version(k) for k in ["numpy","qdrant-client","pyarrow","pydantic","torch","colpali-engine","transformers","peft"]},
                      python=sys.version, code_sha256={str(p.relative_to(ROOT)):digest(p) for p in code_paths},
                      git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                      representation_counts={"full":452*1024,"hierarchical":452*34},
                      raw_vector_bytes={"full":452*1024*128*2,"hierarchical":452*34*128*2,"total":452*1058*128*2})
        shutil.copy2(__file__, record / "hierarchical_two_stage.py")
        save(record / "input-validation.json", {"status":"PASS", "query_zero_present":True, "all_page_ids_aligned":True,
             "qrels_match_official_parquet":True, "all_cache_hashes_verified":True, "max_hierarchical_unit_norm_error":max(norm_errors)})
        print("Input identity, all 452 page pairs, 232 queries incl. 0, 3628 qrels: PASS", flush=True)
        client = QdrantClient(url="http://127.0.0.1:16333", trust_env=False, timeout=300)
        assert client.info().version == "1.14.1" and not client.collection_exists(collection)
        report["qdrant_server"] = client.info().model_dump(mode="json")
        t = time.perf_counter()
        client.create_collection(collection_name=collection,
            vectors_config={f:models.VectorParams(size=128,distance=models.Distance.COSINE,datatype=models.Datatype.FLOAT16,on_disk=False,
                multivector_config=models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM)) for f in ("initial",FIELD)},
            hnsw_config=models.HnswConfigDiff(m=0,full_scan_threshold=10,on_disk=False),
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),on_disk_payload=False)
        for n,(pid,p) in enumerate(points.items(),1):
            client.upsert(collection_name=collection,wait=True,points=[models.PointStruct(id=pid,vector={f:v.tolist() for f,v in p["vectors"].items()},payload=p["payload"])])
            if n % 100 == 0 or n == 452: print(f"Uploaded {n}/452",flush=True)
        report["build_upload_seconds"] = time.perf_counter()-t
        info = client.get_collection(collection)
        assert info.status.value == "green" and client.count(collection,exact=True).count == 452
        assert set(info.config.params.vectors) == {"initial",FIELD}
        for v in info.config.params.vectors.values():
            assert v.size==128 and v.distance==models.Distance.COSINE and v.datatype==models.Datatype.FLOAT16
            assert v.multivector_config.comparator==models.MultiVectorComparator.MAX_SIM and v.on_disk is False and v.quantization_config is None
        assert info.config.hnsw_config.m==0 and info.config.hnsw_config.full_scan_threshold==10
        assert info.config.optimizer_config.indexing_threshold==0 and info.indexed_vectors_count==0
        assert info.config.quantization_config is None and info.config.params.on_disk_payload is False
        save(record / "collection.json",info.model_dump(mode="json"))
        t=time.perf_counter()
        seen,offset,readback_error=set(),None,0.0
        while True:
            batch,offset=client.scroll(collection,offset=offset,limit=8,with_vectors=True,with_payload=True)
            for p in batch:
                pid=str(p.id)
                assert pid in points and pid not in seen and p.payload==points[pid]["payload"]
                seen.add(pid)
                assert set(p.vector)=={"initial",FIELD}
                for f in ("initial",FIELD):
                    v=np.asarray(p.vector[f],dtype=np.float32)
                    raw=points[pid]["vectors"][f].astype(np.float32)
                    assert v.shape==raw.shape and np.isfinite(v).all()
                    normalized=raw/np.linalg.norm(raw,axis=1,keepdims=True)
                    err=float(np.max(np.abs(v-normalized)))
                    assert err <= 1e-3, (pid,f,err)
                    readback_error=max(readback_error,err)
            if offset is None: break
        assert seen==set(points)
        report["readback_seconds"]=time.perf_counter()-t
        report["readback_max_element_error_vs_normalized_cache"]=readback_error
        print("All 452 points, both fields and vector values read back: PASS",flush=True)

        class CheckedClient:
            calls=0
            current=None
            def query_points(self,*args,**kw):
                assert kw["collection_name"]==collection and kw["using"]=="initial" and kw["limit"]==100
                assert kw["search_params"].exact and kw.get("query_filter") is None
                assert kw["with_payload"] is True and not kw.get("with_vectors",False)
                pre=kw["prefetch"]
                assert len(pre)==1 and pre[0].using==FIELD and pre[0].limit==256 and pre[0].params.exact and pre[0].filter is None
                assert np.array_equal(kw["query"],self.current.astype(np.float32)) and np.array_equal(pre[0].query,kw["query"])
                self.calls+=1
                return client.query_points(*args,**kw)
        checked=CheckedClient()
        retriever=MultiVectorRetriever(collection_name=collection,qdrant_client=checked,embedder=object(),
                experimental_vector_name=FIELD,request_timeout=300,max_retries=1)
        def search(q):
            checked.current=q
            return retriever.search_embedded(query_embedding=q,top_k=100,mode="two_stage",prefetch_k=256,stage1_mode="tokens_vs_experimental_pooling")
        def replay(q):
            return client.query_points(collection_name=collection,query=q.astype(np.float32).tolist(),using=FIELD,limit=452,
                    search_params=models.SearchParams(exact=True),with_payload=False,with_vectors=False).points
        def ids(hits): return [str(h["id"]) for h in hits]
        diagnostics=[]
        for qid in FIXED:
            q=queries[qid]
            hits=search(q)
            pooled=replay(q)
            candidates=[str(p.id) for p in pooled[:256]]
            assert pooled[255].score != pooled[256].score, "Fixed diagnostic has ambiguous prefetch boundary"
            explicit=client.query_points(collection_name=collection,query=q.astype(np.float32).tolist(),using="initial",limit=256,
                query_filter=models.Filter(must=[models.HasIdCondition(has_id=candidates)]),
                search_params=models.SearchParams(exact=True),with_payload=False,with_vectors=False).points
            assert {str(p.id) for p in explicit}==set(candidates) and set(ids(hits))<=set(candidates)
            exp_scores={str(p.id):p.score for p in explicit}
            # Exact score/tie-aware equivalence to explicit candidate-only sorting.
            assert [h["score_final"] for h in hits]==[p.score for p in explicit[:100]]
            assert all(h["score_final"]==exp_scores[str(h["id"])] for h in hits)
            refs=local_reference(q,{pid:p["vectors"]["initial"] for pid,p in points.items() if pid in candidates})
            prefrefs=local_reference(q,{pid:p["vectors"][FIELD] for pid,p in points.items()})
            ferr=max(abs(p.score-refs[str(p.id)])/len(q) for p in explicit)
            perr=max(abs(p.score-prefrefs[str(p.id)])/len(q) for p in pooled)
            assert ferr<=TOL and perr<=TOL
            ordered=[refs[pid] for pid in ids(hits)]
            assert all(min(ordered[:i])+2*TOL*len(q)>=ordered[i] for i in range(1,100))
            assert max((refs[pid] for pid in set(candidates)-set(ids(hits))),default=-math.inf)<=min(ordered)+2*TOL*len(q)
            d=dict(query_id=qid,status="PASS",full_reference_max_error_per_token=ferr,prefetch_reference_max_error_per_token=perr,
                   identical_order=ids(hits)==[str(p.id) for p in explicit[:100]],exact_score_sequence_equal=True,
                   candidates=candidates,server_top100=ids(hits),explicit_top256=[{"point_id":str(p.id),"score":p.score} for p in explicit],
                   fp32_candidate_scores=refs)
            diagnostics.append(d)
            save(record / f"diagnostic-{qid}.json",d)
            print(f"Fixed query {qid}: explicit candidate-only reranking and FP32 MaxSim PASS",flush=True)
        save(record / "correctness.json",{"status":"PASS","queries":FIXED,"checks":diagnostics})
        report["stage"]="full_232_evaluation"
        save(record / "report.json",report)
        rows=[]
        t_all=time.perf_counter()
        for n,qid in enumerate(sorted(queries,key=int),1):
            q=queries[qid]
            t=time.perf_counter(); hits=search(q); seconds=time.perf_counter()-t
            ranking=ids(hits)
            assert len(ranking)==len(set(ranking))==100 and set(ranking)<=set(points)
            assert all(h["payload"]==points[str(h["id"])]["payload"] for h in hits)
            assert all(math.isfinite(h["score_final"]) for h in hits)
            assert all(x["score_final"]>=y["score_final"] for x,y in zip(hits,hits[1:]))
            t=time.perf_counter(); pooled=replay(q); replay_seconds=time.perf_counter()-t
            pids=[str(p.id) for p in pooled]; scores=[p.score for p in pooled]
            assert len(pids)==len(set(pids))==452 and set(pids)==set(points)
            assert all(math.isfinite(s) for s in scores) and all(x>=y for x,y in zip(scores,scores[1:]))
            cutoff=scores[255]; strict={pid for pid,s in zip(pids,scores) if s>cutoff}; ties={pid for pid,s in zip(pids,scores) if s==cutoff}
            ambiguous=scores[255]==scores[256]
            assert set(ranking)<=strict|ties
            if not ambiguous: assert set(ranking)<=set(pids[:256])
            rel=set(qrels[qid]); kept=len(rel & set(pids[:256])); slots=256-len(strict)
            low=len(rel&strict)+max(0,slots-len(ties-rel)); high=len(rel&strict)+min(slots,len(ties&rel))
            vals=metrics(ranking,qrels[qid])
            assert vals["Recall@100"]<=high/len(rel)+1e-12
            row=dict(query_id=qid,query_tokens=len(q),search_seconds=seconds,candidate_replay_seconds=replay_seconds,
                     **vals,**{"Recall@256":kept/len(rel),"Hit@256":int(kept>0)},relevant_pages=len(rel),relevant_retained=kept,
                     candidate_recall_lower=low/len(rel),candidate_recall_upper=high/len(rel),boundary_ambiguous=ambiguous,
                     cutoff_score=cutoff,boundary_tie_count=len(ties),boundary_gap=cutoff-scores[256])
            rows.append(row)
            append(record / "rankings.jsonl",dict(query_id=qid,hits=[dict(point_id=str(h["id"]),score=h["score_final"]) for h in hits]))
            append(record / "prefetch-candidates.jsonl",dict(query_id=qid,provenance=report["candidate_note"],
                cutoff_score=cutoff,boundary_ambiguous=ambiguous,boundary_tied_point_ids=sorted(ties),
                candidates=[dict(rank=i+1,point_id=pid,score=s) for i,(pid,s) in enumerate(zip(pids[:256],scores[:256]))],
                all_452_scores=[dict(point_id=pid,score=s) for pid,s in zip(pids,scores)]))
            if n%40==0 or n==232: print(f"Evaluated {n}/232",flush=True)
        tsv(record / "per-query.tsv",rows)
        report.update(status="PASS",stage="finished",completed_queries=232,formal_search_calls=232,diagnostic_search_calls=len(FIXED),
                      all_toolkit_search_calls=checked.calls,evaluation_wall_seconds=time.perf_counter()-t_all,
                      formal_search_seconds=sum(r["search_seconds"] for r in rows),candidate_replay_seconds=sum(r["candidate_replay_seconds"] for r in rows),
                      aggregate_metrics={m:math.fsum(r[m] for r in rows)/232 for m in ["NDCG@5","NDCG@10","Recall@5","Recall@10","Recall@100","Recall@256","Hit@256"]},
                      boundary_ambiguous_queries=[r["query_id"] for r in rows if r["boundary_ambiguous"]],
                      candidate_recall_bounds=[sum(r[k] for r in rows)/232 for k in ["candidate_recall_lower","candidate_recall_upper"]],
                      total_work_seconds=time.perf_counter()-started)
        report["artifact_sha256"]={p.name:digest(p) for p in record.iterdir() if p.is_file() and p.name!="report.json"}
        print(report["aggregate_metrics"],flush=True)
    except Exception as e:
        report.update(status="FAIL",error=str(e),traceback=traceback.format_exc(),total_work_seconds=time.perf_counter()-started)
        raise
    finally:
        save(record / "report.json",report)
        if client: client.close()

if __name__=="__main__":
    main()
