"""Fixed-budget, cache-only Economics single-stage compressed retrieval."""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import argparse
import ast
import csv
from functools import partial
from importlib.metadata import version
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
import pyarrow.parquet as pq
import torch
from qdrant_client import QdrantClient, models
from tqdm import tqdm
from prb_cache import CacheStore, digest
from ac_semantic_evaluate import metrics, local_reference
from visual_rag.retrieval.multi_vector import MultiVectorRetriever
from paper_pooling_methods import sequential, cosine_kmeans, PoolingFailure

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache/encoded-economics-v1"
BASES = {
    "single-stage full late-interaction MaxSim": "p2-formal-a-20261107-012250-612a76",
    "single-stage retrieval over experimental pooled vectors": "p2-formal-c-20261107-014210-68a2ce",
    "single-stage retrieval over HierarchicalTokenPooler pooled vectors": "ac-semantic-eval-20261109-172244-5e9662",
}
METHODS = {
    "sequential": "single-stage retrieval over paper-derived sequential pooling adapted to ColPali",
    "kmeans": "single-stage retrieval over paper-derived K-means pooling adapted to ColPali",
}

def load(p): return json.loads(p.read_text(encoding="utf-8"))

def save(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

def append(p, value):
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")

def tsv(p, rows):
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader(); w.writerows(rows)

def variants(plan):
    return [("sequential", None)] + [("kmeans", s) for s in plan["kmeans"]["seeds"]]

def key(method, seed): return method if seed is None else f"{method}-seed-{seed}"

def pool(method, seed, full, plan):
    if method == "sequential": return sequential(full)
    return cosine_kmeans(full, seed, max_iterations=plan["kmeans"]["max_iterations_guard"])

def inputs(record):
    profile = load(CACHE / "profile.json")
    store = CacheStore(CACHE, profile, create=False)
    a = ROOT / "records" / next(iter(BASES.values()))
    ar = load(a / "report.json")
    assert ar["status"] == "PASS" and store.profile_hash == ar["cache_profile_sha256"]
    for rel, sha in profile["encoding_code_sha256"].items():
        assert digest(ROOT / rel.replace("\\", "/")) == sha
    for package, ver in profile["versions"].items(): assert version(package) == ver
    manifest = store.manifest()
    assert manifest["complete"] and not manifest["pending_directories"]
    assert manifest == load(a / "cache-manifest.json")
    mapping = load(a / "page-id-mapping.json")
    by_corpus = {str(x["corpus_id"]):x for x in mapping}
    by_point = {str(x["point_id"]):x for x in mapping}
    assert len(by_corpus) == len(by_point) == 452
    assert set(by_corpus) == set(profile["expected_page_ids"])
    pages = {}
    for pid in sorted(by_corpus, key=int):
        arrays, meta = store.read("pages", pid)
        assert meta["identity"]["corpus_id"] == pid and meta["identity"]["doc_id"] == by_corpus[pid]["doc_id"]
        pages[pid] = arrays["vectors"]
    data = ROOT / ("data/economics_reports_v2-" + profile["dataset_revision"])
    assert digest(data / "download-manifest.json") == profile["dataset_download_manifest_sha256"]
    corpus = pq.read_table(data / "corpus/test-00000-of-00001.parquet", columns=["corpus-id","doc-id"]).to_pylist()
    assert len(corpus)==452 and {str(x["corpus-id"]):x["doc-id"] for x in corpus} == {p:x["doc_id"] for p,x in by_corpus.items()}
    official = pq.read_table(data / "queries/test-00000-of-00001.parquet", columns=["query-id","query","language"]).to_pylist()
    assert len(official)==232 and {str(x["query-id"]) for x in official}==set(profile["expected_query_ids"])
    queries = {}
    for row in official:
        qid = str(row["query-id"])
        arrays, meta = store.read("queries", qid)
        assert meta["identity"]["text"]==row["query"] and meta["identity"]["language"]==row["language"]
        queries[qid] = arrays["vectors"]
    assert "0" in queries
    qrels = load(a / "qrels.json")
    qr = pq.read_table(data / "qrels/test-00000-of-00001.parquet", columns=["query-id","corpus-id","score"]).to_pylist()
    mapped = {q:{} for q in queries}
    for x in qr:
        mapped[str(x["query-id"])][str(by_corpus[str(x["corpus-id"])]["point_id"])] = x["score"]
    assert len(qr)==sum(map(len, mapped.values()))==3628 and qrels==mapped
    source_hashes = {}
    for name, folder in BASES.items():
        path = ROOT / "records" / folder
        rep = load(path / "report.json")
        assert rep["status"]=="PASS" and rep["completed_queries"]==232
        for fname, sha in rep.get("artifact_sha256", {}).items(): assert digest(path / fname)==sha
        source_hashes[name] = {p.name:digest(p) for p in path.iterdir() if p.name in {"report.json","per-query.tsv","rankings.json","rankings.jsonl","qrels.json"}}
    hcache = ROOT / "records/ac-semantic-full-cache-20261109-124408-721631"
    hm = load(hcache / "manifest.json")
    assert len(hm)==452 and {x["id"] for x in hm}==set(pages)
    for item in hm:
        assert item["source_vectors_sha256"]==digest(CACHE / "pages" / item["id"] / "vectors.npz")
        assert item["pooled_sha256"]==digest(hcache / "pages" / item["id"] / "vectors.npz")
    validation = dict(status="PASS", pages=452, queries=232, qrels=3628, query_zero=True,
                      all_id_sets_align=True, patch_shape=[1024,128], profile_sha256=store.profile_hash,
                      old_method_source_hashes=source_hashes, hierarchical_cache_manifest_sha256=digest(hcache / "manifest.json"))
    return pages, queries, qrels, mapping, official, validation, manifest

def pilot(record, plan, pages):
    out = record / "pilot"
    out.mkdir(exist_ok=False)
    # Independent reference is the three inspected definitions only. Do not
    # import unrelated soft-DTW/CUDA modules or execute a remote module.
    module = ast.parse((record / "sources/kmeans-pytorch-pinned.py").read_text(encoding="utf-8"))
    defs = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in {"initialize","kmeans","pairwise_cosine"}]
    assert len(defs)==3
    scope = dict(torch=torch, np=np, partial=partial, tqdm=tqdm)
    exec(compile(ast.Module(body=defs, type_ignores=[]), "pinned-reference-definitions", "exec"), scope)
    rows = []
    # A small numeric sequence makes the fixed-width/tail coverage explicit.
    labels = np.arange(1024)//31
    assert np.bincount(labels).tolist()==[31]*33+[1]
    for method, seed in variants(plan):
        for pid in plan["pilot_pages"]:
            t=time.perf_counter()
            arrays, stats = pool(method, seed, pages[pid], plan)
            seconds=time.perf_counter()-t
            repeated, repeated_stats = pool(method, seed, pages[pid], plan)
            assert all(np.array_equal(arrays[k], repeated[k]) for k in arrays)
            assert np.array_equal(np.unique(arrays["labels"]), np.arange(34))
            reference = dict(repeated_output_identical=True)
            if method=="kmeans":
                lab, centers=scope["kmeans"](torch.from_numpy(pages[pid].copy()),34,distance="cosine",seed=seed,
                    tol=1e-4,
                    tqdm_flag=False,iter_limit=plan["kmeans"]["max_iterations_guard"])
                assert np.array_equal(lab.numpy(),arrays["labels"])
                assert np.max(np.abs(centers.numpy()-arrays["raw_means"]))<=1e-6
                reference.update(pinned_reference_labels_identical=True,
                    reference_raw_mean_max_error=float(np.max(np.abs(centers.numpy()-arrays["raw_means"]))))
            else:
                direct=np.stack([pages[pid][i:i+31].astype(np.float64).mean(axis=0) for i in range(0,1024,31)])
                assert np.max(np.abs(direct-arrays["raw_means"]))<=1e-7
                assert np.array_equal(arrays["pooled"][-1],pages[pid][-1]) or np.max(np.abs(arrays["pooled"][-1].astype(np.float32)-pages[pid][-1]))<1e-3
                reference.update(float64_direct_group_mean_max_error=float(np.max(np.abs(direct-arrays["raw_means"]))))
            name=key(method,seed)+"-page-"+pid
            np.savez(out/(name+".npz"),**arrays)
            row=dict(variant=key(method,seed),page_id=pid,pooling_seconds=seconds,shape=list(arrays["pooled"].shape),
                     dtype=str(arrays["pooled"].dtype),bare_vector_bytes=arrays["pooled"].nbytes,
                     group_sizes=arrays["group_sizes"].tolist(),**stats,**reference)
            save(out/(name+".json"),row)
            rows.append(row)
            print(f"Pilot {key(method,seed)} page {pid}: PASS, {seconds:.4f}s, iterations={stats['iterations']}",flush=True)
    result=dict(status="PASS",pages=plan["pilot_pages"],variants=[key(m,s) for m,s in variants(plan)],
                all_outputs_34x128=True,nonfinite=0,empty_clusters=0,
                pooling_seconds_by_variant={v:sum(x["pooling_seconds"] for x in rows if x["variant"]==v) for v in {x["variant"] for x in rows}},
                method_code_sha256=digest(ROOT/"scripts/paper_pooling_methods.py"),plan_sha256=digest(record/"plan.json"),
                rows=rows)
    save(out/"report.json",result)
    return result

def formal_variant(record, plan, method, seed, pages, queries, qrels, mapping):
    variant=key(method,seed)
    out=record/variant
    (out/"cache").mkdir(parents=True,exist_ok=False)
    field="paper_sequential_31" if seed is None else f"paper_kmeans_34_seed_{seed}"
    collection="prb_"+record.name.replace("paper-pooling-", "paper_").replace("-", "_")+"_"+variant.replace("-","_")
    report=dict(status="STARTED",method=METHODS[method],variant=variant,seed=seed,collection=collection,
                field=field,model_loads=0,encoding_calls=0,old_method_pooling_calls=0,exact=True,top_k=100,completed_pages=0,completed_queries=0)
    save(out/"report.json",report)
    client=None
    started=time.perf_counter()
    try:
        pooled, rows={},[]
        for n,pid in enumerate(sorted(pages,key=int),1):
            t=time.perf_counter(); arrays,stats=pool(method,seed,pages[pid],plan); seconds=time.perf_counter()-t
            pooled[pid]=arrays["pooled"]
            dest=out/"cache"/pid;dest.mkdir()
            np.savez(dest/"vectors.npz",**arrays)
            meta=dict(page_id=pid,source_cache_npz_sha256=digest(CACHE/"pages"/pid/"vectors.npz"),
                      pooled_cache_sha256=digest(dest/"vectors.npz"),shape=list(arrays["pooled"].shape),dtype=str(arrays["pooled"].dtype),
                      group_sizes=arrays["group_sizes"].tolist(),pooling_seconds=seconds,bare_vector_bytes=arrays["pooled"].nbytes,**stats)
            save(dest/"metadata.json",meta)
            rows.append(meta)
            report["completed_pages"]=n
            if n%100==0 or n==452:
                save(out/"report.json",report)
                print(f"Pooled {variant}: {n}/452",flush=True)
        save(out/"cache-manifest.json",rows)
        report.update(pooling_seconds=sum(r["pooling_seconds"] for r in rows),
                      cache_build_wall_seconds=time.perf_counter()-started,actual_vectors=452*34,actual_vectors_per_page=34,
                      bare_vector_bytes=sum(r["bare_vector_bytes"] for r in rows),nonfinite_values=0,empty_clusters=0,
                      max_unit_norm_error=max(r["max_stored_unit_norm_error"] for r in rows),
                      raw_mean_norm_min=min(r["raw_mean_norm_min"] for r in rows),raw_mean_norm_max=max(r["raw_mean_norm_max"] for r in rows),
                      iteration_min=min(r["iterations"] for r in rows),iteration_max=max(r["iterations"] for r in rows))
        client=QdrantClient(url="http://127.0.0.1:16333",trust_env=False,timeout=300)
        assert client.info().version=="1.14.1" and not client.collection_exists(collection)
        t=time.perf_counter()
        client.create_collection(collection_name=collection,vectors_config={field:models.VectorParams(
            size=128,distance=models.Distance.COSINE,datatype=models.Datatype.FLOAT16,on_disk=False,
            multivector_config=models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM))},
            hnsw_config=models.HnswConfigDiff(m=0,full_scan_threshold=10,on_disk=False),
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),on_disk_payload=False)
        by_point={str(x["point_id"]):x for x in mapping}
        for item in mapping:
            pid=str(item["corpus_id"])
            client.upsert(collection_name=collection,wait=True,points=[models.PointStruct(id=item["point_id"],
                vector={field:pooled[pid].tolist()},payload={"corpus_id":pid,"doc_id":item["doc_id"],"diagnostic":False})])
        report["collection_create_upload_seconds"]=time.perf_counter()-t
        info=client.get_collection(collection)
        assert client.count(collection,exact=True).count==452 and info.status.value=="green"
        assert set(info.config.params.vectors)=={field} and info.indexed_vectors_count==0
        v=info.config.params.vectors[field]
        assert v.size==128 and v.distance==models.Distance.COSINE and v.datatype==models.Datatype.FLOAT16
        assert v.multivector_config.comparator==models.MultiVectorComparator.MAX_SIM and v.on_disk is False
        assert info.config.hnsw_config.m==0 and info.config.hnsw_config.full_scan_threshold==10
        assert info.config.optimizer_config.indexing_threshold==0 and info.config.quantization_config is None and v.quantization_config is None
        save(out/"collection.json",info.model_dump(mode="json"))
        offset=None;seen=set();t=time.perf_counter();maxerr=0.0
        while True:
            batch,offset=client.scroll(collection_name=collection,offset=offset,limit=32,with_vectors=True,with_payload=True)
            for p in batch:
                pid=str(p.id);assert pid in by_point and pid not in seen;seen.add(pid)
                corpus_id=str(by_point[pid]["corpus_id"])
                assert set(p.vector)=={field} and str(p.payload["corpus_id"])==corpus_id and p.payload["doc_id"]==by_point[pid]["doc_id"]
                x=np.asarray(p.vector[field],dtype=np.float32);raw=pooled[corpus_id].astype(np.float32)
                assert x.shape==(34,128) and np.isfinite(x).all()
                error=float(np.max(np.abs(x-raw/np.linalg.norm(raw,axis=1,keepdims=True))))
                assert error<=1e-3;maxerr=max(maxerr,error)
            if offset is None:break
        assert seen==set(by_point)
        report.update(readback_seconds=time.perf_counter()-t,readback_max_error=maxerr,compressed_only_collection_verified=True)
        class Checked:
            query=None
            calls=0
            def query_points(self,*args,**kwargs):
                assert kwargs["collection_name"]==collection and kwargs["using"]==field and kwargs["limit"]==100
                assert kwargs["search_params"].exact and kwargs.get("prefetch") is None and kwargs.get("query_filter") is None
                assert kwargs["with_payload"] is True and kwargs["with_vectors"] is False
                assert np.array_equal(kwargs["query"],self.query.astype(np.float32))
                self.calls+=1
                return client.query_points(*args,**kwargs)
        checked=Checked()
        retriever=MultiVectorRetriever(collection_name=collection,qdrant_client=checked,embedder=object(),
                                     experimental_vector_name=field,request_timeout=300,max_retries=1)
        def search(q):
            checked.query=q
            return retriever.search_embedded(query_embedding=q,mode="single_experimental_tokens",top_k=100)
        # Before formal metrics, compare two fixed queries against FP32 reference.
        for qid in plan["diagnostic_queries"]:
            q=queries[qid];hits=search(q)
            refs=local_reference(q,{str(x["point_id"]):pooled[str(x["corpus_id"])] for x in mapping})
            errors=[abs(h["score_final"]-refs[str(h["id"])])/len(q) for h in hits]
            assert max(errors)<=plan["predeclared_score_tolerance_per_query_token"]
            ids={str(h["id"]) for h in hits}
            min_ref=min(refs[p] for p in ids)
            assert max(refs[p] for p in set(refs)-ids)<=min_ref+2e-3*len(q)
            scores=[refs[str(h["id"])] for h in hits]
            assert all(min(scores[:i])+2e-3*len(q)>=scores[i] for i in range(1,100))
            save(out/f"diagnostic-{qid}.json",dict(status="PASS",query_id=qid,max_score_error_per_token=max(errors),
                 all_452_fp32_scores=refs,returned_top100=[str(h["id"]) for h in hits]))
        print(f"{variant}: compressed-only collection readback and local MaxSim PASS",flush=True)
        result_rows=[];t_all=time.perf_counter()
        for n,qid in enumerate(sorted(queries,key=int),1):
            q=queries[qid];t=time.perf_counter();hits=search(q);seconds=time.perf_counter()-t
            ranking=[str(h["id"]) for h in hits]
            assert len(ranking)==len(set(ranking))==100 and set(ranking)<=set(by_point)
            assert all(math.isfinite(h["score_final"]) for h in hits)
            assert all(a["score_final"]>=b["score_final"] for a,b in zip(hits,hits[1:]))
            row=dict(query_id=qid,query_tokens=len(q),search_seconds=seconds,**metrics(ranking,qrels[qid]))
            result_rows.append(row)
            append(out/"rankings.jsonl",dict(query_id=qid,hits=[dict(point_id=str(h["id"]),score=h["score_final"]) for h in hits]))
            report["completed_queries"]=n
        tsv(out/"per-query.tsv",result_rows)
        report.update(status="PASS",formal_query_calls=232,diagnostic_query_calls=2,total_query_calls=checked.calls,
                      evaluation_wall_seconds=time.perf_counter()-t_all,
                      search_seconds=sum(x["search_seconds"] for x in result_rows),
                      aggregate_metrics={m:math.fsum(x[m] for x in result_rows)/232 for m in ["NDCG@5","NDCG@10","Recall@5","Recall@10","Recall@100"]},
                      total_variant_seconds=time.perf_counter()-started)
        report["artifact_sha256"]={p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!="report.json"}
        print(variant,report["aggregate_metrics"],flush=True)
        return report
    except Exception as error:
        report.update(status="FAIL",error=str(error),traceback=traceback.format_exc())
        if isinstance(error,PoolingFailure): report["pooling_failure"]=error.diagnostics
        raise
    finally:
        save(out/"report.json",report)
        if client:client.close()

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--record",type=Path,required=True);ap.add_argument("--stage",choices=["pilot","formal"],required=True)
    args=ap.parse_args();record=args.record.resolve();plan=load(record/"plan.json")
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    t=time.perf_counter()
    try:
        pages,queries,qrels,mapping,official,validation,manifest=inputs(record)
        if args.stage=="pilot":
            save(record/"input-validation.json",validation);save(record/"source-cache-manifest.json",manifest)
            save(record/"page-id-mapping.json",mapping);save(record/"query-metadata.json",official);save(record/"qrels.json",qrels)
            for name in ["paper_pooling_methods.py","paper_pooling_experiment.py"]:
                shutil.copy2(ROOT/"scripts"/name,record/name)
            pilot(record,plan,pages)
        else:
            accepted=load(record/"pilot/report.json")
            assert accepted["status"]=="PASS" and accepted["plan_sha256"]==digest(record/"plan.json")
            assert accepted["method_code_sha256"]==digest(ROOT/"scripts/paper_pooling_methods.py")
            assert validation==load(record/"input-validation.json")
            assert digest(ROOT/"scripts/paper_pooling_experiment.py")==digest(record/"paper_pooling_experiment.py")
            reports=[]
            for method,seed in variants(plan):reports.append(formal_variant(record,plan,method,seed,pages,queries,qrels,mapping))
            report=dict(status="PASS",scope="two methods, single-stage compressed-only",variants=reports,
                counts=dict(pages=452,queries=232,qrels=3628),model_loads=0,encoding_calls=0,old_method_pooling_calls=0,
                plan_sha256=digest(record/"plan.json"),input_validation_sha256=digest(record/"input-validation.json"),
                source_cache_profile_sha256=validation["profile_sha256"],
                versions={p:version(p) for p in ["numpy","scipy","torch","qdrant-client","pyarrow","colpali-engine","transformers","peft"]},
                python=sys.version,python_executable=sys.executable,
                git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                commands=[[sys.executable,"-X","utf8","-B",str(Path(__file__)),"--record",str(record),"--stage",stage] for stage in ["pilot","formal"]],
                code_sha256={str(p.relative_to(ROOT)):digest(p) for p in [ROOT/"scripts/paper_pooling_methods.py",Path(__file__),
                    ROOT/"scripts/prb_cache.py",ROOT/"scripts/ac_semantic_evaluate.py",
                    ROOT/"visual-rag-toolkit/visual_rag/retrieval/single_stage.py",
                    ROOT/"visual-rag-toolkit/benchmarks/vidore_tatdqa_test/metrics.py"]},
                formal_stage_wall_seconds=time.perf_counter()-t)
            save(record/"report.json",report)
    except Exception as error:
        failure=dict(status="FAIL",stage=args.stage,error=str(error),traceback=traceback.format_exc())
        if isinstance(error,PoolingFailure):failure["pooling_failure"]=error.diagnostics
        save(record/(args.stage+"-failure.json"),failure)
        raise

if __name__=="__main__":main()
