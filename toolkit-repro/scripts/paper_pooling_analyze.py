"""Paired fixed-budget analysis; translation clusters and seeds stay distinct."""
import argparse
import csv
import json
import math
from pathlib import Path
from itertools import combinations
import numpy as np
from paper_pooling_experiment import ROOT, BASES, METHODS, load, save, tsv
from prb_cache import digest
from ac_semantic_compare import read_rankings
from ac_semantic_evaluate import metrics

METRICS=("NDCG@5","NDCG@10","Recall@5","Recall@10","Recall@100")

def read_rows(path):
    with path.open(encoding="utf-8",newline="") as f:
        return {r["query_id"]:{m:float(r[m]) for m in METRICS} for r in csv.DictReader(f,delimiter="\t")}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("record",type=Path);args=ap.parse_args()
    record=args.record.resolve();run=load(record/"report.json");assert run["status"]=="PASS"
    out=record/"analysis";out.mkdir(exist_ok=False)
    qrels=load(record/"qrels.json");qids=sorted(qrels,key=int)
    queries={str(q["query-id"]):q for q in load(record/"query-metadata.json")}
    paths={name:ROOT/"records"/folder for name,folder in BASES.items()}
    seq=METHODS["sequential"];km=METHODS["kmeans"]
    paths[seq]=record/"sequential"
    for seed in load(record/"plan.json")["kmeans"]["seeds"]:
        paths[km+f"; seed={seed}"]=record/f"kmeans-seed-{seed}"
    values={name:read_rows(p/"per-query.tsv") for name,p in paths.items()}
    ranks={name:read_rankings(p/("rankings.json" if (p/"rankings.json").exists() else "rankings.jsonl")) for name,p in paths.items()}
    sources={}
    for name,path in paths.items():
        rep=load(path/"report.json");assert rep["status"]=="PASS" and rep["completed_queries"]==232
        for fname,sha in rep.get("artifact_sha256",{}).items():assert digest(path/fname)==sha
        assert set(values[name])==set(ranks[name])==set(qids)
        for q in qids:
            assert len(ranks[name][q])==len(set(ranks[name][q]))==100
            recomputed=metrics(ranks[name][q],qrels[q])
            assert all(abs(recomputed[m]-values[name][q][m])<1e-12 for m in METRICS)
        assert all(abs(math.fsum(values[name][q][m] for q in qids)/232-rep["aggregate_metrics"][m])<1e-12 for m in METRICS)
        sources[name]=dict(record=str(path),report_sha256=digest(path/"report.json"),per_query_sha256=digest(path/"per-query.tsv"))
    seednames=[n for n in paths if n.startswith(km)]
    values[km]={q:{m:math.fsum(values[n][q][m] for n in seednames)/len(seednames) for m in METRICS} for q in qids}
    aligned=load(ROOT/"records/hierarchical-two-stage-20261110-013230-b63a62/analysis-final/intent-mapping.json")
    groups=[list(row["query_ids"].values()) for row in aligned]
    assert len(groups)==58 and sorted((q for g in groups for q in g),key=int)==qids
    assert all(len(g)==4 and all(qrels[q]==qrels[g[0]] for q in g) for g in groups)
    save(out/"intent-mapping.json",aligned)
    sampled=np.random.default_rng(20261110).integers(0,58,size=(20000,58))
    def paired(new,base):
        result={}
        for m in METRICS:
            delta={q:new[q][m]-base[q][m] for q in qids}
            delta={q:0.0 if abs(d)<=1e-12 else d for q,d in delta.items()}
            cluster=np.array([np.mean([delta[q] for q in group]) for group in groups])
            result[m]=dict(mean_delta=math.fsum(delta.values())/232,
                higher=sum(d>0 for d in delta.values()),equal=sum(d==0 for d in delta.values()),lower=sum(d<0 for d in delta.values()),
                intent_cluster_bootstrap_95pct_CI=np.percentile(cluster[sampled].mean(axis=1),[2.5,97.5]).tolist())
        return result
    aggregate={name:{m:math.fsum(rows[q][m] for q in qids)/232 for m in METRICS} for name,rows in values.items()}
    comparisons={};delta_rows=[]
    for name in [seq]+seednames+[km]:
        comparisons[name]={}
        for base in list(BASES)+([seq] if name!=seq else []):
            comparisons[name][base]=paired(values[name],values[base])
            for q in qids:
                delta_rows.append(dict(method=name,comparison=base,query_id=q,
                    **{m:values[name][q][m]-values[base][q][m] for m in METRICS}))
    tsv(out/"paired-per-query.tsv",delta_rows)
    tsv(out/"all-per-query.tsv",[dict(method=n,query_id=q,**rows[q]) for n,rows in values.items() for q in qids])
    seed_summary={m:dict(mean=aggregate[km][m],minimum=min(aggregate[n][m] for n in seednames),
                         maximum=max(aggregate[n][m] for n in seednames),
                         seed_std_ddof1=float(np.std([aggregate[n][m] for n in seednames],ddof=1))) for m in METRICS}
    stability={}
    for n1,n2 in combinations(seednames,2):
        stability[n1+" versus "+n2]=dict(
            mean_top10_overlap=float(np.mean([len(set(ranks[n1][q][:10])&set(ranks[n2][q][:10])) for q in qids])),
            mean_top100_overlap=float(np.mean([len(set(ranks[n1][q])&set(ranks[n2][q])) for q in qids])))
    # All-intent audit, with a fixed descriptive definition. This is exploratory,
    # not a preregistered hypothesis test or an independent replication.
    full=next(iter(BASES));compressed=list(BASES)[1:]+[seq]+seednames
    case_rows=[]
    for i,group in enumerate(groups):
        per_method={n:{m:float(np.mean([values[n][q][m] for q in group])) for m in METRICS} for n in values}
        all_drop=all(values[n][q]["NDCG@10"]<values[full][q]["NDCG@10"]-1e-12 for n in compressed for q in group)
        km_stable_drop=all(values[n][q]["NDCG@10"]<values[full][q]["NDCG@10"]-1e-12 for n in seednames for q in group)
        rec_loss=all(per_method[n]["Recall@100"]<per_method[full]["Recall@100"]-1e-12 for n in compressed)
        row=dict(intent_index=i,query_ids=group,english_query=queries[group[0]]["query"],
                 metrics=per_method,all_compressed_all_translations_ndcg10_loss=all_drop,
                 all_kmeans_seeds_all_translations_ndcg10_loss=km_stable_drop,
                 all_compressed_intent_mean_recall100_loss=rec_loss,
                 minimum_ndcg10_loss_across_compressed=float(min(per_method[full]["NDCG@10"]-per_method[n]["NDCG@10"] for n in compressed)),
                 relevant_pages=len(qrels[group[0]]))
        case_rows.append(row)
    save(out/"all-intent-cases.json",case_rows)
    common=[x for x in case_rows if x["all_compressed_all_translations_ndcg10_loss"]]
    shortlisted=sorted(common,key=lambda x:x["minimum_ndcg10_loss_across_compressed"],reverse=True)[:3]
    mapping={str(x["point_id"]):x for x in load(record/"page-id-mapping.json")}
    cases=[]
    for case in shortlisted:
        q=case["query_ids"][0];rel=set(qrels[q])
        case=dict(case)
        case["english_positive_ranks"]={name:{mapping[p]["corpus_id"]:ranks[name][q].index(p)+1 if p in ranks[name][q] else None for p in rel} for name in ranks}
        case["english_top10_corpus_ids"]={name:[mapping[p]["corpus_id"] for p in ranks[name][q][:10]] for name in ranks}
        cases.append(case)
    save(out/"stable-loss-shortlist.json",cases)
    result=dict(status="PASS",aggregate_metrics=aggregate,paired_differences=comparisons,sources=sources,
        seed_summary=seed_summary,seed_ranking_stability=stability,
        kmeans_main_row="Arithmetic mean of all 3 preregistered seeds' query metrics; no single ensemble ranking",
        bootstrap=dict(unit="58 frozen intent clusters, all translations sampled together",replicates=20000,seed=20261110,
                       seed_handling="Seed metrics averaged within each query, then cluster-resampled; seeds are not extra samples",
                       limitation="Pointwise descriptive percentile intervals; not simultaneous multiple-comparison inference"),
        stable_loss_diagnostics=dict(exploratory=True,all_compressed_all_translations_ndcg10_loss_intents=[x["intent_index"] for x in common],
            all_kmeans_seeds_all_translations_ndcg10_loss_intents=[x["intent_index"] for x in case_rows if x["all_kmeans_seeds_all_translations_ndcg10_loss"]],
            all_compressed_intent_mean_recall100_loss_intents=[x["intent_index"] for x in case_rows if x["all_compressed_intent_mean_recall100_loss"]],
            selected_intents=[x["intent_index"] for x in cases]),
        analysis_script_sha256=digest(Path(__file__)),run_report_sha256=digest(record/"report.json"))
    save(out/"report.json",result)
    save(out/"artifact-sha256.json",{p.name:digest(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps(dict(aggregate_metrics=aggregate,seed_summary=seed_summary,stable_loss_diagnostics=result["stable_loss_diagnostics"]),ensure_ascii=False,indent=2))

if __name__=="__main__":main()
