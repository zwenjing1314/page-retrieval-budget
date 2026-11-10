"""Read-only paired analysis of one completed fixed experiment and frozen records."""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
from hierarchical_two_stage import ROOT, BASES, METHOD, load, save, tsv
from prb_cache import digest
from ac_semantic_compare import read_rankings
from ac_semantic_evaluate import metrics

METRICS=("NDCG@5","NDCG@10","Recall@5","Recall@10","Recall@100")

def read_rows(p):
    with p.open(encoding="utf-8",newline="") as f:
        return {r["query_id"]:r for r in csv.DictReader(f,delimiter="\t")}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("record",type=Path)
    ap.add_argument("--output-name",default="analysis")
    args=ap.parse_args()
    run=args.record.resolve()
    r=load(run/"report.json")
    assert r["status"]=="PASS" and r["completed_queries"]==232
    for name,sha in r["artifact_sha256"].items(): assert digest(run/name)==sha
    assert Path(args.output_name).name==args.output_name
    out=run/args.output_name
    out.mkdir(exist_ok=False)
    qrels=load(run/"qrels.json")
    queries=load(run/"query-metadata.json")
    sources={name:ROOT/"records"/folder for name,folder in BASES.items()}
    sources[METHOD]=run
    values={name:read_rows(p/"per-query.tsv") for name,p in sources.items()}
    ranks={name:read_rankings(p/("rankings.json" if (p/"rankings.json").exists() else "rankings.jsonl")) for name,p in sources.items()}
    for name in sources:
        assert set(values[name])==set(ranks[name])==set(qrels) and len(values[name])==232
        for qid in qrels:
            assert len(ranks[name][qid])==len(set(ranks[name][qid]))==100
            computed=metrics(ranks[name][qid],qrels[qid])
            for m in METRICS:
                assert abs(computed[m]-float(values[name][qid][m]))<1e-12,(name,qid,m)
        agg=load(sources[name]/"report.json")["aggregate_metrics"]
        for m in METRICS: assert abs(math.fsum(float(x[m]) for x in values[name].values())/232-agg[m])<1e-12
    languages=("english","french","spanish","german")
    grouped={lang:sorted([str(x["query-id"]) for x in queries if x["language"].lower()==lang],key=int) for lang in languages}
    assert all(len(v)==58 for v in grouped.values())
    aligned=list(zip(*(grouped[l] for l in languages)))
    for group in aligned: assert all(qrels[q]==qrels[group[0]] for q in group)
    save(out/"intent-mapping.json",[{"intent_index":i,"query_ids":dict(zip(languages,group))} for i,group in enumerate(aligned)])
    # Same mapping protocol as the accepted language-paired analysis; no row bootstrap.
    samples=np.random.default_rng(20261110).integers(0,58,size=(20000,58))
    def paired(d):
        # The frozen full baseline used math.fsum; Toolkit sums sequentially.
        # Remove only machine-rounding dust, using the pre-existing 1e-12 equality rule.
        d={q:(0.0 if abs(v)<=1e-12 else v) for q,v in d.items()}
        flat=list(d.values())
        clusters=np.array([np.mean([d[q] for q in group]) for group in aligned])
        ci=np.percentile(clusters[samples].mean(axis=1),[2.5,97.5]).tolist()
        return dict(mean_delta=math.fsum(flat)/232,higher=sum(v>1e-12 for v in flat),
                    equal=sum(abs(v)<=1e-12 for v in flat),lower=sum(v< -1e-12 for v in flat),
                    intent_cluster_bootstrap_95pct_CI=ci)
    result=dict(status="PASS",method=METHOD,all_frozen_rankings_recomputed=True,
        analysis_script_sha256=digest(Path(__file__)),source_report_sha256=digest(run/"report.json"),
        mapping_reference="records/ac-semantic-language-paired-20261109-235341-0d8470/report.json",
        design=dict(intent_clusters=58,translations_per_intent=4,replicates=20000,seed=20261110,
                    interval="paired percentile, pointwise descriptive intervals; no simultaneous coverage claim"),
        aggregate_metrics={name:{m:math.fsum(float(x[m]) for x in rows.values())/232 for m in METRICS} for name,rows in values.items()},
        paired_differences={},candidate_diagnostics={},ranking_diagnostics={})
    delta_rows=[]
    for name in BASES:
        result["paired_differences"][name]={m:paired({q:float(values[METHOD][q][m])-float(values[name][q][m]) for q in qrels}) for m in METRICS}
        result["ranking_diagnostics"][name]={
            "same_order_top100_queries":sum(ranks[METHOD][q]==ranks[name][q] for q in qrels),
            "same_set_top100_queries":sum(set(ranks[METHOD][q])==set(ranks[name][q]) for q in qrels),
            "mean_top10_overlap":np.mean([len(set(ranks[METHOD][q][:10])&set(ranks[name][q][:10])) for q in qrels]),
            "mean_top100_overlap":np.mean([len(set(ranks[METHOD][q])&set(ranks[name][q])) for q in qrels])}
        for q in sorted(qrels,key=int):
            delta_rows.append(dict(method=METHOD,comparison=name,query_id=q,**{m:float(values[METHOD][q][m])-float(values[name][q][m]) for m in METRICS}))
    tsv(out/"paired-per-query.tsv",delta_rows)
    candidate_rows=[]
    for name,path in [(list(BASES)[1],list(sources.values())[1]),(METHOD,run)]:
        entries=[json.loads(line) for line in (path/"prefetch-candidates.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(entries)==232 and {e["query_id"] for e in entries}==set(qrels)
        rows=[]
        for e in entries:
            q=e["query_id"]; rel=set(qrels[q]); cand=e["candidates"]
            ids={str(p["point_id"]) for p in cand}; assert len(ids)==256
            cutoff=e["cutoff_score"]
            strict={str(p["point_id"]) for p in cand if p["score"]>cutoff}
            ties=set(e["boundary_tied_point_ids"]); slots=256-len(strict)
            assert len(strict)<256 and len(strict|ties)>=256 and ids<=strict|ties
            lower=len(strict&rel)+max(0,slots-len(ties-rel)); upper=len(strict&rel)+min(slots,len(ties&rel))
            kept=len(ids&rel)
            row=dict(method=name,query_id=q,**{"Recall@256":kept/len(rel),"Hit@256":int(kept>0)},
                     recall_lower=lower/len(rel),recall_upper=upper/len(rel),relevant_pages=len(rel),relevant_retained=kept,
                     boundary_ambiguous=e["boundary_ambiguous"],missing_relevant_point_ids=sorted(rel-ids),
                     full_top100_retained=len(ids&set(ranks[list(BASES)[0]][q])),
                     full_top10_retained=len(ids&set(ranks[list(BASES)[0]][q][:10])))
            rows.append(row)
        result["candidate_diagnostics"][name]={
            "provenance":"独立重放的候选诊断；不是内部 Prefetch 候选日志",
            "Recall@256":math.fsum(x["Recall@256"] for x in rows)/232,
            "Hit@256":sum(x["Hit@256"] for x in rows)/232,
            "micro_recall256":sum(x["relevant_retained"] for x in rows)/3628,
            "relevant_query_page_pairs_lost":sum(x["relevant_pages"]-x["relevant_retained"] for x in rows),
            "queries_missing_any_relevant":sum(x["Recall@256"]<1 for x in rows),
            "boundary_ambiguous_query_ids":[x["query_id"] for x in rows if x["boundary_ambiguous"]],
            "macro_recall_tie_bounds":[math.fsum(x[k] for x in rows)/232 for k in ["recall_lower","recall_upper"]],
            "mean_full_top100_retained":np.mean([x["full_top100_retained"] for x in rows]),
            "mean_full_top10_retained":np.mean([x["full_top10_retained"] for x in rows])}
        candidate_rows.extend(rows)
    cand_new={x["query_id"]:x for x in candidate_rows if x["method"]==METHOD}
    cand_old={x["query_id"]:x for x in candidate_rows if x["method"]!=METHOD}
    result["candidate_paired_difference"]={m:paired({q:cand_new[q][m]-cand_old[q][m] for q in qrels}) for m in ["Recall@256","Hit@256"]}
    raw={name:{x["query_id"]:x["hits"] for x in [json.loads(line) for line in (path/"rankings.jsonl").read_text(encoding="utf-8").splitlines()]}
         for name,path in [(list(BASES)[0],next(iter(sources.values()))),(METHOD,run)]}
    audit=[]
    for q in sorted(qrels,key=int):
        old=raw[list(BASES)[0]][q]; new=raw[METHOD][q]
        old_scores={x["point_id"]:x["score"] for x in old}; new_scores={x["point_id"]:x["score"] for x in new}
        common=set(old_scores)&set(new_scores)
        error=max(abs(old_scores[p]-new_scores[p])/int(values[METHOD][q]["query_tokens"]) for p in common)
        assert error<=1e-3
        rel=set(qrels[q]); gone=set(old_scores)-set(new_scores); added=set(new_scores)-set(old_scores)
        row=dict(query_id=q,common_full_score_error_per_token=error,
                 removed_relevant=sorted(gone&rel),added_relevant=sorted(added&rel),
                 removed_without_positive_qrel=sorted(gone-rel),added_without_positive_qrel=sorted(added-rel),
                 ndcg10_delta=float(values[METHOD][q]["NDCG@10"])-float(values[list(BASES)[0]][q]["NDCG@10"]),
                 old_top10=ranks[list(BASES)[0]][q][:10],new_top10=ranks[METHOD][q][:10])
        audit.append(row)
    save(out/"full-baseline-audit.json",audit)
    result["full_baseline_audit"]={"max_common_score_error_per_token":max(x["common_full_score_error_per_token"] for x in audit),
        "ndcg10_changed_queries":[x["query_id"] for x in audit if abs(x["ndcg10_delta"])>1e-12],
        "removed_positive_query_page_pairs":sum(len(x["removed_relevant"]) for x in audit),
        "added_positive_query_page_pairs":sum(len(x["added_relevant"]) for x in audit),
        "removed_without_positive_qrel_pairs":sum(len(x["removed_without_positive_qrel"]) for x in audit),
        "all_common_scores_verified":True}
    tsv(out/"candidate-per-query.tsv",candidate_rows)
    save(out/"report.json",result)
    summary=["| 方法 | "+" | ".join(METRICS)+" |","|---|"+"---:|"*len(METRICS)]
    for name,v in result["aggregate_metrics"].items(): summary.append("| "+name+" | "+" | ".join(f"{v[m]:.6f}" for m in METRICS)+" |")
    (out/"metrics-table.md").write_text("\n".join(summary)+"\n",encoding="utf-8")
    save(out/"artifact-sha256.json",{p.name:digest(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps({k:result[k] for k in ["aggregate_metrics","paired_differences","candidate_diagnostics","candidate_paired_difference","ranking_diagnostics"]},ensure_ascii=False,indent=2))

if __name__=="__main__": main()
