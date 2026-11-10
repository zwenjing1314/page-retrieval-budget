"""Read-only audit and partial closeout after the preregistered empty-cluster stop.

Never retries formal retrieval, repairs clusters, changes seeds, or averages the
surviving K-means seeds. Writes only a fresh audit directory under the run.
"""
import argparse
import ast
import csv
from contextlib import closing
from datetime import datetime, timezone
from importlib.metadata import version
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
from qdrant_client import QdrantClient, models
from paper_pooling_experiment import ROOT, CACHE, BASES, METHODS, inputs, load, save, tsv
from paper_pooling_methods import cosine_kmeans, PoolingFailure
from paper_pooling_analyze import read_rows, METRICS
from ac_semantic_compare import read_rankings
from ac_semantic_evaluate import metrics
from prb_cache import digest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('record', type=Path)
    ap.add_argument('--output-name', default='partial-audit')
    args = ap.parse_args()
    record = args.record.resolve()
    out = record / args.output_name
    out.mkdir(exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    pages, queries, qrels, mapping, official, validation, manifest = inputs(record)
    assert validation == load(record / 'input-validation.json')
    assert manifest == load(record / 'source-cache-manifest.json')
    assert mapping == load(record / 'page-id-mapping.json')
    assert official == load(record / 'query-metadata.json')
    assert qrels == load(record / 'qrels.json')
    for name in ['paper_pooling_methods.py', 'paper_pooling_experiment.py']:
        assert digest(ROOT / 'scripts' / name) == digest(record / name)
    plan = load(record / 'plan.json')
    pilot = load(record / 'pilot/report.json')
    assert pilot['status'] == 'PASS' and pilot['plan_sha256'] == digest(record / 'plan.json')
    assert pilot['method_code_sha256'] == digest(ROOT / 'scripts/paper_pooling_methods.py')
    for fn in ['manifest.json', 'pinned-manifest.json', 'author-source-manifest.json']:
        for row in load(record / 'sources' / fn):
            if 'sha256' in row:
                assert digest(record / 'sources' / row['file']) == row['sha256']
    # Parse every delivered adapter without creating pyc or executing main().
    for name in ['paper_pooling_methods.py', 'paper_pooling_experiment.py',
                 'paper_pooling_analyze.py', 'paper_pooling_partial_audit.py']:
        ast.parse((ROOT / 'scripts' / name).read_text(encoding='utf-8'))
        shutil.copy2(ROOT / 'scripts' / name, out / name)
    qids = sorted(qrels, key=int)
    paths = {name: ROOT / 'records' / folder for name, folder in BASES.items()}
    names = [METHODS['sequential']] + [METHODS['kmeans'] + f'; seed={s}' for s in (0, 1)]
    paths.update(dict(zip(names, [record / v for v in ['sequential', 'kmeans-seed-0', 'kmeans-seed-1']])))
    values, ranks, sources = {}, {}, {}
    point_ids = {str(row['point_id']) for row in mapping}
    for name, path in paths.items():
        rep = load(path / 'report.json')
        assert rep['status'] == 'PASS' and rep['completed_queries'] == 232
        for fname, sha in rep.get('artifact_sha256', {}).items():
            assert digest(path / fname) == sha
        values[name] = read_rows(path / 'per-query.tsv')
        ranks[name] = read_rankings(path / ('rankings.json' if (path / 'rankings.json').exists() else 'rankings.jsonl'))
        assert set(values[name]) == set(ranks[name]) == set(qids)
        for q in qids:
            ranking = ranks[name][q]
            assert len(ranking) == len(set(ranking)) == 100 and set(ranking) <= point_ids
            recomputed = metrics(ranking, qrels[q])
            assert all(abs(recomputed[m] - values[name][q][m]) < 1e-12 for m in METRICS)
        assert all(abs(math.fsum(values[name][q][m] for q in qids) / 232 - rep['aggregate_metrics'][m]) < 1e-12 for m in METRICS)
        sources[name] = dict(record=str(path), report_sha256=digest(path / 'report.json'),
                             per_query_sha256=digest(path / 'per-query.tsv'))
    # Audit saved memberships, all vector files and live compressed-only config.
    resources = {}
    with closing(QdrantClient(url='http://127.0.0.1:16333', trust_env=False, timeout=60)) as client:
        assert client.info().version == '1.14.1'
        for name in names:
            path = paths[name]
            rep = load(path / 'report.json')
            rows = load(path / 'cache-manifest.json')
            assert len(rows) == 452 and {r['page_id'] for r in rows} == set(pages)
            norms, sizes = [], []
            for row in rows:
                pid = row['page_id']
                assert row['source_cache_npz_sha256'] == digest(CACHE / 'pages' / pid / 'vectors.npz')
                assert row['pooled_cache_sha256'] == digest(path / 'cache' / pid / 'vectors.npz')
                with np.load(path / 'cache' / pid / 'vectors.npz', allow_pickle=False) as z:
                    p, lab, raw, counts = (z[k] for k in ['pooled', 'labels', 'raw_means', 'group_sizes'])
                    assert p.shape == raw.shape == (34, 128) and p.dtype == np.float16
                    assert np.isfinite(p).all() and np.isfinite(raw).all()
                    assert lab.shape == (1024,) and np.array_equal(np.unique(lab), np.arange(34))
                    assert np.array_equal(np.bincount(lab, minlength=34), counts) and counts.sum() == 1024
                    assert counts.tolist() == row['group_sizes']
                    direct = np.stack([pages[pid][lab == i].astype(np.float32).mean(axis=0) for i in range(34)])
                    assert np.max(np.abs(direct - raw)) <= 1e-7
                    assert np.array_equal((raw / np.linalg.norm(raw, axis=1, keepdims=True)).astype(np.float16), p)
                    if name == METHODS['sequential']:
                        assert np.array_equal(lab, np.arange(1024) // 31)
                        assert counts.tolist() == [31] * 33 + [1]
                    norms.extend(np.linalg.norm(p.astype(np.float32), axis=1).tolist())
                    sizes.extend(counts.tolist())
            info = client.get_collection(rep['collection'])
            assert client.count(rep['collection'], exact=True).count == 452
            assert set(info.config.params.vectors) == {rep['field']}
            v = info.config.params.vectors[rep['field']]
            assert v.size == 128 and v.distance == models.Distance.COSINE and v.datatype == models.Datatype.FLOAT16
            assert v.multivector_config.comparator == models.MultiVectorComparator.MAX_SIM and v.on_disk is False
            assert info.config.hnsw_config.m == 0 and info.config.hnsw_config.full_scan_threshold == 10
            assert info.config.optimizer_config.indexing_threshold == 0 and info.indexed_vectors_count == 0
            assert info.status.value == 'green' and info.config.quantization_config is None and v.quantization_config is None
            save(out / (path.name + '-live-collection.json'), info.model_dump(mode='json'))
            resources[name] = {k:rep[k] for k in ['actual_vectors', 'actual_vectors_per_page', 'bare_vector_bytes',
                'pooling_seconds', 'cache_build_wall_seconds', 'collection_create_upload_seconds', 'readback_seconds',
                'evaluation_wall_seconds', 'total_variant_seconds', 'raw_mean_norm_min', 'raw_mean_norm_max',
                'max_unit_norm_error', 'iteration_min', 'iteration_max']}
            resources[name].update(stored_norm_min=min(norms), stored_norm_max=max(norms),
                smallest_group=min(sizes), largest_group=max(sizes), nonfinite=0, empty_clusters=0,
                maximum_diagnostic_score_error_per_token=max(load(path / f'diagnostic-{q}.json')['max_score_error_per_token'] for q in plan['diagnostic_queries']))
        failed = load(record / 'kmeans-seed-2/report.json')
        assert failed['status'] == 'FAIL' and failed['completed_pages'] == 258 and failed['completed_queries'] == 0
        assert not client.collection_exists(failed['collection'])
    # Read-only diagnosis repeats only the failing page, never resumes the run.
    done = {p.name for p in (record / 'kmeans-seed-2/cache').iterdir() if p.is_dir()}
    expected_done = set(sorted(pages, key=int)[:258])
    assert done == expected_done
    failed_pid = sorted(pages, key=int)[258]
    failure_checks = []
    for _ in range(2):
        t = time.perf_counter()
        try:
            cosine_kmeans(pages[failed_pid], 2, max_iterations=plan['kmeans']['max_iterations_guard'])
            raise AssertionError('Expected deterministic empty cluster')
        except PoolingFailure as error:
            assert error.diagnostics == failed['pooling_failure']
            failure_checks.append(dict(seconds=time.perf_counter()-t, diagnostics=error.diagnostics))
    partial_meta = [load(record / 'kmeans-seed-2/cache' / pid / 'metadata.json') for pid in sorted(done, key=int)]
    save(out / 'empty-cluster-diagnosis.json', dict(status='REPRODUCED_AS_DIAGNOSTIC_ONLY', corpus_id=failed_pid,
        page_identity=next(row for row in mapping if str(row['corpus_id']) == failed_pid),
        source_vectors_sha256=digest(CACHE / 'pages' / failed_pid / 'vectors.npz'),
        completed_pages=258, completed_queries=0, collection_exists=False, repeated_checks=failure_checks,
        partial_success_pooling_seconds=sum(r['pooling_seconds'] for r in partial_meta),
        partial_success_bare_vector_bytes=sum(r['bare_vector_bytes'] for r in partial_meta),
        note='Pinned third-party reference reseeds empty clusters; this preregistered adapter deliberately stops instead. No reseeding authorized or applied.'))
    aligned = load(ROOT / 'records/hierarchical-two-stage-20261110-013230-b63a62/analysis-final/intent-mapping.json')
    groups = [list(row['query_ids'].values()) for row in aligned]
    assert len(groups) == 58 and sorted((q for g in groups for q in g), key=int) == qids
    assert all(len(g) == 4 and all(qrels[q] == qrels[g[0]] for q in g) for g in groups)
    save(out / 'intent-mapping.json', aligned)
    sampled = np.random.default_rng(plan['bootstrap']['seed']).integers(0, 58, size=(20000, 58))
    comparisons, deltas = {}, []
    for name in names:
        comparisons[name] = {}
        for base in list(BASES) + ([names[0]] if name != names[0] else []):
            comparisons[name][base] = {}
            for m in METRICS:
                ds = np.array([values[name][q][m]-values[base][q][m] for q in qids])
                ds[np.abs(ds) <= 1e-12] = 0
                clusters = np.array([np.mean([values[name][q][m]-values[base][q][m] for q in g]) for g in groups])
                comparisons[name][base][m] = dict(mean_delta=float(ds.mean()), higher=int((ds>0).sum()),
                    equal=int((ds==0).sum()), lower=int((ds<0).sum()),
                    intent_cluster_bootstrap_95pct_CI=np.percentile(clusters[sampled].mean(axis=1), [2.5,97.5]).tolist())
            deltas.extend(dict(method=name, comparison=base, query_id=q,
                **{m:values[name][q][m]-values[base][q][m] for m in METRICS}) for q in qids)
    tsv(out / 'paired-per-query.tsv', deltas)
    tsv(out / 'all-per-query.tsv', [dict(method=n, query_id=q, **row) for n, rows in values.items() for q, row in rows.items()])
    aggregate = {name:{m:math.fsum(rows[q][m] for q in qids)/232 for m in METRICS} for name, rows in values.items()}
    full = next(iter(BASES))
    compressed = list(BASES)[1:] + names
    cases = []
    query_metadata = {str(row['query-id']):row for row in official}
    by_point = {str(row['point_id']):str(row['corpus_id']) for row in mapping}
    for row, group in zip(aligned, groups):
        means = {name:{m:float(np.mean([values[name][q][m] for q in group])) for m in METRICS} for name in values}
        common = all(values[n][q]['NDCG@10'] < values[full][q]['NDCG@10']-1e-12 for n in compressed for q in group)
        cases.append(dict(intent_index=row['intent_index'], query_ids=group,
            english_query=query_metadata[row['query_ids']['english']]['query'], metrics=means,
            all_completed_compressed_all_translations_ndcg10_loss=common,
            minimum_intent_mean_loss=min(means[full]['NDCG@10']-means[n]['NDCG@10'] for n in compressed)))
    save(out / 'all-intent-cases.json', cases)
    selected = sorted([c for c in cases if c['all_completed_compressed_all_translations_ndcg10_loss']],
                      key=lambda c:c['minimum_intent_mean_loss'], reverse=True)[:3]
    for case in selected:
        q = next(row['query_ids']['english'] for row in aligned if row['intent_index']==case['intent_index'])
        case['english_positive_ranks'] = {name:{by_point[p]: ranks[name][q].index(p)+1 if p in ranks[name][q] else None
            for p in qrels[q]} for name in ranks}
        case['english_top10_corpus_ids'] = {name:[by_point[p] for p in ranks[name][q][:10]] for name in ranks}
    save(out / 'provisional-loss-shortlist.json', selected)
    for command, filename in [(['git','status','--short'], 'git-status.txt'),
                               (['git','diff','--no-ext-diff'], 'tracked-diff.patch'),
                               ([sys.executable,'-m','pip','freeze'], 'environment-freeze.txt')]:
        (out / filename).write_text(subprocess.check_output(command,cwd=ROOT,text=True,encoding='utf-8'),encoding='utf-8')
    report = dict(status='PARTIAL_AUDIT_PASS_EXPERIMENT_BLOCKED', timestamp=datetime.now(timezone.utc).isoformat(),
        reason='Preregistered seed 2 encounters empty cluster on corpus page '+failed_pid,
        completed_variants=['sequential','kmeans-seed-0','kmeans-seed-1'], failed_variant='kmeans-seed-2',
        no_kmeans_seed_average=True, no_seed_replacement=True, no_formal_rerun=True,
        aggregate_metrics=aggregate, paired_differences=comparisons, resources=resources, sources=sources,
        input_validation=validation, plan_sha256=digest(record/'plan.json'),
        original_failure_sha256=digest(record/'formal-failure.json'),
        bootstrap=dict(unit='58 frozen intent clusters; translations sampled together',replicates=20000,seed=20261110,
            limitation='Pointwise descriptive intervals; complete-seed conditional results only; no K-means main estimate'),
        provisional_common_loss_intents=[c['intent_index'] for c in cases if c['all_completed_compressed_all_translations_ndcg10_loss']],
        common_loss_limitation='Only completed methods and seeds 0/1; not stability across all preregistered seeds or across datasets',
        python=sys.version, python_executable=sys.executable,
        versions={p:version(p) for p in ['numpy','scipy','torch','qdrant-client','pyarrow','colpali-engine','transformers','peft']},
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        commands=[[sys.executable,'-X','utf8','-B',str(ROOT/'scripts/paper_pooling_experiment.py'),'--record',str(record),'--stage',s] for s in ['pilot','formal']]+
                 [[sys.executable,'-X','utf8','-B',str(Path(__file__)),str(record),'--output-name',args.output_name]],
        audit_wall_seconds=time.perf_counter()-started,
        code_sha256={str(p.relative_to(ROOT)):digest(p) for p in [(ROOT/n if n.startswith('visual-rag-toolkit/') else ROOT/'scripts'/n) for n in
            ['paper_pooling_methods.py','paper_pooling_experiment.py','paper_pooling_analyze.py','paper_pooling_partial_audit.py',
             'prb_cache.py','ac_semantic_evaluate.py','ac_semantic_compare.py','visual-rag-toolkit/visual_rag/retrieval/single_stage.py',
             'visual-rag-toolkit/benchmarks/vidore_tatdqa_test/metrics.py']]})
    save(out / 'report.json', report)
    save(out / 'artifact-sha256.json', {p.name:digest(p) for p in out.iterdir() if p.is_file()})
    print(report['status'], flush=True)
    print('Failing corpus page:', failed_pid, flush=True)
    print('Provisional common loss intents:', report['provisional_common_loss_intents'], flush=True)
    for name in names:
        print(name, aggregate[name], flush=True)


if __name__ == '__main__':
    main()
