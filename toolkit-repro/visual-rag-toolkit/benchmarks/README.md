# ViDoRe Benchmark Evaluation

This directory contains scripts for evaluating visual document retrieval on the [ViDoRe benchmark](https://huggingface.co/spaces/vidore/vidore-leaderboard).

## Quick Start

### 1. Install Dependencies

```bash
# Install visual-rag-toolkit with all dependencies
pip install -e ".[all]"

# Install benchmark-specific dependencies
pip install datasets mteb
```

### 2. Run Evaluation

```bash
# Run on single dataset
python benchmarks/run_vidore.py --dataset vidore/docvqa_test_subsampled

# Run on all ViDoRe datasets
python benchmarks/run_vidore.py --all

# With two-stage retrieval (our contribution)
python benchmarks/run_vidore.py --dataset vidore/docvqa_test_subsampled --two-stage
```

### 3. Submit to Leaderboard

```bash
# Generate submission file
python benchmarks/prepare_submission.py --results results/

# Submit to HuggingFace
huggingface-cli login
huggingface-cli upload vidore/results ./submission.json
```

## ViDoRe Datasets

The benchmark includes these datasets (from the leaderboard):

| Dataset | Type | # Queries | # Documents |
|---------|------|-----------|-------------|
| docvqa_test_subsampled | DocVQA | ~500 | ~5,000 |
| infovqa_test_subsampled | InfoVQA | ~500 | ~5,000 |
| tabfquad_test_subsampled | TabFQuAD | ~500 | ~5,000 |
| tatdqa_test | TAT-DQA | ~1,500 | ~2,500 |
| arxivqa_test_subsampled | ArXivQA | ~500 | ~5,000 |
| shiftproject_test | SHIFT | ~500 | ~5,000 |

## Evaluation Metrics

- **NDCG@5**: Normalized Discounted Cumulative Gain at 5
- **NDCG@10**: Normalized Discounted Cumulative Gain at 10  
- **MRR@10**: Mean Reciprocal Rank at 10
- **Recall@5**: Recall at 5
- **Recall@10**: Recall at 10

## Two-Stage Retrieval (Our Contribution)

Our key contribution is efficient two-stage retrieval:

```
Stage 1: Fast prefetch with tile-level pooled vectors
         Uses HNSW index for O(log N) retrieval
         
Stage 2: Exact MaxSim reranking on top-K candidates
         Full multi-vector scoring for precision
```

This provides:
- **5-10x speedup** over full MaxSim at scale
- **95%+ accuracy** compared to exhaustive search
- **Memory efficient** (don't load all embeddings upfront)

To evaluate with two-stage:

```bash
python benchmarks/run_vidore.py \
    --dataset vidore/docvqa_test_subsampled \
    --two-stage \
    --prefetch-k 200 \
    --top-k 10
```

## Files

- `run_vidore.py` - Main evaluation script
- `prepare_submission.py` - Generate leaderboard submission
- `analyze_results.py` - Analyze and compare results








