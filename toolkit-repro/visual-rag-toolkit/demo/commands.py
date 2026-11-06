"""Command builders and code generators."""

from typing import Any, Dict


def build_index_command(config: Dict[str, Any]) -> str:
    cmd_parts = ["python -m benchmarks.vidore_beir_qdrant.run_qdrant_beir"]
    cmd_parts.append(f"--datasets {' '.join(config['datasets'])}")
    cmd_parts.append(f"--collection {config['collection']}")
    cmd_parts.append(f"--model {config['model']}")
    cmd_parts.append("--index")
    if config.get("recreate"):
        cmd_parts.append("--recreate")
    if config.get("resume"):
        cmd_parts.append("--resume")
    if config.get("prefer_grpc"):
        cmd_parts.append("--prefer-grpc")
    else:
        cmd_parts.append("--no-prefer-grpc")
    cmd_parts.append(f"--torch-dtype {config.get('torch_dtype', 'float16')}")
    cmd_parts.append(f"--qdrant-vector-dtype {config.get('qdrant_vector_dtype', 'float16')}")
    cmd_parts.append(f"--batch-size {config.get('batch_size', 4)}")
    cmd_parts.append(f"--upload-batch-size {config.get('upload_batch_size', 8)}")
    cmd_parts.append(f"--qdrant-timeout {config.get('qdrant_timeout', 180)}")
    cmd_parts.append(f"--qdrant-retries {config.get('qdrant_retries', 5)}")
    if config.get("crop_empty"):
        cmd_parts.append("--crop-empty")
        cmd_parts.append(f"--crop-empty-percentage-to-remove {config.get('crop_percentage', 0.99)}")
    if config.get("no_cloudinary"):
        cmd_parts.append("--no-cloudinary")
    max_docs = config.get("max_docs")
    if max_docs and max_docs > 0:
        cmd_parts.append(f"--max-corpus-docs {max_docs}")
    cmd_parts.append("--no-eval")
    return " \\\n  ".join(cmd_parts)


def generate_python_index_code(config: Dict[str, Any]) -> str:
    datasets_str = ", ".join([f'"{ds}"' for ds in config.get("datasets", [])])
    model = config.get("model", "vidore/colpali-v1.3")
    collection = config.get("collection", "")
    batch_size = config.get("batch_size", 4)
    prefer_grpc = config.get("prefer_grpc", True)
    crop_empty = config.get("crop_empty", False)
    max_docs = config.get("max_docs")

    torch_dtype = config.get("torch_dtype", "float16")
    qdrant_dtype = config.get("qdrant_vector_dtype", "float16")

    torch_dtype_map = {
        "float16": "torch.float16",
        "float32": "torch.float32",
        "bfloat16": "torch.bfloat16",
    }
    torch_dtype_val = torch_dtype_map.get(torch_dtype, "torch.float16")

    code_lines = [
        "import os",
        "import torch",
        "from visual_rag import VisualEmbedder",
        "from visual_rag.indexing import QdrantIndexer",
        "from benchmarks.vidore_tatdqa_test.dataset_loader import load_vidore_beir_dataset",
        "",
        "# Configuration",
        f'COLLECTION = "{collection}"',
        f'MODEL = "{model}"',
        f"BATCH_SIZE = {batch_size}",
        f"DATASETS = [{datasets_str}]",
        f"TORCH_DTYPE = {torch_dtype_val}",
        f'QDRANT_DTYPE = "{qdrant_dtype}"',
    ]

    if max_docs:
        code_lines.append(f"MAX_DOCS = {max_docs}  # Limit docs per dataset")

    code_lines.extend(
        [
            "",
            "# Initialize embedder",
            "embedder = VisualEmbedder(",
            "    model_name=MODEL,",
            "    torch_dtype=TORCH_DTYPE,",
            ")",
            "",
            "# Initialize indexer",
            "indexer = QdrantIndexer(",
            '    url=os.getenv("QDRANT_URL"),',
            '    api_key=os.getenv("QDRANT_API_KEY"),',
            "    collection_name=COLLECTION,",
            f"    prefer_grpc={prefer_grpc},",
            "    vector_datatype=QDRANT_DTYPE,",
            ")",
            "",
            "# Create collection",
            f"indexer.create_collection(force_recreate={config.get('recreate', False)})",
            "indexer.create_payload_indexes(fields=[",
            '    {"field": "dataset", "type": "keyword"},',
            '    {"field": "doc_id", "type": "keyword"},',
            '    {"field": "source_doc_id", "type": "keyword"},',
            "])",
            "",
            "# Index each dataset",
            "for ds_name in DATASETS:",
            "    print(f'Loading {ds_name}...')",
            "    corpus, queries, qrels = load_vidore_beir_dataset(ds_name)",
        ]
    )

    if max_docs:
        code_lines.append("    corpus = corpus[:MAX_DOCS]  # Limit")

    code_lines.extend(
        [
            "    print(f'Indexing {len(corpus)} documents...')",
            "",
            "    for i in range(0, len(corpus), BATCH_SIZE):",
            "        batch = corpus[i:i + BATCH_SIZE]",
            "        images = [doc.image for doc in batch]",
            "",
            "        # Embed images",
            "        embeddings, token_infos = embedder.embed_images(",
            "            images, return_token_info=True",
            "        )",
            "",
            "        # Build points with multi-vector representations",
            "        points = []",
            "        for doc, emb, info in zip(batch, embeddings, token_infos):",
            "            emb_np = emb.cpu().numpy()",
            "            visual_idx = info.get('visual_token_indices', range(len(emb_np)))",
            "            visual_emb = emb_np[visual_idx]",
            "",
            "            tile_pooled = embedder.mean_pool_visual_embedding(visual_emb, info)",
            "            experimental = embedder.experimental_pool_visual_embedding(",
            "                visual_emb, info, mean_pool=tile_pooled",
            "            )",
            "            global_pooled = embedder.global_pool_from_mean_pool(tile_pooled)",
            "",
            "            points.append({",
            '                "id": f"{ds_name}_{doc.doc_id}",',
            '                "visual_embedding": visual_emb,',
            '                "tile_pooled_embedding": tile_pooled,',
            '                "experimental_pooled_embedding": experimental,',
            '                "global_pooled_embedding": global_pooled,',
            '                "metadata": {',
            '                    "dataset": ds_name,',
            '                    "doc_id": doc.doc_id,',
            '                    "source_doc_id": doc.payload.get("source_doc_id"),',
            "                },",
            "            })",
            "",
            "        indexer.upload_batch(points)",
            "        print(f'  Batch {i//BATCH_SIZE + 1}: {len(points)} uploaded')",
            "",
            '    print(f"Done: {ds_name}")',
        ]
    )

    if crop_empty:
        code_lines.insert(
            3, "from visual_rag.preprocessing.crop_empty import crop_empty, CropEmptyConfig"
        )
        code_lines.insert(
            len(code_lines) - 20, "        # Note: Add crop_empty() preprocessing before embedding"
        )

    return "\n".join(code_lines)


def build_eval_command(config: Dict[str, Any]) -> str:
    cmd_parts = ["python -m benchmarks.vidore_beir_qdrant.run_qdrant_beir"]
    cmd_parts.append(f"--datasets {' '.join(config['datasets'])}")
    cmd_parts.append(f"--collection {config['collection']}")
    cmd_parts.append(f"--model {config['model']}")
    cmd_parts.append(f"--mode {config['mode']}")
    if config["mode"] == "two_stage":
        cmd_parts.append(f"--stage1-mode {config.get('stage1_mode', 'tokens_vs_tiles')}")
        cmd_parts.append(f"--prefetch-k {config.get('prefetch_k', 256)}")
    elif config["mode"] == "three_stage":
        cmd_parts.append(f"--stage1-k {config.get('stage1_k', 1000)}")
        cmd_parts.append(f"--stage2-k {config.get('stage2_k', 300)}")
    cmd_parts.append(f"--top-k {config.get('top_k', 100)}")
    cmd_parts.append(f"--evaluation-scope {config.get('evaluation_scope', 'union')}")
    if config.get("prefer_grpc"):
        cmd_parts.append("--prefer-grpc")
    else:
        cmd_parts.append("--no-prefer-grpc")
    cmd_parts.append(f"--torch-dtype {config.get('torch_dtype', 'float16')}")
    cmd_parts.append(f"--qdrant-vector-dtype {config.get('qdrant_vector_dtype', 'float16')}")
    cmd_parts.append(f"--qdrant-timeout {config.get('qdrant_timeout', 180)}")
    if config.get("result_prefix"):
        cmd_parts.append(f"--output {config['result_prefix']}")
    return " \\\n  ".join(cmd_parts)


def generate_python_eval_code(config: Dict[str, Any]) -> str:
    datasets_str = ", ".join([f'"{ds}"' for ds in config.get("datasets", [])])
    mode = config.get("mode", "single_full")
    model = config.get("model", "vidore/colpali-v1.3")
    collection = config.get("collection", "")
    top_k = config.get("top_k", 100)
    scope = config.get("evaluation_scope", "union")
    prefer_grpc = config.get("prefer_grpc", True)
    torch_dtype = config.get("torch_dtype", "float16")

    torch_dtype_map = {
        "float16": "torch.float16",
        "float32": "torch.float32",
        "bfloat16": "torch.bfloat16",
    }
    torch_dtype_val = torch_dtype_map.get(torch_dtype, "torch.float16")

    code_lines = [
        "import os",
        "import torch",
        "from qdrant_client import QdrantClient",
        "from visual_rag import VisualEmbedder",
        "from visual_rag.retrieval import MultiVectorRetriever",
        "",
        "# Configuration",
        f'COLLECTION = "{collection}"',
        f'MODEL = "{model}"',
        f"TOP_K = {top_k}",
        f"DATASETS = [{datasets_str}]",
        f"TORCH_DTYPE = {torch_dtype_val}",
        "",
        "# Initialize clients",
        "client = QdrantClient(",
        '    url=os.getenv("QDRANT_URL"),',
        '    api_key=os.getenv("QDRANT_API_KEY"),',
        f"    prefer_grpc={prefer_grpc},",
        ")",
        "",
        "embedder = VisualEmbedder(",
        "    model_name=MODEL,",
        "    torch_dtype=TORCH_DTYPE,",
        ")",
        "",
        "# Initialize retriever",
        "retriever = MultiVectorRetriever(",
        "    client=client,",
        "    collection_name=COLLECTION,",
        "    embedder=embedder,",
        ")",
        "",
    ]

    if mode == "single_full":
        code_lines.extend(
            [
                "# Single-stage full retrieval",
                "def search(query: str):",
                "    query_embedding = embedder.embed_query(query)",
                "    return retriever.search_single_stage(",
                "        query_embedding=query_embedding,",
                f"        limit={top_k},",
                '        vector_name="initial",',
                "    )",
            ]
        )
    elif mode == "single_tiles":
        code_lines.extend(
            [
                "# Single-stage tiles retrieval",
                "def search(query: str):",
                "    query_embedding = embedder.embed_query(query)",
                "    return retriever.search_single_stage(",
                "        query_embedding=query_embedding,",
                f"        limit={top_k},",
                '        vector_name="mean_pooling",',
                "    )",
            ]
        )
    elif mode == "single_global":
        code_lines.extend(
            [
                "# Single-stage global retrieval",
                "def search(query: str):",
                "    query_embedding = embedder.embed_query(query)",
                "    return retriever.search_single_stage(",
                "        query_embedding=query_embedding,",
                f"        limit={top_k},",
                '        vector_name="global_pooling",',
                "    )",
            ]
        )
    elif mode == "two_stage":
        prefetch_k = config.get("prefetch_k", 256)
        stage1_mode = config.get("stage1_mode", "tokens_vs_standard_pooling")
        code_lines.extend(
            [
                "# Two-stage retrieval",
                "from visual_rag.retrieval import TwoStageRetriever",
                "",
                "two_stage = TwoStageRetriever(",
                "    client=client,",
                "    collection_name=COLLECTION,",
                "    embedder=embedder,",
                ")",
                "",
                "def search(query: str):",
                "    query_embedding = embedder.embed_query(query)",
                "    return two_stage.search(",
                "        query_embedding=query_embedding,",
                f"        prefetch_limit={prefetch_k},",
                f"        limit={top_k},",
                f'        stage1_mode="{stage1_mode}",',
                "    )",
            ]
        )
    elif mode == "three_stage":
        stage1_k = config.get("stage1_k", 1000)
        stage2_k = config.get("stage2_k", 300)
        code_lines.extend(
            [
                "# Three-stage retrieval",
                "from visual_rag.retrieval import ThreeStageRetriever",
                "",
                "three_stage = ThreeStageRetriever(",
                "    client=client,",
                "    collection_name=COLLECTION,",
                "    embedder=embedder,",
                ")",
                "",
                "def search(query: str):",
                "    query_embedding = embedder.embed_query(query)",
                "    return three_stage.search(",
                "        query_embedding=query_embedding,",
                f"        stage1_limit={stage1_k},",
                f"        stage2_limit={stage2_k},",
                f"        limit={top_k},",
                "    )",
            ]
        )

    if scope == "per_dataset":
        code_lines.extend(
            [
                "",
                "# Per-dataset filtering",
                "from qdrant_client.models import Filter, FieldCondition, MatchValue",
                "",
                'def search_dataset(query: str, dataset: str = "vidore/esg_reports_v2"):',
                "    query_embedding = embedder.embed_query(query)",
                "    dataset_filter = Filter(",
                "        must=[FieldCondition(",
                '            key="dataset",',
                "            match=MatchValue(value=dataset),",
                "        )]",
                "    )",
                "    # Add filter to your search call",
            ]
        )

    code_lines.extend(
        [
            "",
            "# Example usage",
            'results = search("What is the company revenue?")',
            "for r in results:",
            "    print(f\"Score: {r.score:.4f}, Doc: {r.payload.get('doc_id')}\")",
        ]
    )

    return "\n".join(code_lines)
