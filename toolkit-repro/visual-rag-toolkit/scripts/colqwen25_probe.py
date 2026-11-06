"""
Probe script for vidore/colqwen2.5-v0.2 embedding layout.

Usage:
  python scripts/colqwen25_probe.py --model vidore/colqwen2.5-v0.2 --device cuda:0

Notes:
- ColQwen2.5 requires colpali-engine>=0.3.7 and transformers>=4.45.0
  (the model card recommends installing from source).
- This script prints embedding shapes + token_info (grid_h/grid_w when available),
  and runs mean/experimental pooling to validate compatibility with the pipeline.
"""

from __future__ import annotations

import argparse

from PIL import Image

from visual_rag import VisualEmbedder


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="vidore/colqwen2.5-v0.2")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = p.parse_args()

    img = Image.new("RGB", (1024, 768), color="white")

    embedder = VisualEmbedder(model_name=args.model, device=args.device, torch_dtype=args.dtype)
    embs, infos = embedder.embed_images([img], return_token_info=True, show_progress=False)

    emb = embs[0]
    info = infos[0]
    print("Model:", args.model)
    print("Full embedding:", tuple(emb.shape), "dtype:", emb.dtype)
    print(
        "token_info:",
        {
            k: info.get(k)
            for k in [
                "num_visual_tokens",
                "grid_t",
                "grid_h",
                "grid_w",
                "n_rows",
                "n_cols",
                "num_tiles",
            ]
        },
    )

    visual = embedder.extract_visual_embedding(emb, info)
    print("Visual embedding:", tuple(visual.shape), "dtype:", visual.dtype)

    mean_pool = embedder.mean_pool_visual_embedding(visual, info, target_vectors=32)
    exp_pool = embedder.experimental_pool_visual_embedding(
        visual, info, target_vectors=32, mean_pool=mean_pool
    )
    global_pool = embedder.global_pool_from_mean_pool(mean_pool)

    print("mean_pool:", tuple(mean_pool.shape))
    print("exp_pool:", tuple(exp_pool.shape))
    print("global_pool:", tuple(global_pool.shape))


if __name__ == "__main__":
    main()
