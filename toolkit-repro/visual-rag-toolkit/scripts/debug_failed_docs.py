import argparse
import json
from pathlib import Path


def _ensure_pil(img):
    from PIL import Image

    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    try:
        return img.convert("RGB")
    except Exception:
        raise TypeError(f"Unsupported image type: {type(img)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, default="vidore/colSmol-500M")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--torch-dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"]
    )
    parser.add_argument(
        "--processor-speed", type=str, default="fast", choices=["fast", "slow", "auto"]
    )
    parser.add_argument("--source-doc-ids", type=str, nargs="+", required=True)
    parser.add_argument("--crop-empty", action="store_true", default=False)
    parser.add_argument("--crop-empty-percentage-to-remove", type=float, default=0.99)
    parser.add_argument("--crop-empty-remove-page-number", action="store_true", default=False)
    parser.add_argument("--crop-empty-preserve-border-px", type=int, default=1)
    parser.add_argument("--crop-empty-uniform-std-threshold", type=float, default=0.0)
    parser.add_argument("--out-dir", type=str, default="results/paper_eval/debug_failed_docs")
    args = parser.parse_args()

    import numpy as np
    import torch

    from benchmarks.vidore_tatdqa_test.dataset_loader import load_vidore_beir_dataset
    from visual_rag.embedding.visual_embedder import VisualEmbedder
    from visual_rag.preprocessing.crop_empty import CropEmptyConfig, crop_empty

    torch_dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[
        args.torch_dtype
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus, _, _ = load_vidore_beir_dataset(args.dataset)
    wanted = set(str(x) for x in args.source_doc_ids)
    found = []
    for d in corpus:
        sid = str((d.payload or {}).get("source_doc_id") or "")
        if sid in wanted:
            found.append(d)
    found_by_id = {str((d.payload or {}).get("source_doc_id") or ""): d for d in found}
    missing = [x for x in args.source_doc_ids if str(x) not in found_by_id]
    if missing:
        raise SystemExit(f"Could not find source_doc_id(s) in corpus: {missing}")

    embedder = VisualEmbedder(
        model_name=str(args.model),
        device=args.device,
        torch_dtype=torch_dtype,
        processor_speed=str(args.processor_speed),
        batch_size=1,
    )

    report = {}
    for sid in args.source_doc_ids:
        d = found_by_id[str(sid)]
        original_img = _ensure_pil(d.image)
        original_path = (
            out_dir / f"{args.dataset.replace('/', '__')}__source_doc_id={sid}__original.png"
        )
        original_img.save(original_path)

        crop_meta = {
            "applied": False,
            "crop_box": None,
            "original_width": int(original_img.width),
            "original_height": int(original_img.height),
            "cropped_width": int(original_img.width),
            "cropped_height": int(original_img.height),
        }
        embed_img = original_img
        if bool(args.crop_empty):
            embed_img, crop_meta = crop_empty(
                original_img,
                config=CropEmptyConfig(
                    percentage_to_remove=float(args.crop_empty_percentage_to_remove),
                    remove_page_number=bool(args.crop_empty_remove_page_number),
                    preserve_border_px=int(args.crop_empty_preserve_border_px),
                    uniform_rowcol_std_threshold=float(args.crop_empty_uniform_std_threshold),
                ),
            )

        cropped_path = (
            out_dir / f"{args.dataset.replace('/', '__')}__source_doc_id={sid}__cropped.png"
        )
        _ensure_pil(embed_img).save(cropped_path)

        embeddings, token_infos = embedder.embed_images(
            [embed_img],
            batch_size=1,
            return_token_info=True,
            show_progress=False,
        )
        emb = embeddings[0]
        token_info = token_infos[0] or {}

        emb_np = (
            emb.cpu().float().numpy() if hasattr(emb, "cpu") else np.array(emb, dtype=np.float32)
        )
        visual_indices = token_info.get("visual_token_indices") or list(range(int(emb_np.shape[0])))
        visual_embedding = emb_np[visual_indices].astype(np.float32)

        tile_pooled = embedder.mean_pool_visual_embedding(
            visual_embedding, token_info, target_vectors=32
        )
        experimental_pooled = embedder.experimental_pool_visual_embedding(
            visual_embedding,
            token_info,
            target_vectors=32,
            mean_pool=tile_pooled,
        )

        n_rows = token_info.get("n_rows")
        n_cols = token_info.get("n_cols")
        num_tiles_from_info = token_info.get("num_tiles")
        num_visual_tokens = token_info.get("num_visual_tokens")
        num_tiles_from_tokens = int(visual_embedding.shape[0]) // 64 + (
            1 if int(visual_embedding.shape[0]) % 64 else 0
        )

        report[str(sid)] = {
            "dataset": str(args.dataset),
            "model": str(args.model),
            "source_doc_id": str(sid),
            "doc_id": str(getattr(d, "doc_id", "")),
            "payload_source_doc_id": str((d.payload or {}).get("source_doc_id") or ""),
            "original_image": {
                "path": str(original_path),
                "width": int(original_img.width),
                "height": int(original_img.height),
            },
            "crop_meta": crop_meta,
            "cropped_image": {
                "path": str(cropped_path),
                "width": int(_ensure_pil(embed_img).width),
                "height": int(_ensure_pil(embed_img).height),
            },
            "processor": {
                "n_rows": None if n_rows is None else int(n_rows),
                "n_cols": None if n_cols is None else int(n_cols),
                "num_tiles": None if num_tiles_from_info is None else int(num_tiles_from_info),
                "num_visual_tokens": None if num_visual_tokens is None else int(num_visual_tokens),
                "visual_token_indices_len": int(len(visual_indices)),
                "num_tiles_from_visual_tokens_div64": int(num_tiles_from_tokens),
            },
            "embeddings": {
                "full_embedding_shape": [int(x) for x in emb_np.shape],
                "visual_embedding_shape": [int(x) for x in visual_embedding.shape],
                "mean_pool_shape": [int(x) for x in tile_pooled.shape],
                "experimental_pool_shape": [int(x) for x in experimental_pooled.shape],
            },
        }

    out_json = out_dir / f"{args.dataset.replace('/', '__')}__debug_report.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(str(out_json))


if __name__ == "__main__":
    main()
