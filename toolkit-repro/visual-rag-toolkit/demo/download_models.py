#!/usr/bin/env python3
"""Pre-download HuggingFace models for Visual RAG Toolkit.

This script downloads models during Docker build to cache them in the image,
avoiding download delays during container startup.
"""

import os

os.environ.setdefault("HF_HOME", "/app/.cache/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/app/.cache/huggingface")

MODELS_TO_DOWNLOAD = [
    "vidore/colpali-v1.3",
    "vidore/colSmol-500M",
    "vidore/colqwen2.5-v0.2",
]


def download_colpali_models():
    """Download ColPali models and their processors."""
    print("=" * 60)
    print("Downloading ColPali models for Visual RAG Toolkit")
    print("=" * 60)

    try:
        from colpali_engine.models import ColPali, ColPaliProcessor
    except ImportError:
        print("[WARN] colpali-engine not installed, trying transformers directly")
        from transformers import AutoModel, AutoProcessor

        for model_name in MODELS_TO_DOWNLOAD:
            print(f"\n[INFO] Downloading model: {model_name}")
            try:
                AutoModel.from_pretrained(model_name, trust_remote_code=True)
                AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
                print(f"[OK] Downloaded: {model_name}")
            except Exception as e:
                print(f"[WARN] Could not download {model_name}: {e}")
        return

    for model_name in MODELS_TO_DOWNLOAD:
        print(f"\n[INFO] Downloading model: {model_name}")
        try:
            model_lower = model_name.lower()
            if "colqwen2.5" in model_lower or "colqwen2_5" in model_lower:
                from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

                ColQwen2_5.from_pretrained(model_name, trust_remote_code=True)
                ColQwen2_5_Processor.from_pretrained(model_name, trust_remote_code=True)
            elif "colqwen" in model_lower:
                from colpali_engine.models import ColQwen2, ColQwen2Processor

                ColQwen2.from_pretrained(model_name, trust_remote_code=True)
                ColQwen2Processor.from_pretrained(model_name, trust_remote_code=True)
            else:
                ColPali.from_pretrained(model_name, trust_remote_code=True)
                ColPaliProcessor.from_pretrained(model_name, trust_remote_code=True)
            print(f"[OK] Downloaded: {model_name}")
        except Exception as e:
            print(f"[WARN] Could not download {model_name} with colpali-engine: {e}")
            try:
                from transformers import AutoModel, AutoProcessor

                AutoModel.from_pretrained(model_name, trust_remote_code=True)
                AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
                print(f"[OK] Downloaded via transformers: {model_name}")
            except Exception as e2:
                print(f"[ERROR] Failed to download {model_name}: {e2}")


def main():
    print(f"[INFO] HF_HOME: {os.environ.get('HF_HOME', 'not set')}")
    print(f"[INFO] Cache dir: {os.environ.get('TRANSFORMERS_CACHE', 'not set')}")

    download_colpali_models()

    print("\n" + "=" * 60)
    print("Model download complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
