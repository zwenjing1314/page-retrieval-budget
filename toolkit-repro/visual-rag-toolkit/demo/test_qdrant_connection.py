#!/usr/bin/env python3
"""Test Qdrant connection and collection creation."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).parent.parent / ".env")
load_dotenv(Path(__file__).parent.parent.parent / ".env")


def test_connection():
    from qdrant_client import QdrantClient
    from qdrant_client.http import models

    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    print(f"URL: {url}")
    print(f"API Key: {'***' + api_key[-4:] if api_key else 'NOT SET'}")

    if not url or not api_key:
        print("ERROR: QDRANT_URL or QDRANT_API_KEY not set")
        return

    print("\n1. Creating client...")
    client = QdrantClient(url=url, api_key=api_key, timeout=60)

    print("\n2. Getting collections...")
    try:
        collections = client.get_collections()
        print(f"   Found {len(collections.collections)} collections:")
        for c in collections.collections:
            print(f"     - {c.name}")
    except Exception as e:
        print(f"   ERROR: {e}")
        return

    test_collection = "_test_visual_rag_toolkit"

    print(f"\n3. Checking if '{test_collection}' exists...")
    exists = any(c.name == test_collection for c in collections.collections)
    print(f"   Exists: {exists}")

    if exists:
        print("\n4. Deleting test collection...")
        try:
            client.delete_collection(test_collection)
            print("   Deleted")
        except Exception as e:
            print(f"   ERROR: {e}")

    print("\n5. Creating SIMPLE collection (single vector)...")
    try:
        client.create_collection(
            collection_name=test_collection,
            vectors_config=models.VectorParams(
                size=128,
                distance=models.Distance.COSINE,
            ),
        )
        print("   SUCCESS: Simple collection created")
    except Exception as e:
        print(f"   ERROR: {e}")
        print("\n   This means basic collection creation is failing.")
        print("   Check your Qdrant Cloud cluster status/limits.")
        return

    print("\n6. Deleting test collection...")
    try:
        client.delete_collection(test_collection)
        print("   Deleted")
    except Exception as e:
        print(f"   ERROR: {e}")

    print("\n7. Creating MULTI-VECTOR collection (like visual-rag)...")
    try:
        client.create_collection(
            collection_name=test_collection,
            vectors_config={
                "initial": models.VectorParams(
                    size=128,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                ),
                "mean_pooling": models.VectorParams(
                    size=128,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                ),
            },
        )
        print("   SUCCESS: Multi-vector collection created")
    except Exception as e:
        print(f"   ERROR: {e}")
        print("\n   Multi-vector collection failed but simple worked.")
        print("   Your Qdrant version may not support multi-vector.")
        return

    print("\n8. Final cleanup...")
    try:
        client.delete_collection(test_collection)
        print("   Deleted")
    except Exception as e:
        print(f"   ERROR: {e}")

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED - Qdrant connection is working!")
    print("=" * 50)


if __name__ == "__main__":
    test_connection()
