import argparse
from pprint import pprint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=str, required=True)
    parser.add_argument("--prefer-grpc", action="store_true", default=False)
    args = parser.parse_args()

    from visual_rag import QdrantAdmin

    admin = QdrantAdmin(prefer_grpc=bool(args.prefer_grpc), timeout=60)
    before = admin.get_collection_info(collection_name=str(args.collection))
    print("BEFORE points_count=", before.get("points_count"))
    existing = sorted(
        (((before.get("config") or {}).get("params") or {}).get("vectors") or {}).keys()
    )
    print("BEFORE vectors=", existing)

    after = admin.ensure_collection_all_on_disk(collection_name=str(args.collection))

    print("AFTER points_count=", after.get("points_count"))
    print("AFTER params.vectors:")
    pprint(((after.get("config") or {}).get("params") or {}).get("vectors"))


if __name__ == "__main__":
    main()
