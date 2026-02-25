"""
infini-gram-mini CLI query tool.

Usage
-----
    # Count occurrences:
    python query.py --config api_config.json count "natural language processing"

    # Search and print matching document contexts:
    python query.py --config api_config.json search "transformer model"

    # Use a specific index (default: first one in config):
    python query.py --config api_config.json --index v2_pileval count "BERT"

    # Search with custom result count and context window:
    python query.py --config api_config.json search "GPT" --max_results 5 --max_ctx_len 300

    # Or query a running API server instead of loading the engine locally:
    python query.py --api_url http://localhost:5000 count "BERT"
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _print_count(result: dict) -> None:
    count   = result.get("count", 0)
    latency = result.get("latency", 0)
    print(f"Count:   {count:,}")
    print(f"Latency: {latency:.1f} ms")


def _print_search(result: dict, query: str) -> None:
    docs    = result.get("results", [])
    latency = result.get("latency", 0)
    print(f"Results: {len(docs)}  |  Latency: {latency:.1f} ms\n")
    for i, doc in enumerate(docs, 1):
        if "error" in doc:
            print(f"[{i}] Error: {doc['error']}\n")
            continue
        meta = doc.get("metadata", "")
        try:
            meta_obj = json.loads(meta) if isinstance(meta, str) else meta
            source   = meta_obj.get("source", str(meta)) if isinstance(meta_obj, dict) else str(meta)
        except Exception:
            source = str(meta)

        # Reconstruct plain text from spans, marking the query with >>>...<<<
        spans = doc.get("spans") or [(doc.get("text", ""), None)]
        parts = []
        for fragment, label in spans:
            parts.append(f">>>{fragment}<<<" if label is not None else fragment)
        text = "".join(parts)

        print(f"[{i}] {source}")
        print(textwrap.indent(text.strip(), "    "))
        print()


def _print_find(result: dict) -> None:
    cnt      = result.get("cnt", 0)
    segments = result.get("segment_by_shard", [])
    latency  = result.get("latency", 0)
    print(f"Count:   {cnt:,}")
    print(f"Latency: {latency:.1f} ms")
    for i, (lo, hi) in enumerate(segments):
        print(f"  Shard {i}: [{lo:,}, {hi:,})  ({hi - lo:,} occurrences)")


# ---------------------------------------------------------------------------
# Query backends
# ---------------------------------------------------------------------------

def _query_via_api(api_url: str, index: str, query_type: str, query: str, extra: dict) -> dict:
    import requests
    payload = {"index": index, "query_type": query_type, "query": query, **extra}
    try:
        r = requests.post(f"{api_url}/query", json=payload, timeout=60)
        return r.json()
    except requests.exceptions.ConnectionError:
        print(f"Error: could not connect to API server at {api_url}", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.Timeout:
        print("Error: request timed out.", file=sys.stderr)
        sys.exit(1)


def _query_direct(config_path: str, index_name: str, query_type: str, query: str, extra: dict) -> dict:
    """Load the engine directly (no API server needed)."""
    try:
        from engine.src.engine import InfiniGramMiniEngine
    except ImportError:
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from engine.src.engine import InfiniGramMiniEngine

    with open(config_path) as fh:
        configs = json.load(fh)

    cfg = next((c for c in configs if c["name"] == index_name), None)
    if cfg is None:
        names = [c["name"] for c in configs]
        print(f"Error: index {index_name!r} not found. Available: {names}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading index '{index_name}' …", file=sys.stderr)
    t0     = time.perf_counter()
    engine = InfiniGramMiniEngine(
        index_dirs=cfg["index_dirs"],
        load_to_ram=cfg["load_to_ram"],
        get_metadata=cfg["get_metadata"],
    )
    print(f"Loaded in {time.perf_counter() - t0:.2f} s\n", file=sys.stderr)

    t0 = time.perf_counter()
    if query_type == "count":
        result = engine.count(query)
    elif query_type == "find":
        result = engine.find(query)
    elif query_type == "search":
        result = {"results": engine.search(query, **extra)}
    else:
        print(f"Error: unknown query_type {query_type!r}", file=sys.stderr)
        sys.exit(1)
    result["latency"] = (time.perf_counter() - t0) * 1000

    # Add highlight spans for search results.
    if query_type == "search":
        for doc in result["results"]:
            if "error" not in doc:
                text   = doc.get("text", "")
                needle = query
                spans: list[tuple[str, str | None]] = []
                n = len(needle)
                while text:
                    pos = text.find(needle)
                    if pos == -1:
                        break
                    if pos > 0:
                        spans.append((text[:pos], None))
                    spans.append((text[pos : pos + n], "0"))
                    text = text[pos + n:]
                if text:
                    spans.append((text, None))
                doc["spans"] = spans

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query an infini-gram-mini index from the command line.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Source: either a local config file (direct engine load) or a running API server.
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--config",  default="api_config.json",
                     help="Path to the index config JSON (loads engine directly, no server needed).")
    src.add_argument("--api_url", default=None,
                     help="Base URL of a running API server (e.g. http://localhost:5000).")

    parser.add_argument("--index", default=None,
                        help="Index name to query (default: first entry in config).")

    subparsers = parser.add_subparsers(dest="query_type", required=True)

    # count
    p_count = subparsers.add_parser("count", help="Count occurrences of a query string.")
    p_count.add_argument("query", help="Query string.")

    # search
    p_search = subparsers.add_parser("search", help="Search and print matching document contexts.")
    p_search.add_argument("query",                                                  help="Query string.")
    p_search.add_argument("--max_results", type=int, default=10,                   help="Maximum number of documents to return.")
    p_search.add_argument("--max_ctx_len", type=int, default=200,                  help="Context window size in bytes.")

    # find
    p_find = subparsers.add_parser("find", help="Return raw suffix-array intervals per shard.")
    p_find.add_argument("query", help="Query string.")

    args = parser.parse_args()

    # Resolve index name.
    if args.api_url:
        # When using the API, fetch index list to pick a default if needed.
        if args.index is None:
            import requests
            try:
                r = requests.get(f"{args.api_url}/indexes", timeout=10)
                indexes = r.json().get("indexes", [])
                args.index = indexes[0] if indexes else ""
            except Exception:
                print("Error: could not fetch index list from API server.", file=sys.stderr)
                sys.exit(1)
    else:
        if args.index is None:
            with open(args.config) as fh:
                configs = json.load(fh)
            args.index = configs[0]["name"] if configs else ""

    # Build extra kwargs.
    extra: dict = {}
    if args.query_type == "search":
        extra = {"max_results": args.max_results, "max_ctx_len": args.max_ctx_len}

    # Dispatch.
    if args.api_url:
        result = _query_via_api(args.api_url, args.index, args.query_type, args.query, extra)
    else:
        result = _query_direct(args.config, args.index, args.query_type, args.query, extra)

    if "error" in result:
        print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)

    if args.query_type == "count":
        _print_count(result)
    elif args.query_type == "search":
        _print_search(result, args.query)
    elif args.query_type == "find":
        _print_find(result)


if __name__ == "__main__":
    main()
