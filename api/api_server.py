"""
infini-gram-mini REST API server.

Every query is a POST to /query (or / for backwards compatibility) with a JSON body:

    {
        "index":      "<index name from config>",
        "query_type": "count" | "find" | "get_doc_by_rank" | "search",
        "query":      "<UTF-8 text>",

        # Additional fields per query_type:
        # get_doc_by_rank: "s" (int), "rank" (int), "max_ctx_len" (int, default 200)
        # search:          "max_results" (int, default 10), "max_ctx_len" (int, default 200)
    }

All responses include a "latency" field (milliseconds).

Usage
-----
    python api_server.py --config api_config.json --port 5000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from typing import Any

from flask import Flask, jsonify, request

# Engine import — works when the package is installed via `pip install -e .`
# or when PYTHONPATH includes the repo root.
try:
    from engine.src.engine import InfiniGramMiniEngine
except ImportError:
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    from engine.src.engine import InfiniGramMiniEngine

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _highlight_spans(text: str, needle: str) -> list[tuple[str, str | None]]:
    """Split *text* into (fragment, label) pairs, marking every occurrence of *needle*.

    Unlabelled fragments have ``label=None``; matched fragments have ``label='0'``.
    This format is consumed by the Gradio UI to render highlighted results.
    """
    if not needle:
        return [(text, None)]

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
    return spans


# ---------------------------------------------------------------------------
# Processor — wraps one InfiniGramMiniEngine with query dispatch
# ---------------------------------------------------------------------------

class Processor:
    """Holds a loaded engine for one named index and dispatches query calls."""

    def __init__(self, config: dict[str, Any]) -> None:
        for key in ("name", "index_dirs", "load_to_ram", "get_metadata"):
            if key not in config:
                raise KeyError(f"Index config is missing required key: {key!r}")

        t0 = time.perf_counter()
        self.engine = InfiniGramMiniEngine(
            index_dirs=config["index_dirs"],
            load_to_ram=config["load_to_ram"],
            get_metadata=config["get_metadata"],
        )
        elapsed = time.perf_counter() - t0
        log.info('Loaded index "%s" (%d shard(s)) in %.2f s',
                 config["name"], self.engine.num_shards, elapsed)

    # ------------------------------------------------------------------
    # Query methods (each returns a dict with a "latency" key added by dispatch)
    # ------------------------------------------------------------------

    def count(self, query: str) -> dict:
        return self.engine.count(query)

    def find(self, query: str) -> dict:
        return self.engine.find(query)

    def get_doc_by_rank(
        self, query: str, s: int, rank: int, max_ctx_len: int = 200,
    ) -> dict:
        result = self.engine.get_doc_by_rank(
            s=s, rank=rank, needle_len=len(query.encode()), max_ctx_len=max_ctx_len,
        )
        if "error" not in result:
            result["spans"] = _highlight_spans(result["text"], query)
        return result

    def search(
        self, query: str, max_results: int = 10, max_ctx_len: int = 200,
    ) -> dict:
        """Run a search and return up to *max_results* documents with span highlights."""
        docs = self.engine.search(query, max_results=max_results, max_ctx_len=max_ctx_len)
        for doc in docs:
            if "error" not in doc:
                doc["spans"] = _highlight_spans(doc["text"], query)
        return {"results": docs}

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    _ALLOWED_QUERY_TYPES = frozenset({"count", "find", "get_doc_by_rank", "search"})

    def dispatch(self, query_type: str, query: str, **kwargs: Any) -> dict:
        """Validate and call the appropriate query method, returning the result dict."""
        if not isinstance(query, str):
            return {"error": "'query' must be a string."}
        if query_type not in self._ALLOWED_QUERY_TYPES:
            return {"error": f"Unknown query_type {query_type!r}. "
                             f"Allowed: {sorted(self._ALLOWED_QUERY_TYPES)}."}

        t0 = time.perf_counter()
        result = getattr(self, query_type)(query, **kwargs)
        result["latency"] = (time.perf_counter() - t0) * 1000  # ms
        return result


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def load_processors(config_path: str) -> dict[str, Processor]:
    """Load all indexes defined in *config_path* and return a name→Processor map."""
    with open(config_path) as fh:
        configs = json.load(fh)
    if not isinstance(configs, list):
        raise ValueError(f"Config file must contain a JSON array, got {type(configs)}")

    processors: dict[str, Processor] = {}
    for cfg in configs:
        name = cfg.get("name", "<unnamed>")
        try:
            processors[name] = Processor(cfg)
        except Exception as exc:
            log.error('Failed to load index "%s": %s', name, exc)
    return processors


def create_app(processors: dict[str, Processor]) -> Flask:
    """Create and return the Flask application.

    Separating creation from ``app.run`` makes the server testable and
    compatible with WSGI runners such as Gunicorn::

        gunicorn "api_server:create_app_from_config('api_config.json')" -w 1 -b 0.0.0.0:5000
    """
    app = Flask(__name__)

    # ------------------------------------------------------------------ health
    @app.get("/health")
    def health():
        """Liveness probe — returns 200 when the server is ready."""
        return jsonify({"status": "ok", "indexes": list(processors)})

    # ----------------------------------------------------------------- indexes
    @app.get("/indexes")
    def list_indexes():
        """Return the names of all loaded indexes."""
        return jsonify({"indexes": list(processors)})

    # ------------------------------------------------------------------- query
    def _handle_query():
        data: dict = request.get_json(force=True, silent=True) or {}
        log.debug("Request: %s", json.dumps(data))

        # Extract and validate required top-level fields.
        try:
            query_type = data["query_type"]
            index      = data["index"]
            query      = data["query"]
        except KeyError as exc:
            return jsonify({"error": f"Missing required field: {exc}"}), 400

        processor = processors.get(index)
        if processor is None:
            return jsonify({"error": f"Unknown index: {index!r}. "
                                     f"Available: {list(processors)}."}), 400

        # Pass remaining fields as kwargs to the query method.
        extra = {k: v for k, v in data.items()
                 if k not in {"query_type", "index", "query"}}

        try:
            result = processor.dispatch(query_type, query, **extra)
        except TypeError as exc:
            # Wrong kwargs for the chosen query_type.
            return jsonify({"error": f"Bad arguments for {query_type!r}: {exc}"}), 400
        except Exception as exc:
            log.exception("Unhandled error processing request")
            return jsonify({"error": f"Internal server error: {exc}"}), 500

        return jsonify(result), 200

    # Bind to both / (backwards compatibility) and /query.
    app.add_url_rule("/",      "query_root",  _handle_query, methods=["POST"])
    app.add_url_rule("/query", "query",       _handle_query, methods=["POST"])

    return app


def create_app_from_config(config_path: str) -> Flask:
    """Entry point for WSGI runners (e.g. Gunicorn)."""
    processors = load_processors(config_path)
    return create_app(processors)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="infini-gram-mini API server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",   default="api_config.json",
                        help="Path to the JSON index config file.")
    parser.add_argument("--port",     type=int, default=5000,
                        help="Port to listen on.")
    parser.add_argument("--host",     default="0.0.0.0",
                        help="Host/interface to bind to.")
    parser.add_argument("--workers",  type=int, default=1,
                        help="Number of Flask worker processes. For production, "
                             "prefer running under Gunicorn instead.")
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging verbosity.")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    processors = load_processors(args.config)
    if not processors:
        log.error("No indexes loaded — check your config file. Exiting.")
        sys.exit(1)

    app = create_app(processors)

    log.info("Starting server on %s:%d with %d worker(s).", args.host, args.port, args.workers)
    log.info("Loaded indexes: %s", list(processors))

    # For production traffic, run behind Gunicorn:
    #   gunicorn "api_server:create_app_from_config('api_config.json')" \
    #       -w 1 --threads 4 -b 0.0.0.0:5000
    app.run(
        host=args.host,
        port=args.port,
        threaded=(args.workers == 1),
        processes=args.workers if args.workers > 1 else 1,
        debug=False,
    )


if __name__ == "__main__":
    main()
