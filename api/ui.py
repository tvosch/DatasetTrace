"""
infini-gram-mini Gradio UI.

Connects to the API server and provides a simple web interface for:
  - Counting n-gram occurrences across indexed corpora
  - Searching for and browsing matching documents

Usage
-----
    # Start the API server first:
    python api_server.py --config api_config.json --port 5000

    # Then launch the UI:
    python ui.py --api_url http://localhost:5000 --port 7860

    # Or create a public share link (useful for demos):
    python ui.py --api_url http://localhost:5000 --share
"""

from __future__ import annotations

import argparse
import html as _html
import json

import gradio as gr
import requests

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _post(api_url: str, payload: dict, timeout: int = 60) -> tuple[dict, int]:
    """POST *payload* to the API and return (response_dict, http_status)."""
    try:
        r = requests.post(f"{api_url}/query", json=payload, timeout=timeout)
        return r.json(), r.status_code
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. The server may be busy."}, 504
    except requests.exceptions.ConnectionError:
        return {"error": f"Could not connect to API server at {api_url}."}, 503
    except Exception as exc:
        return {"error": f"Unexpected error: {exc}"}, 500


def _get_indexes(api_url: str) -> list[str]:
    """Fetch the list of available index names from the API."""
    try:
        r = requests.get(f"{api_url}/indexes", timeout=10)
        return r.json().get("indexes", [])
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Result rendering
# ---------------------------------------------------------------------------

_CARD_STYLE = (
    "border:1px solid #ddd; border-radius:8px; padding:14px; margin:10px 0; "
    "font-family:monospace; font-size:13px; line-height:1.6; background:#fafafa;"
)
_META_STYLE = "color:#666; font-size:11px; margin-bottom:6px; word-break:break-all;"
_MARK_STYLE = "background:#ffe066; padding:0 2px; border-radius:3px;"


def _render_doc(doc: dict, query: str) -> str:
    """Render a single DocResponse as an HTML card with highlighted matches."""
    if "error" in doc:
        msg = _html.escape(doc["error"])
        return f'<div style="{_CARD_STYLE} border-color:#f88;"><em>{msg}</em></div>'

    # Metadata line (source path + any other fields)
    meta_raw = doc.get("metadata", "")
    try:
        meta_obj = json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw
        meta_str = _html.escape(
            meta_obj.get("source", str(meta_raw)) if isinstance(meta_obj, dict) else str(meta_raw)
        )
    except Exception:
        meta_str = _html.escape(str(meta_raw))

    # Build highlighted text from pre-computed spans (if present), else plain text.
    spans = doc.get("spans") or [(doc.get("text", ""), None)]
    parts: list[str] = []
    for fragment, label in spans:
        escaped = _html.escape(fragment).replace("\n", "<br>")
        if label is not None:
            parts.append(f'<mark style="{_MARK_STYLE}">{escaped}</mark>')
        else:
            parts.append(escaped)

    text_html = "".join(parts)
    return (
        f'<div style="{_CARD_STYLE}">'
        f'<div style="{_META_STYLE}">{meta_str}</div>'
        f'<div>{text_html}</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Tab implementations
# ---------------------------------------------------------------------------

def run_count(api_url: str, index: str, query: str) -> str:
    """Count tab: return markdown-formatted count result."""
    if not query.strip():
        return "_Please enter a query._"
    if not index:
        return "_Please select an index._"

    data, status = _post(api_url, {"index": index, "query_type": "count", "query": query})
    if "error" in data:
        return f"**Error:** {data['error']}"

    count   = data.get("count", 0)
    latency = data.get("latency", 0)
    return f"**{count:,}** occurrence(s)  \n_Latency: {latency:.1f} ms_"


def run_search(
    api_url: str,
    index: str,
    query: str,
    max_results: int,
    max_ctx_len: int,
) -> str:
    """Search tab: return HTML-formatted document cards."""
    if not query.strip():
        return "<p><em>Please enter a query.</em></p>"
    if not index:
        return "<p><em>Please select an index.</em></p>"

    data, status = _post(api_url, {
        "index":       index,
        "query_type":  "search",
        "query":       query,
        "max_results": int(max_results),
        "max_ctx_len": int(max_ctx_len),
    })

    if "error" in data:
        return f"<p><strong>Error:</strong> {_html.escape(data['error'])}</p>"

    docs = data.get("results", [])
    if not docs:
        return "<p><em>No results found.</em></p>"

    latency = data.get("latency", 0)
    header  = f"<p style='color:#666; font-size:12px;'>Showing {len(docs)} result(s) — {latency:.1f} ms</p>"
    cards   = "".join(_render_doc(doc, query) for doc in docs)
    return header + cards


def run_find(api_url: str, index: str, query: str) -> str:
    """Find tab: return raw suffix-array interval JSON (advanced / debugging view)."""
    if not query.strip():
        return "_Please enter a query._"
    if not index:
        return "_Please select an index._"

    data, _ = _post(api_url, {"index": index, "query_type": "find", "query": query})
    if "error" in data:
        return f"**Error:** {data['error']}"

    cnt      = data.get("cnt", 0)
    segments = data.get("segment_by_shard", [])
    latency  = data.get("latency", 0)

    lines = [f"**Total occurrences:** {cnt:,}  \n_Latency: {latency:.1f} ms_\n"]
    for i, (lo, hi) in enumerate(segments):
        lines.append(f"- Shard {i}: [{lo:,}, {hi:,})  (count: {hi - lo:,})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------

def build_ui(api_url: str, index_names: list[str]) -> gr.Blocks:
    """Construct the Gradio interface."""
    default_index = index_names[0] if index_names else None

    with gr.Blocks(title="infini-gram-mini", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# infini-gram-mini\n"
            "Exact n-gram search over large text corpora via FM-index."
        )

        # Shared index selector at the top
        index_dd = gr.Dropdown(
            choices=index_names,
            value=default_index,
            label="Index",
            info="Select the corpus to search.",
        )

        # ---- Count tab ----
        with gr.Tab("Count"):
            gr.Markdown("Count how many times a query string appears in the corpus.")
            q_count  = gr.Textbox(label="Query", placeholder="natural language processing")
            btn_count = gr.Button("Count", variant="primary")
            out_count = gr.Markdown()

            btn_count.click(
                fn=lambda idx, q: run_count(api_url, idx, q),
                inputs=[index_dd, q_count],
                outputs=out_count,
            )
            q_count.submit(
                fn=lambda idx, q: run_count(api_url, idx, q),
                inputs=[index_dd, q_count],
                outputs=out_count,
            )

        # ---- Search tab ----
        with gr.Tab("Search"):
            gr.Markdown("Search for a query string and browse matching document contexts.")
            q_search = gr.Textbox(label="Query", placeholder="transformer model")
            with gr.Row():
                max_results = gr.Slider(1, 50,   value=10,  step=1,  label="Max results")
                max_ctx_len = gr.Slider(50, 2000, value=200, step=50, label="Context (bytes)")
            btn_search  = gr.Button("Search", variant="primary")
            out_search  = gr.HTML()

            btn_search.click(
                fn=lambda idx, q, mr, mc: run_search(api_url, idx, q, mr, mc),
                inputs=[index_dd, q_search, max_results, max_ctx_len],
                outputs=out_search,
            )
            q_search.submit(
                fn=lambda idx, q, mr, mc: run_search(api_url, idx, q, mr, mc),
                inputs=[index_dd, q_search, max_results, max_ctx_len],
                outputs=out_search,
            )

        # ---- Find (advanced) tab ----
        with gr.Tab("Find (advanced)"):
            gr.Markdown(
                "Return the raw suffix-array intervals per shard.  \n"
                "Useful for debugging or building custom result retrieval."
            )
            q_find   = gr.Textbox(label="Query", placeholder="large language models")
            btn_find = gr.Button("Find", variant="primary")
            out_find = gr.Markdown()

            btn_find.click(
                fn=lambda idx, q: run_find(api_url, idx, q),
                inputs=[index_dd, q_find],
                outputs=out_find,
            )
            q_find.submit(
                fn=lambda idx, q: run_find(api_url, idx, q),
                inputs=[index_dd, q_find],
                outputs=out_find,
            )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the infini-gram-mini Gradio UI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--api_url", default="http://localhost:5000",
                        help="Base URL of the running API server.")
    parser.add_argument("--port",    type=int, default=7860,
                        help="Local port to serve the Gradio app on.")
    parser.add_argument("--share",   action="store_true",
                        help="Create a public Gradio share link (useful for demos).")
    args = parser.parse_args()

    print(f"Fetching index list from {args.api_url} …")
    index_names = _get_indexes(args.api_url)
    if not index_names:
        print("Warning: no indexes found. The API server may not be running yet.")

    demo = build_ui(args.api_url, index_names)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
