#!/usr/bin/env python3
"""Pinpoint entry point.

    python3 /app/app.py --query-id q01 --query "What is the memory capacity of the AMD Instinct MI300X?"
    python3 /app/app.py --query-id q02 --url https://www.python.org --query "Latest Python 3 release?"
    python3 /app/app.py --input /app/input/tasks.json        # list of {"id"/"query_id", "query"/"question", "url"}
    python3 /app/app.py --serve                              # start the resident model server (container CMD)

Writes /app/output/<query-id>_output.json with "answer" (value only, "" when not found) and "sources"
(the page URLs the answer was read from; also copied to "citations").
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pinpoint import server  # noqa: E402

OUT_DIR = os.environ.get("PINPOINT_OUTPUT", "/app/output")


def answer(question: str, url: str | None) -> dict:
    # Wait for the resident server if it is still loading; otherwise answer in-process.
    deadline = time.time() + float(os.environ.get("PINPOINT_WAIT_S", "20"))
    while time.time() < deadline:
        if os.path.exists(server.READY):
            try:
                return server.ask({"question": question, "url": url})
            except OSError:
                pass
        time.sleep(0.5)
    from pinpoint.agent import Agent
    from pinpoint.llm import make_llm
    return Agent(make_llm()).run(question, url).to_json()


def write(qid: str | None, result: dict) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    out = {"answer": result.get("answer", "") or "", "sources": result.get("sources", []) or [],
           "citations": result.get("sources", []) or [], "confidence": result.get("confidence", 0.0)}
    for k in ("evidence", "verified", "steps", "pages_read", "seconds", "error"):
        if k in result:
            out[k] = result[k]
    path = os.path.join(OUT_DIR, f"{qid}_output.json" if qid else "output.json")
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question", nargs="?", help="natural-language request")
    ap.add_argument("--query", "--question", "-q", dest="query")
    ap.add_argument("--query-id", "--id", "--task-id", dest="qid")
    ap.add_argument("--url", "--start-url", "--website", "--site", dest="url")
    ap.add_argument("--input", "--input-file", "--tasks", dest="input")
    ap.add_argument("--output-dir", dest="out")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--index", help="accepted for harness compatibility; nothing to index for live web")
    a, _unknown = ap.parse_known_args(argv)
    global OUT_DIR
    if a.out:
        OUT_DIR = a.out
    if a.serve:
        server.main()
        return
    if a.index is not None and not (a.query or a.question or a.input):
        print(json.dumps({"indexed": 0, "note": "live web, nothing to index"}))
        return
    tasks = []
    if a.input:
        data = json.load(open(a.input))
        items = data if isinstance(data, list) else data.get("tasks") or data.get("queries") or [data]
        for i, t in enumerate(items, 1):
            tasks.append((t.get("query_id") or t.get("id") or f"query_{i:02d}",
                          t.get("query") or t.get("question") or t.get("task"), t.get("url") or t.get("start_url")))
    else:
        q = a.query or a.question
        if not q:
            ap.error("give a question")
        tasks.append((a.qid, q, a.url))
    for qid, q, url in tasks:
        try:
            r = answer(q, url)
        except Exception as e:
            r = {"answer": "", "sources": [], "error": f"{type(e).__name__}: {e}"}
        path = write(qid, r)
        print(json.dumps({"id": qid, "answer": r.get("answer", ""), "sources": r.get("sources", []),
                          "seconds": r.get("seconds"), "file": path}, ensure_ascii=False))


if __name__ == "__main__":
    main()
