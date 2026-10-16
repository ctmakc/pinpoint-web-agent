#!/usr/bin/env python3
"""Run questions.jsonl through app.py (one process per question, like the League harness) and grade."""
import json, os, re, subprocess, sys, time, unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.environ.get("PINPOINT_APP", os.path.join(HERE, "..", "app", "app.py"))
OUT = os.environ.get("PINPOINT_OUTPUT", "/tmp/pinpoint_out")


def squash(s):
    return re.sub(r"[\s\-.·_,]", "", unicodedata.normalize("NFKC", s)).upper()


def ok(answer, expect):
    if expect == [""]:
        return answer == ""
    a = squash(answer)
    return bool(a) and any(squash(e) in a or (len(a) > 0 and a in squash(e) and len(a) >= len(squash(e)) - 1) for e in expect if e)


QFILE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "questions.jsonl")
rows = [json.loads(l) for l in open(QFILE) if l.strip()]
results, passed = [], 0
for r in rows:
    cmd = [sys.executable, APP, "--query-id", r["id"], "--query", r["query"], "--output-dir", OUT]
    if r.get("url"):
        cmd += ["--url", r["url"]]
    t = time.time()
    subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    wall = round(time.time() - t, 1)
    try:
        out = json.load(open(os.path.join(OUT, f"{r['id']}_output.json")))
    except Exception as e:
        out = {"answer": "", "sources": [], "error": str(e)}
    good = ok(out.get("answer", ""), r["expect"])
    passed += good
    results.append({"id": r["id"], "query": r["query"], "url": r.get("url"), "expect": r["expect"],
                    "answer": out.get("answer", ""), "sources": out.get("sources", []), "pass": good,
                    "wall_s": wall, "steps": out.get("steps", [])})
    print(f"{r['id']} {'PASS' if good else 'FAIL'} {wall:5.1f}s  {out.get('answer','')!r:40.40} {out.get('sources', [])[:1]}", flush=True)
print(f"\n{passed}/{len(rows)} correct")
json.dump({"passed": passed, "total": len(rows), "results": results}, open(os.path.join(OUT, "eval.json"), "w"), indent=2)
