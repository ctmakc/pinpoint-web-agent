# Pinpoint

A web retrieval agent for **AMD AI League, Match 4: Intelligent Web Information Retrieval**.

You ask for one specific thing in plain English, optionally with a page to start from. Pinpoint
searches, opens pages, follows the links that look relevant, reads them with an open-weights model
running on an AMD GPU, and returns the value itself together with the URL of the page it was read from.
When no page it opened contains the value, it returns an empty answer.

```
$ python3 /app/app.py --query-id q01 --url https://pypi.org/project/requests/ \
      --query "Who is the author of the requests library?"
{"id": "q01", "answer": "Kenneth Reitz", "sources": ["https://pypi.org/project/requests/"], "seconds": 8.2, ...}
```

## How it works

1. **Start.** If the request carries a URL (`--url` or inside the text), Pinpoint opens it and, in the
   same step, the three links on it whose anchor text best matches the request. Otherwise it searches
   (DuckDuckGo Lite, DuckDuckGo HTML, Bing, then the Wikipedia API, first one that answers) and opens the top four results.
2. **Rank.** Every page becomes plain text (scripts, navigation and citation footers dropped, table
   rows kept on one line, JSON API responses flattened to `field: value` lines). The text is cut into
   ~700-character passages and the best eight are picked by term overlap with the request.
3. **Read.** Qwen2.5-7B-Instruct, held resident on the GPU, gets the numbered passages plus up to six
   candidate links and must reply with JSON: the value only, the passages it used, an exact quote, and
   optionally a link to open next.
4. **Verify.** The value must literally occur in the passage it was read from (after Unicode and
   whitespace normalisation). If it does not, the answer is dropped and the agent opens the suggested link
   or the next search results. Up to three read rounds, inside a 24-second budget per request.
5. **Answer.** `/app/output/<query-id>_output.json` gets `answer`, `sources` (also copied to
   `citations`), `evidence`, `confidence` and the step trace.

The model is loaded once by a resident server (`python3 /app/app.py --serve`, the container CMD).
Each `app.py` call is a thin client over a unix socket, so a new process per question costs well under
a second instead of a model load.

## Run on AMD

Measured on the AMD hackathon notebook pool (session image = the League base image
`rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0`), 16 Oct 2026:

| | |
|---|---|
| GPU | AMD Radeon Pro W7900D (gfx1100, 48 GB) |
| Stack | Python 3.14.4, torch 2.13.0+rocm10.0.0, HIP 7.15, transformers 5.20.0 |
| Model | Qwen/Qwen2.5-7B-Instruct, bf16 |
| Server ready (load + warm-up) | 25.4 s (weights already on disk; first download of 15 GB took 58 s in the pod) |
| VRAM peak | 15.0 GiB allocated by torch, 16.4 GiB device total (rocm-smi, sampled every 3 s) |
| Latency per request | 7.9 s mean, 16.1 s max, fresh `app.py` process each time |
| Result | **11 / 15** correct on [eval/questions_amd_pod.jsonl](eval/questions_amd_pod.jsonl) |

The notebook pod only reaches an allowlist of sites (python.org, pypi.org, arxiv.org, api.github.com and a
few more), so the AMD run uses questions on those sites. Search engines are blocked there, which is why
a08 (no start URL) fails in that run. Full log: [docs/amd-run/](docs/amd-run/).

| id | question | start | answer | result | seconds |
|---|---|---|---|---|---|
| a01 | What maximum line length does PEP 8 recommend? | https://peps.python.org/ | 79 characters | pass | 3.6 |
| a02 | Which PEP number is The Zen of Python? | https://peps.python.org/ | 20 | pass | 2.7 |
| a03 | Which PEP introduced formatted string literals (f-strings)? | https://peps.python.org/ | (empty) | fail | 16.1 |
| a04 | What is the title of PEP 572? | https://peps.python.org/ | PEP 572 – Assignment Expressions | pass | 8.1 |
| a05 | What HTTP status code means I'm a teapot? | https://docs.python.org/3/library/http.html | 418 | pass | 2.1 |
| a06 | Under which license is the requests package published? | https://pypi.org/project/requests/ | OSI Approved :: Apache Software License | pass | 8.0 |
| a07 | Who is the author of the requests library? | https://pypi.org/project/requests/ | Kenneth Reitz | pass | 8.2 |
| a08 | What is the title of arXiv paper 1706.03762? | (search) | (empty) | fail | 5.6 |
| a09 | Who is the first author listed on arXiv paper 1706.03762? | https://arxiv.org/abs/1706.03762 | Ashish Vaswani | pass | 15.6 |
| a10 | On what date was arXiv paper 2005.14165 first submitted? | https://arxiv.org/abs/2005.14165 | 28 May 2020 | pass | 8.8 |
| a11 | What license does the pytorch/pytorch GitHub repository use? | https://api.github.com/repos/pytorch/pytorch | Other | pass | 2.1 |
| a12 | What is the phone number of the Python Software Foundation office in Reykjavik? | https://www.python.org/psf/ | (empty) | pass | 13.2 |
| a13 | In which Python module is the function urljoin defined? | https://docs.python.org/3/library/ | (empty) | fail | 11.5 |
| a14 | What is the default value of the timeout parameter of socket.create_connection? | https://docs.python.org/3/library/socket.html | GLOBAL_DEFAULT | pass | 2.8 |
| a15 | Which PEP describes the Python packaging core metadata field Requires-Python? | https://peps.python.org/ | (empty) | fail | 10.1 |

Two of the failures are reading errors (a03, a15: the model named a wrong PEP and verification rejected
it, so the answer stayed empty instead of wrong), one is navigation (a13: the right page was not opened
within the budget), and a08 is the blocked search.

## Container

```
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0
```

- `/app/app.py`: entry point. Accepts `--query`, `--query-id`, `--url`, `--input <tasks.json>`,
  `--output-dir`, and `--index` (no-op, nothing to index for the live web).
- `/app/requirements.txt`: pinned; none of the packages depend on torch, so the ROCm torch of the base
  image stays in place.
- Model weights are downloaded at start-up into `/models` (or read from there if already present).
- No secrets, keys or tokens anywhere: every provider used is keyless.

See [Dockerfile](Dockerfile). The submitted image adds exactly one layer (code + pinned wheels) on top
of the unmodified base layers, see [build/assemble.sh](build/assemble.sh).

## Run it yourself

```bash
# on any ROCm machine with the base image's Python
pip install --no-deps -r app/requirements.txt
python3 app/app.py --serve &            # loads the model once
python3 app/app.py --query "How tall is the CN Tower in metres?"
python3 eval/run_eval.py eval/questions.jsonl   # open-web set (needs unrestricted egress)
```

Environment knobs: `PINPOINT_MODEL` (default `Qwen/Qwen2.5-7B-Instruct`), `PINPOINT_MODEL_DIR`
(`/models`), `PINPOINT_BUDGET_S` (24), `PINPOINT_OUTPUT` (`/app/output`).

## Limits

- Plain HTTP fetching: pages that render their content only with JavaScript come back nearly empty.
- PDFs are skipped.
- Keyless search can be rate-limited from busy IPs; the agent then falls through to the next provider.

## License

MIT, see [LICENSE](LICENSE). Qwen2.5-7B-Instruct is Apache-2.0.
