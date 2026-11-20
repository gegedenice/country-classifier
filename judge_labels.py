# /// script
# requires-python = ">=3.10"
# dependencies = ["openai>=1.40", "datasets>=3.0", "huggingface_hub>=0.30"]
# ///
"""LLM-as-a-judge validation of the rule labels (HF Inference Providers).

For each row, the judge sees the title/abstract and the countries proposed by the rules,
and decides for each one whether the publication is really ABOUT that country (study
area, data, case, policy, population...), as opposed to a passing mention (a journal,
an organisation, a funder, a comparison made in one sentence). It can also add countries
the rules missed (named through a city, region or institution).

The result is pushed as a new config "judged" of the dataset, with, for each row:
  labels         rule labels (unchanged)
  judge_labels   countries kept or added by the judge = the reference for evaluation
  judge_removed  rule labels rejected by the judge
  judge_added    countries added by the judge

Run on HF Jobs (CPU is enough):
  hf jobs uv run --flavor cpu-basic --secrets HF_TOKEN \
     -e DS_REPO=Geraldine/pub-country-classification-dataset-v2 \
     https://raw.githubusercontent.com/gegedenice/country-classifier/main/judge_labels.py

Env: HF_TOKEN, DS_REPO, JUDGE_MODEL, JUDGE_SPLITS (default "validation,test"),
     JUDGE_MAX_ROWS (per split, default all), JUDGE_CONCURRENCY (default 16).
"""
import asyncio
import json
import os
import re

from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import hf_hub_download
from openai import AsyncOpenAI

DS_REPO = os.environ.get("DS_REPO", "Geraldine/pub-country-classification-dataset-v2")
MODEL = os.environ.get("JUDGE_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507")
SPLITS = os.environ.get("JUDGE_SPLITS", "validation,test").split(",")
MAX_ROWS = int(os.environ.get("JUDGE_MAX_ROWS", "0"))
CONC = int(os.environ.get("JUDGE_CONCURRENCY", "16"))
TOKEN = os.environ["HF_TOKEN"]

lex = json.load(open(hf_hub_download(DS_REPO, "country_lexicon.json", repo_type="dataset", token=TOKEN)))
CODES = set(lex["codes"])
NAMES = lex["names"]

SYSTEM = """You are an expert indexer of scholarly publications. Your job: decide which countries a
publication is ABOUT, for geographic subject indexing in a library catalogue.

A country is ABOUT the publication when the country is a study area, a source of data, a case,
a population, a policy context or a subject of the work (a comparison between countries counts).
It is NOT about the publication when the name only appears as part of a journal, society,
company or organisation name ("American Journal of...", "German Research Foundation"), a funder,
an author affiliation, a language ("written in French"), a breed, species, food or disease name
("Spanish flu", "Guinea pig"), a US state or other place with a country-like name ("Georgia, USA",
"New Mexico"), or a passing example in a single clause with no relation to the study.

Use ISO 3166-1 alpha-2 codes (plus XK for Kosovo). Answer with JSON only:
{"keep": [codes from the candidate list that are about the publication],
 "add": [other countries clearly about the publication, e.g. named through a city or region]}"""


def prompt(text, labels):
    cands = ", ".join(f"{c} ({NAMES.get(c, c)})" for c in labels) or "none"
    return f"Publication (title, then abstract):\n<<<\n{text[:4000]}\n>>>\n\nCandidate countries from rules: {cands}"


def parse(raw, labels):
    m = re.search(r"\{.*\}", raw or "", re.S)
    j = json.loads(m.group(0))
    keep = sorted({c for c in j.get("keep", []) if c in labels})
    add = sorted({c for c in j.get("add", []) if c in CODES and c not in labels})
    return keep, add


async def judge_one(client, sem, text, labels):
    async with sem:
        for attempt in range(5):
            try:
                r = await client.chat.completions.create(
                    model=MODEL, temperature=0, max_tokens=200,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": prompt(text, labels)}])
                raw = r.choices[0].message.content
                keep, add = parse(raw, labels)
                return keep, add, raw, None
            except Exception as e:  # noqa: BLE001
                err = repr(e)
                await asyncio.sleep(2 ** attempt)
        return list(labels), [], "", err


async def run_split(client, ds):
    sem = asyncio.Semaphore(CONC)
    tasks = [judge_one(client, sem, t, l) for t, l in zip(ds["text"], ds["labels"])]
    out, done = [], 0
    for i in range(0, len(tasks), 500):
        out += await asyncio.gather(*tasks[i:i + 500])
        done += len(tasks[i:i + 500])
        print(f"  {done}/{len(tasks)}", flush=True)
    return out


async def main():
    client = AsyncOpenAI(base_url="https://router.huggingface.co/v1", api_key=TOKEN)
    # preflight: fail fast if the model is not served
    await client.chat.completions.create(model=MODEL, max_tokens=5, messages=[{"role": "user", "content": "ping"}])
    ds = load_dataset(DS_REPO, token=TOKEN)
    out = {}
    for split in SPLITS:
        d = ds[split]
        if MAX_ROWS:
            d = d.shuffle(seed=0).select(range(min(MAX_ROWS, len(d))))
        print(f"[judge] {split}: {len(d)} rows with {MODEL}", flush=True)
        res = await run_split(client, d)
        keep = [r[0] for r in res]
        add = [r[1] for r in res]
        d = d.add_column("judge_labels", [sorted(set(k) | set(a)) for k, a in zip(keep, add)])
        d = d.add_column("judge_removed", [sorted(set(l) - set(k)) for l, k in zip(d["labels"], keep)])
        d = d.add_column("judge_added", add)
        d = d.add_column("judge_error", [r[3] or "" for r in res])
        d = d.add_column("judge_model", [MODEL] * len(d))
        n_err = sum(bool(r[3]) for r in res)
        agree = sum(a == b for a, b in zip(d["labels"], d["judge_labels"])) / max(1, len(d))
        print(f"[judge] {split}: exact agreement rules/judge {agree:.3f}, errors {n_err}", flush=True)
        out[split] = d
    DatasetDict(out).push_to_hub(DS_REPO, config_name="judged", token=TOKEN)
    print("[done] pushed config 'judged'", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
