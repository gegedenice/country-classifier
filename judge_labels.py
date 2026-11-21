# /// script
# requires-python = ">=3.10"
# dependencies = ["openai>=1.40", "datasets>=3.0", "huggingface_hub>=0.30", "pandas", "pyarrow"]
# ///
"""LLM check of the rule labels: is each country MENTION correct?

The task is deliberately narrow. For each publication the LLM sees the text and the
countries found by the rules, and answers two questions:
  1. reject: which of these are NOT a mention of the country? (homonyms: a species,
     a city or US state with the same name, a person's name, a word in another sense);
  2. missed: which countries are explicitly NAMED in the text (name, variant or
     demonym) but missing from the list? Each one must come with the exact words of
     the text; the script checks that those words really appear in the text.
It never judges relevance ("is the paper about it?") and never infers a country from
the language of the text, a city or an institution.

    verified_labels = (rule labels - rejected) + missed (with a quote found in the text)

Output: parquet files verified/{split}-NNNNN.parquet in the dataset repo, written chunk
by chunk, so an interrupted job resumes where it stopped. train_country_model.py uses
them automatically.

Run on HF Jobs (CPU is enough), from a clone of the repository:
  hf jobs uv run --flavor cpu-basic --timeout 8h --secrets HF_TOKEN \
     -e DS_REPO=Geraldine/pub-country-classification-dataset-v2.1 judge_labels.py

Env: HF_TOKEN, DS_REPO, JUDGE_MODEL, JUDGE_SPLITS (default "validation,test,train"),
     JUDGE_CONCURRENCY (default 32), CHUNK (default 5000).
"""
import asyncio
import io
import json
import os
import re

import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from openai import AsyncOpenAI

DS_REPO = os.environ.get("DS_REPO", "Geraldine/pub-country-classification-dataset-v2.1")
MODEL = os.environ.get("JUDGE_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507")
SPLITS = os.environ.get("JUDGE_SPLITS", "validation,test,train").split(",")
CONC = int(os.environ.get("JUDGE_CONCURRENCY", "32"))
CHUNK = int(os.environ.get("CHUNK", "5000"))
TOKEN = os.environ["HF_TOKEN"]
PROMPT_VERSION = "mention-check-1"

api = HfApi(token=TOKEN)
lex = json.load(open(hf_hub_download(DS_REPO, "country_lexicon.json", repo_type="dataset", token=TOKEN)))
CODES = set(lex["codes"])
NAMES = lex["names"]

SYSTEM = """You check automatic country tags on scholarly publications. A tag is correct when the
text explicitly MENTIONS that country: its name, a variant (e.g. "USA", "Côte d'Ivoire") or a
demonym ("Kenyan", "brasileiro", "sénégalais"), used for the country itself, its people, or
something from it.

Reject a tag when the word that triggered it means something else: a species ("Anguilla anguilla"),
a city, region or US state with the same name ("Salvador, Bahia", "Granada, Spain", "Georgia, USA",
"New Jersey"), a person's name ("Dominique", "Jordan"), a language name used only as a language
("written in French"), or any other homonym.

Do NOT judge relevance: a correct mention stays, even a passing one. Do NOT add a country because
of the language of the text, a city, a region or an institution: add it only if the country itself
is named. Codes are ISO 3166-1 alpha-2 (XK = Kosovo).

Answer with JSON only:
{"reject": [codes from the tag list that are not mentions],
 "missed": [{"code": "XX", "quote": "exact words from the text naming the country"}]}"""


def user_prompt(text, labels):
    tags = ", ".join(f"{c} ({NAMES.get(c, c)})" for c in labels) or "none"
    return f"Text:\n<<<\n{text[:4000]}\n>>>\n\nTags: {tags}"


def parse(raw, text, labels):
    j = json.loads(re.search(r"\{.*\}", raw or "", re.S).group(0))
    reject = sorted({c for c in j.get("reject", []) if c in labels})
    low = text.casefold()
    missed, quotes = [], []
    for m in j.get("missed", []) or []:
        c, q = m.get("code"), (m.get("quote") or "").strip()
        if c in CODES and c not in labels and len(q) >= 3 and q.casefold() in low:
            missed.append(c)
            quotes.append(f"{c}:{q}")
    return reject, sorted(set(missed)), quotes


async def check(client, sem, text, labels):
    async with sem:
        err = ""
        for attempt in range(5):
            try:
                r = await client.chat.completions.create(
                    model=MODEL, temperature=0, max_tokens=200,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": user_prompt(text, labels)}])
                return (*parse(r.choices[0].message.content, text, labels), "")
            except Exception as e:  # noqa: BLE001
                err = repr(e)[:300]
                await asyncio.sleep(2 ** attempt)
        return [], [], [], err          # on failure the rule labels are kept unchanged


async def main():
    client = AsyncOpenAI(base_url="https://router.huggingface.co/v1", api_key=TOKEN)
    await client.chat.completions.create(model=MODEL, max_tokens=5, messages=[{"role": "user", "content": "ping"}])
    done = set(api.list_repo_files(DS_REPO, repo_type="dataset"))
    ds = load_dataset(DS_REPO, token=TOKEN)
    sem = asyncio.Semaphore(CONC)
    for split in SPLITS:
        d = ds[split]
        n_chunks = (len(d) + CHUNK - 1) // CHUNK
        for k in range(n_chunks):
            path = f"verified/{split}-{k:05d}.parquet"
            if path in done:
                print(f"[skip] {path} already done", flush=True)
                continue
            part = d.select(range(k * CHUNK, min(len(d), (k + 1) * CHUNK)))
            res = await asyncio.gather(*[check(client, sem, t, l) for t, l in zip(part["text"], part["labels"])])
            df = part.to_pandas()
            df["judge_rejected"] = [r[0] for r in res]
            df["judge_added"] = [r[1] for r in res]
            df["judge_quotes"] = [r[2] for r in res]
            df["judge_error"] = [r[3] for r in res]
            df["verified_labels"] = [sorted((set(l) - set(r[0])) | set(r[1])) for l, r in zip(part["labels"], res)]
            df["judge_model"] = MODEL
            df["judge_prompt"] = PROMPT_VERSION
            buf = io.BytesIO()
            df.to_parquet(buf, index=False)
            api.upload_file(path_or_fileobj=buf.getvalue(), path_in_repo=path, repo_id=DS_REPO,
                            repo_type="dataset", commit_message=f"LLM mention check: {path}")
            changed = sum(sorted(a) != list(b) for a, b in zip(df["labels"], df["verified_labels"]))
            print(f"[{split} {k + 1}/{n_chunks}] {len(df)} rows, changed {changed}, "
                  f"rejected {sum(map(len, df['judge_rejected']))}, added {sum(map(len, df['judge_added']))}, "
                  f"errors {sum(bool(e) for e in df['judge_error'])}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
