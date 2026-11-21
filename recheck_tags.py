# /// script
# requires-python = ">=3.10"
# dependencies = ["openai>=1.40", "huggingface_hub>=0.30", "pandas", "pyarrow"]
# ///
"""Second-pass LLM check of individual country tags (prompt "mention-check-2").

Why a second prompt: a blind review of 120 decisions of the first pass (judge_labels.py,
prompt mention-check-1) showed that its KEEP decisions were all right, but about 40 % of its
REJECT decisions were wrong. The first prompt had three gaps:
  1. it spoke of "countries", so dependent territories (Martinique, Norfolk Island...) were
     rejected as "not a country";
  2. the LLM was not told which word had triggered the tag ("GB" for "Écosse");
  3. nothing was said about historical or colonial forms ("British raj", "Belgian Congo"),
     and passing mentions were still dropped as "irrelevant".
Prompt v2 fixes these: one verdict per tag, the triggering words are given, territories and
historical forms are explicitly in scope, and a short reason is required.

Input:  a parquet file in the dataset repo with columns
        item_id, text, code, words (list of the words that triggered the tag)
Output: the same rows + verdict ("keep" / "reject"), reason, error, model, prompt.

Run on HF Jobs (CPU), from a clone of the repository:
  hf jobs uv run --flavor cpu-basic --timeout 2h --secrets HF_TOKEN \
     -e DS_REPO=Geraldine/pub-country-classification-dataset-v2.1 \
     -e INPUT=eval/judge_sample.parquet -e OUTPUT=eval/judge_sample_v2.parquet recheck_tags.py
"""
import asyncio
import io
import json
import os
import re

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from openai import AsyncOpenAI

DS_REPO = os.environ.get("DS_REPO", "Geraldine/pub-country-classification-dataset-v2.1")
MODEL = os.environ.get("JUDGE_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507")
INPUT = os.environ.get("INPUT", "eval/judge_sample.parquet")
OUTPUT = os.environ.get("OUTPUT", "eval/judge_sample_v2.parquet")
CONC = int(os.environ.get("JUDGE_CONCURRENCY", "32"))
TOKEN = os.environ["HF_TOKEN"]
PROMPT_VERSION = "mention-check-2"

api = HfApi(token=TOKEN)
lex = json.load(open(hf_hub_download(DS_REPO, "country_lexicon.json", repo_type="dataset", token=TOKEN)))
NAMES = lex["names"]

SYSTEM = """You check automatic geographic tags on scholarly publications.

Each tag is a code from the ISO 3166-1 list (plus XK = Kosovo). The list contains sovereign states
AND dependent territories: Réunion (RE), Puerto Rico (PR), Guam (GU), Aruba (AW), Hong Kong (HK),
Gibraltar (GI)... A territory is a valid tag in its own right. Never reject a tag because the
place "is not a country".

For each tag you are given the word(s) of the text that triggered it. Decide whether, in this
text, those words refer to that country or territory, in any of these forms:
- its name in any language, an official or former name, an abbreviation ("Écosse" -> GB,
  "Burma" -> MM, "USA" -> US, "Zaïre" -> CD);
- a demonym or adjective for its people, state, culture or products ("Italian migrants",
  "Germans living in Brazil", "British rule in Kenya", "French Algeria", "Spanish Florida");
- a historical or colonial state of that country ("British India", "Soviet Russia",
  "Ottoman Turkey"). An ancient kingdom or empire that is not a former form of the modern state
  does not count ("Ancient Rome" is not Italy).

KEEP the tag when the words refer to the country, even in a passing mention, a list, a comparison,
a reference, a case name or an example. Relevance does not matter.

REJECT the tag only when the words mean something else:
- a species or scientific name ("Anguilla anguilla", "Coffea arabica");
- a person's name ("Jordan" as a surname, "Florence Chad");
- another place with the same name: a city, US state, county, parish, district, street, sea, gulf,
  channel or ocean ("Georgia, USA", "Salvador, Bahia", "Gulf of Mexico", "Persian Gulf");
- the name of a journal, society or company that merely contains the word ("Journal of Chinese
  Studies", "American Chemical Society");
- a language used as a language ("translated from Italian", "Portuguese grammar");
- a common word or other homonym ("chile peppers", "jersey cattle", "turkey meat").

Answer with JSON only:
{"tags": [{"code": "XX", "verdict": "keep" or "reject", "reason": "a few words"}]}"""


def user_prompt(text, code, words):
    w = ", ".join(f'"{x}"' for x in words) or "(not given)"
    return (f"Text:\n<<<\n{text[:4000]}\n>>>\n\n"
            f"Tag: {code} ({NAMES.get(code, code)}), triggered by: {w}")


def parse(raw, code):
    j = json.loads(re.search(r"\{.*\}", raw or "", re.S).group(0))
    for t in j.get("tags", []):
        if t.get("code") == code and t.get("verdict") in ("keep", "reject"):
            return t["verdict"], (t.get("reason") or "")[:200]
    raise ValueError(f"no verdict for {code}: {raw[:200]}")


async def check(client, sem, text, code, words):
    async with sem:
        err = ""
        for attempt in range(5):
            try:
                r = await client.chat.completions.create(
                    model=MODEL, temperature=0, max_tokens=200,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": user_prompt(text, code, list(words))}])
                v, why = parse(r.choices[0].message.content, code)
                return v, why, ""
            except Exception as e:  # noqa: BLE001
                err = repr(e)[:300]
                await asyncio.sleep(2 ** attempt)
        return "", "", err


async def main():
    client = AsyncOpenAI(base_url="https://router.huggingface.co/v1", api_key=TOKEN)
    await client.chat.completions.create(model=MODEL, max_tokens=5, messages=[{"role": "user", "content": "ping"}])
    df = pd.read_parquet(hf_hub_download(DS_REPO, INPUT, repo_type="dataset", token=TOKEN))
    print(f"[recheck] {len(df)} tags from {INPUT} with {MODEL}", flush=True)
    sem = asyncio.Semaphore(CONC)
    res = []
    for i in range(0, len(df), 2000):
        part = df.iloc[i:i + 2000]
        res += await asyncio.gather(*[check(client, sem, t, c, w)
                                      for t, c, w in zip(part["text"], part["code"], part["words"])])
        print(f"  {len(res)}/{len(df)}", flush=True)
    df["verdict"] = [r[0] for r in res]
    df["reason"] = [r[1] for r in res]
    df["error"] = [r[2] for r in res]
    df["model"] = MODEL
    df["prompt"] = PROMPT_VERSION
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    try:
        api.upload_file(path_or_fileobj=buf.getvalue(), path_in_repo=OUTPUT, repo_id=DS_REPO, repo_type="dataset",
                        commit_message=f"LLM tag recheck ({PROMPT_VERSION}): {OUTPUT}")
    except Exception as e:  # noqa: BLE001  (e.g. a read-only token): keep the results in the logs
        print(f"[upload failed] {e!r}"[:300], flush=True)
        for row in df[["item_id", "code", "verdict", "reason", "error"]].to_dict("records"):
            print("RESULT " + json.dumps(row, ensure_ascii=False), flush=True)
    print(f"[done] {OUTPUT}: keep {sum(df.verdict == 'keep')}, reject {sum(df.verdict == 'reject')}, "
          f"errors {sum(df.error != '')}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
