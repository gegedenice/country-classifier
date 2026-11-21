"""Build the working dataset (rule labels) from OpenAlex.

Pipeline
  1. For each of the 247 OpenAlex countries, fetch works whose title/abstract mention it:
     - English works: search on the English name (up to MAX_EN works);
     - fr/es/de/pt works: one boolean search on the country's names in those languages
       (up to MAX_OTHER works).
  2. Add a random sample of works (OpenAlex `sample`), in the same languages, so that the
     dataset also reflects what a real bibliographic database looks like: most works name
     no country at all. These give realistic negatives across every discipline.
  3. Label every work with the rule labeler (countries/labeler.py). Works with no country
     are KEPT (empty label list): those coming from a country query are hard negatives
     ("Jordan curve", "New Jersey", "American Journal of ...").
  4. Duplicate TITLE_ONLY_FRAC of the works that have an abstract as title-only rows,
     relabelled from the title alone.
  5. Split 90/5/5 by a hash of the OpenAlex id (stable, a work and its title-only copy
     always share a split) and push to the Hub with codes.json, counts.json, the lexicon
     and build_stats.json.

Raw API pages are cached in CACHE_DIR, so an interrupted run resumes without spending
the OpenAlex budget again.

Env: OPENALEX_API_KEY (sent as Bearer header), HF_TOKEN (push), DS_REPO (default below),
     DS_PRIVATE=1|0 (default 1), CACHE_DIR (default .cache/openalex), PUSH=0 to skip upload.
"""
import hashlib
import re
import json
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from countries.labeler import Labeler  # noqa: E402

API = "https://api.openalex.org/works"
KEY = os.environ.get("OPENALEX_API_KEY", "")
DS_REPO = os.environ.get("DS_REPO", "Geraldine/pub-country-mentions-build")
DS_PRIVATE = os.environ.get("DS_PRIVATE", "1") == "1"
CACHE = Path(os.environ.get("CACHE_DIR", ROOT / ".cache" / "openalex"))
OUT_DIR = Path(os.environ.get("OUT_DIR", ROOT / "build"))
HEADERS = {"User-Agent": "country-classifier-build/2.0 (https://github.com/gegedenice/country-classifier)"}
if KEY:
    HEADERS["Authorization"] = f"Bearer {KEY}"

LANGS_OTHER = ["fr", "es", "de", "pt"]
TYPES = "article|book-chapter|preprint|dissertation|book|review|report"
SELECT = "id,title,abstract_inverted_index,language,publication_year,type"
MAX_EN = 400            # works per country, English
MAX_OTHER = 200         # works per country, fr/es/de/pt together
RANDOM_MIXED = 24000    # random works, all 5 languages (mostly English, as in OpenAlex)
RANDOM_PER_LANG = 4000  # extra random works per non-English language
TITLE_ONLY_FRAC = 0.15
MAX_LABELS = 12         # more countries than this = a list/menu page, not a publication
SEED = 42
random.seed(SEED)

lab = Labeler()
lex = json.loads((ROOT / "data" / "country_lexicon.json").read_text())
CODES = lex["codes"]


# ---------------------------------------------------------------- OpenAlex access
def get(params):
    """GET with on-disk cache and retries. Returns the JSON body."""
    key = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()
    path = CACHE / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    for attempt in range(8):
        try:
            r = requests.get(API, params=params, headers=HEADERS, timeout=90)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(60, 2 ** attempt))
                continue
            if r.status_code == 403 and "budget" in r.text.lower():
                raise SystemExit("[openalex] daily budget exhausted: rerun tomorrow, cached pages are kept")
            r.raise_for_status()
            j = r.json()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(j))
            return j
        except requests.RequestException:
            if attempt == 7:
                raise
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"failed: {params}")


def fetch_search(terms, langs, max_works):
    """Works whose title/abstract match any of `terms`, in `langs`."""
    clean = dict.fromkeys(re.sub(r"\s+", " ", re.sub(r"[,\"()|:]", " ", t)).strip() for t in terms)
    q = " OR ".join(f'"{t}"' for t in clean if len(t) > 2)
    filt = f"title_and_abstract.search:{q},language:{'|'.join(langs)},type:{TYPES}"
    out, page = [], 1
    while len(out) < max_works:
        j = get({"filter": filt, "per-page": 200, "page": page, "select": SELECT})
        res = j.get("results", [])
        out += res
        if len(res) < 200:
            break
        page += 1
    return out[:max_works]


def fetch_sample(n, seed, langs):
    filt = f"language:{'|'.join(langs)},type:{TYPES}"
    out = []
    for page in range(1, n // 200 + 1):
        j = get({"sample": n, "seed": seed, "filter": filt, "per-page": 200, "page": page, "select": SELECT})
        out += j.get("results", [])
    return out


def inv2txt(inv):
    if not inv:
        return ""
    pos = sorted((i, w) for w, idxs in inv.items() for i in idxs)
    return " ".join(w for _, w in pos)


def search_terms(code):
    """Query terms: English (display name + a few manual variants) and fr/es/de/pt names.

    Queries only decide which works are fetched; labels always come from the labeler,
    so a broad or ambiguous query (e.g. "Georgia") simply brings in hard negatives.
    """
    al = lex["aliases"][code]
    name = lex["names"][code]
    manual_ascii = [a["text"] for a in al if a["source"] == "manual" and a["text"].isascii()
                    and len(a["text"]) > 3 and a["text"] != name]
    en = [name] + manual_ascii[:3]
    names = {a["text"] for a in al if a["case_sensitive"] and not a["source"].startswith("wikidata_demonym")
             and not a["source"].startswith("wikidata_label_en") and a["text"] not in en and len(a["text"]) > 3}
    other = sorted(names, key=lambda t: (len(t), t))[:8] or en
    return en, other


def label_and_trap(text):
    ex = lab.explain(text)
    return sorted({d["code"] for d in ex if d["code"]}), any(d["code"] is None for d in ex)


# ---------------------------------------------------------------- main
def main():
    works = {}   # id -> record

    def ingest(ws, origin):
        for w in ws:
            title = (w.get("title") or "").strip()
            if len(title) < 8:
                continue
            if w["id"] not in works:
                works[w["id"]] = {"id": w["id"].rsplit("/", 1)[-1], "title": title,
                                  "abstract": inv2txt(w.get("abstract_inverted_index")).strip(),
                                  "language": w.get("language"), "year": w.get("publication_year"),
                                  "type": w.get("type"), "origin": origin}

    def job(code):
        en, other = search_terms(code)
        return code, fetch_search(en, ["en"], MAX_EN), fetch_search(other, LANGS_OTHER, MAX_OTHER)

    with ThreadPoolExecutor(6) as ex:
        for n, (code, ws_en, ws_other) in enumerate(ex.map(job, CODES), 1):
            ingest(ws_en, f"country:{code}")
            ingest(ws_other, f"country:{code}")
            if n % 25 == 0:
                print(f"[fetch] {n}/{len(CODES)} countries, {len(works)} works", flush=True)

    for seed in range(4):
        ingest(fetch_sample(RANDOM_MIXED // 4, 1000 + seed, ["en"] + LANGS_OTHER), "random")
    for i, lang in enumerate(LANGS_OTHER):
        ingest(fetch_sample(RANDOM_PER_LANG, 2000 + i, [lang]), "random")
    print(f"[fetch] total {len(works)} unique works", flush=True)

    rows, junk = [], 0
    for w in works.values():
        text = w["title"] + ("\n" + w["abstract"] if w["abstract"] else "")
        labels, trap = label_and_trap(text)
        if len(labels) > MAX_LABELS:
            junk += 1
            continue
        h = int(hashlib.md5(w["id"].encode()).hexdigest(), 16) % 100
        split = "test" if h < 5 else "validation" if h < 10 else "train"
        base = {"id": w["id"], "language": w["language"], "year": w["year"], "type": w["type"],
                "origin": "random" if w["origin"] == "random" else "country_query", "split": split}
        rows.append({**base, "text": text, "labels": labels, "has_abstract": bool(w["abstract"]),
                     "variant": "full", "has_trap": trap})
        if w["abstract"] and random.random() < TITLE_ONLY_FRAC:
            t_labels, t_trap = label_and_trap(w["title"])
            rows.append({**base, "text": w["title"], "labels": t_labels,
                         "has_abstract": False, "variant": "title_only", "has_trap": t_trap})
    print(f"[label] {len(rows)} rows, {junk} junk works dropped (> {MAX_LABELS} countries)", flush=True)

    splits = {s: [r for r in rows if r["split"] == s] for s in ("train", "validation", "test")}
    for s in splits.values():
        random.shuffle(s)
        for r in s:
            del r["split"]
    counts = Counter(c for r in splits["train"] for c in r["labels"])
    stats = {
        "rows": {s: len(v) for s, v in splits.items()},
        "works": len(works), "junk_dropped": junk,
        "negatives_share": {s: round(sum(not r["labels"] for r in v) / max(1, len(v)), 4) for s, v in splits.items()},
        "languages_train": Counter(r["language"] for r in splits["train"]).most_common(),
        "origin_train": Counter(r["origin"] for r in splits["train"]).most_common(),
        "codes_with_zero_train": [c for c in CODES if counts[c] == 0],
        "codes_under_50_train": {c: counts[c] for c in CODES if 0 < counts[c] < 50},
    }
    print(json.dumps(stats, indent=1), flush=True)

    OUT_DIR.mkdir(exist_ok=True)
    for s, v in splits.items():
        with open(OUT_DIR / f"{s}.jsonl", "w") as f:
            for r in v:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (OUT_DIR / "codes.json").write_text(json.dumps(CODES))
    (OUT_DIR / "counts.json").write_text(json.dumps({c: counts[c] for c in CODES}))
    (OUT_DIR / "build_stats.json").write_text(json.dumps(stats, indent=1))

    if os.environ.get("PUSH", "1") == "1":
        from datasets import load_dataset
        from huggingface_hub import HfApi
        ds = load_dataset("json", data_files={s: str(OUT_DIR / f"{s}.jsonl") for s in splits})
        ds.push_to_hub(DS_REPO, private=DS_PRIVATE, token=os.environ.get("HF_TOKEN"))
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        for name, path in [("codes.json", OUT_DIR / "codes.json"), ("counts.json", OUT_DIR / "counts.json"),
                           ("build_stats.json", OUT_DIR / "build_stats.json"),
                           ("country_lexicon.json", ROOT / "data" / "country_lexicon.json")]:
            api.upload_file(path_or_fileobj=str(path), path_in_repo=name, repo_id=DS_REPO, repo_type="dataset")
        print(f"[push] {DS_REPO} (private={DS_PRIVATE})", flush=True)


if __name__ == "__main__":
    main()
