"""Second LLM step (preparation and merge) and publication of the final dataset.

    python finalize_labels.py prepare   # list the tags rejected by step 1, with their trigger words
                                        #   -> recheck/input-NNN.parquet (input of recheck_tags.py)
    python finalize_labels.py merge     # apply the step-2 verdicts -> checked/{split}-00000.parquet
    python finalize_labels.py publish   # final dataset, clean columns -> FINAL_REPO

Label logic, for each row:
    removed = tags rejected by step 1 (judge_labels.py) AND confirmed by step 2 (recheck_tags.py)
              (a tag whose step-2 call failed stays removed)
    added   = countries explicitly named but missed by the rules (step 1, quote checked in the text)
    labels  = (rule labels - removed) + added

Env: HF_TOKEN, BUILD_REPO (working dataset), FINAL_REPO (published dataset), FINAL_PRIVATE (1/0).
"""
import io
import json
import sys
from collections import Counter
from pathlib import Path
import os

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

BUILD_REPO = os.environ.get("BUILD_REPO", "Geraldine/pub-country-build")
FINAL_REPO = os.environ.get("FINAL_REPO", "Geraldine/pub-country")
FINAL_PRIVATE = os.environ.get("FINAL_PRIVATE", "1") == "1"
TOKEN = os.environ.get("HF_TOKEN")
SPLITS = ("train", "validation", "test")
api = HfApi(token=TOKEN)


def upload_parquet(df, path, msg):
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    api.upload_file(path_or_fileobj=buf.getvalue(), path_in_repo=path, repo_id=BUILD_REPO,
                    repo_type="dataset", commit_message=msg)


def step1():
    return load_dataset(BUILD_REPO, data_files={s: f"verified/{s}-*.parquet" for s in SPLITS}, token=TOKEN)


def prepare(chunk=1700):
    from countries.labeler import Labeler
    lab = Labeler()
    items = []
    for split, d in step1().items():
        for r in d.filter(lambda r: len(r["judge_rejected"]) > 0):
            ex = lab.explain(r["text"])
            for c in r["judge_rejected"]:
                items.append({"item_id": f"{split}|{r['id']}|{r['variant']}|{c}", "text": r["text"], "code": c,
                              "words": sorted({x["span"] for x in ex if x["code"] == c})})
    df = pd.DataFrame(items)
    for k in range(0, len(df), chunk):
        upload_parquet(df.iloc[k:k + chunk], f"recheck/input-{k // chunk:03d}.parquet", "step-2 input")
    print(f"[prepare] {len(df)} tags in {(len(df) + chunk - 1) // chunk} files")


def merge():
    verdicts = {}
    for f in api.list_repo_files(BUILD_REPO, repo_type="dataset"):
        if f.startswith("recheck/output-"):
            df = pd.read_parquet(hf_hub_download(BUILD_REPO, f, repo_type="dataset", token=TOKEN))
            verdicts.update({r.item_id: (r.verdict, r.error) for r in df.itertuples()})
    print(f"[merge] {len(verdicts)} step-2 verdicts")
    for split, d in step1().items():
        df = d.to_pandas()
        removed = []
        for rej, wid, var in zip(df.judge_rejected, df.id, df.variant):
            keep = {c for c in rej if verdicts.get(f"{split}|{wid}|{var}|{c}", ("", "x")) [0] == "keep"
                    and not verdicts[f"{split}|{wid}|{var}|{c}"][1]}
            removed.append(sorted(set(rej) - keep))
        df["llm_removed"] = removed
        df["final_labels"] = [sorted((set(l) - set(r)) | set(a)) for l, r, a in zip(df.labels, removed, df.judge_added)]
        upload_parquet(df, f"checked/{split}-00000.parquet", f"checked/{split}: step-1 and step-2 labels")
        print(f"[merge] {split}: removed {sum(map(len, removed))}, added {sum(map(len, df.judge_added))}")


def publish():
    ds = load_dataset(BUILD_REPO, data_files={s: f"checked/{s}-*.parquet" for s in SPLITS}, token=TOKEN)
    out = {}
    for split, d in ds.items():
        df = d.to_pandas()
        out[split] = Dataset.from_pandas(pd.DataFrame({
            "id": df.id, "text": df.text, "labels": df.final_labels, "rule_labels": df.labels,
            "llm_removed": df.llm_removed, "llm_added": df.judge_added, "llm_added_quotes": df.judge_quotes,
            "language": df.language, "year": df.year, "type": df.type, "origin": df.origin,
            "variant": df.variant, "has_abstract": df.has_abstract, "has_trap": df.has_trap,
        }), preserve_index=False)
    DatasetDict(out).push_to_hub(FINAL_REPO, private=FINAL_PRIVATE, token=TOKEN)
    codes = json.load(open(hf_hub_download(BUILD_REPO, "codes.json", repo_type="dataset", token=TOKEN)))
    counts = Counter(c for l in out["train"]["labels"] for c in l)
    files = {"codes.json": json.dumps(codes), "counts.json": json.dumps({c: counts[c] for c in codes}),
             "country_lexicon.json": open(hf_hub_download(BUILD_REPO, "country_lexicon.json", repo_type="dataset",
                                                          token=TOKEN)).read()}
    for name, content in files.items():
        api.upload_file(path_or_fileobj=content.encode(), path_in_repo=name, repo_id=FINAL_REPO, repo_type="dataset")
    print(f"[publish] {FINAL_REPO}: " + ", ".join(f"{s} {len(d)}" for s, d in out.items()))


if __name__ == "__main__":
    {"prepare": prepare, "merge": merge, "publish": publish}[sys.argv[1]]()
