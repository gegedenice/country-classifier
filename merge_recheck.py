"""Merge the second-pass verdicts (recheck_tags.py, prompt mention-check-3) into the labels.

Only the tags REJECTED by the first pass (judge_labels.py) are rechecked: the first pass's
KEEP decisions were all right in a blind review, its rejections were wrong about 40 % of the
time. For each row of verified/{split}:

    judge_rejected_final = first-pass rejections confirmed by the recheck (or recheck failed)
    recheck_restored     = first-pass rejections overturned by the recheck
    verified_labels      = (labels - judge_rejected_final) + judge_added

The result is written to verified2/{split}-00000.parquet (train with LABELS_DIR=verified2).

Inputs: the recheck verdicts, either as recheck/output-*.parquet files in the dataset repo, or
as a local JSON file {item_id: {"verdict": "keep"|"reject", "error": ""}} (VERDICTS=path),
item_id = "{split}|{work id}|{variant}|{code}".

Usage:  HF_TOKEN=... python merge_recheck.py
"""
import io
import json
import os

import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download

DS_REPO = os.environ.get("DS_REPO", "Geraldine/pub-country-classification-dataset-v2.1")
TOKEN = os.environ.get("HF_TOKEN")
api = HfApi(token=TOKEN)

if os.environ.get("VERDICTS"):
    verdicts = json.load(open(os.environ["VERDICTS"]))
else:
    verdicts = {}
    for f in api.list_repo_files(DS_REPO, repo_type="dataset"):
        if f.startswith("recheck/output-"):
            df = pd.read_parquet(hf_hub_download(DS_REPO, f, repo_type="dataset", token=TOKEN))
            verdicts.update({r.item_id: {"verdict": r.verdict, "error": r.error} for r in df.itertuples()})
print(f"[merge] {len(verdicts)} verdicts")

ds = load_dataset(DS_REPO, data_files={s: f"verified/{s}-*.parquet" for s in ("train", "validation", "test")}, token=TOKEN)
for split in ds:
    df = ds[split].to_pandas()
    restored, final = [], []
    for rej, wid, var in zip(df.judge_rejected, df.id, df.variant):
        keep = [c for c in rej if (v := verdicts.get(f"{split}|{wid}|{var}|{c}")) and v["verdict"] == "keep" and not v["error"]]
        restored.append(sorted(keep))
        final.append(sorted(set(rej) - set(keep)))
    df["recheck_restored"] = restored
    df["judge_rejected_final"] = final
    df["verified_labels_v1"] = df["verified_labels"]
    df["verified_labels"] = [sorted((set(l) - set(f)) | set(a)) for l, f, a in zip(df.labels, final, df.judge_added)]
    df["recheck_prompt"] = "mention-check-3"
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    api.upload_file(path_or_fileobj=buf.getvalue(), path_in_repo=f"verified2/{split}-00000.parquet", repo_id=DS_REPO,
                    repo_type="dataset", commit_message=f"verified2/{split}: rejections rechecked (mention-check-3)")
    print(f"[merge] {split}: restored {sum(map(len, restored))} of {sum(map(len, df.judge_rejected))} rejections")
