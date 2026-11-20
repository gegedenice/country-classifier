"""Minimal CLI inference for the publication country classifier.

Usage:
  python inference_example.py --title "..." [--abstract "..."] [--repo Geraldine/pub-country-classifier-v2]
  python inference_example.py "Title on the first line\nAbstract on the second"
"""
import argparse
import json

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    ap = argparse.ArgumentParser(description="Detect the countries a publication is about")
    ap.add_argument("text", nargs="?", help="publication text (title, then abstract on line 2)")
    ap.add_argument("--title")
    ap.add_argument("--abstract")
    ap.add_argument("--repo", default="Geraldine/pub-country-classifier-v2")
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args()

    text = a.text or (a.title or "")
    if a.abstract:
        text = (text + "\n" + a.abstract).strip()
    if not text.strip():
        ap.error("provide a text, or --title/--abstract")

    tok = AutoTokenizer.from_pretrained(a.repo)
    model = AutoModelForSequenceClassification.from_pretrained(a.repo).eval()
    with open(hf_hub_download(a.repo, "codes.json")) as f:
        codes = json.load(f)
    with open(hf_hub_download(a.repo, "thresholds.json")) as f:
        thr = json.load(f)

    enc = tok(text, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        probs = torch.sigmoid(model(**enc).logits)[0]

    found = sorted(((c, round(float(p), 3)) for c, p in zip(codes, probs) if float(p) >= thr[c]),
                   key=lambda x: -x[1])
    print(json.dumps(found[: a.top], indent=2))
    if not found:
        print("No country detected above the calibrated thresholds.")


if __name__ == "__main__":
    main()
