# inference_example.py
# Minimal CLI inference for Geraldine/pub-country-classifier.
# Deps: torch transformers huggingface_hub
# Usage:
#   python inference_example.py --title "..." --abstract "..."
#   python inference_example.py "Full text with title on first line"
import argparse, json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import huggingface_hub

REPO = "Geraldine/pub-country-classifier"


def main():
    ap = argparse.ArgumentParser(description="Detect countries mentioned/about in a publication")
    ap.add_argument("text", nargs="?", help="publication text (title, then abstract on line 2)")
    ap.add_argument("--title", help="title")
    ap.add_argument("--abstract", help="abstract")
    ap.add_argument("--top", type=int, default=10, help="how many detections to print")
    a = ap.parse_args()

    text = a.text or (a.title or "")
    if a.abstract:
        text = (text + chr(10) + a.abstract).strip()
    if not text.strip():
        ap.error("provide a text, or --title/--abstract")

    tok = AutoTokenizer.from_pretrained(REPO)
    model = AutoModelForSequenceClassification.from_pretrained(REPO)
    codes = json.load(huggingface_hub.hf_hub_download(REPO, "codes.json"))
    thr = json.load(huggingface_hub.hf_hub_download(REPO, "thresholds.json"))

    enc = tok(text, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        probs = torch.sigmoid(model(**enc).logits)[0]

    detections = sorted(
        [(codes[i], round(float(probs[i]), 3)) for i in range(len(codes)) if float(probs[i]) >= thr[codes[i]]],
        key=lambda x: -x[1])
    print(json.dumps(detections[: a.top], indent=2))
    if not detections:
        print("No country detected above calibrated thresholds.")


if __name__ == "__main__":
    main()
