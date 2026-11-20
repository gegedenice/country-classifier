# train_country_model.py
# Fine-tunes answerdotai/ModernBERT-base as a 249-label multi-label country classifier,
# calibrates per-label thresholds on validation, evaluates on test, pushes everything to the Hub.
# Env: HF_TOKEN.  Deps: torch transformers datasets accelerate numpy huggingface_hub.  GPU >= 16GB.
import os, json, random, re, tempfile
import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download, HfApi
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments)

TOK = os.environ["HF_TOKEN"]
DS_REPO = "Geraldine/pub-country-classification-dataset"
MODEL_REPO = "Geraldine/pub-country-classifier"
BASE = "answerdotai/ModernBERT-base"
MAXLEN = 512

random.seed(42); np.random.seed(42); torch.manual_seed(42)

codes_path = hf_hub_download(DS_REPO, "codes.json", repo_type="dataset", token=TOK)
codes = json.load(open(codes_path))
N = len(codes)
code2idx = {c: i for i, c in enumerate(codes)}
print(f"[init] {N} labels", flush=True)

ds = load_dataset(DS_REPO)
tok = AutoTokenizer.from_pretrained(BASE)

def tok_fn(batch):
    return tok(batch["text"], truncation=True, max_length=MAXLEN)

def lab_fn(batch):
    out = []
    for labs in batch["labels"]:
        y = [0.0] * N
        for c in labs:
            y[code2idx[c]] = 1.0
        out.append(y)
    return {"labels": out}

tr = ds["train"].map(tok_fn, batched=True, num_proc=8).map(lab_fn, batched=True, num_proc=8)

class Collator:
    def __call__(self, feats):
        batch = tok.pad([{k: f[k] for k in ("input_ids", "attention_mask")} for f in feats],
                        padding=True, return_tensors="pt")
        batch["labels"] = torch.tensor([f["labels"] for f in feats], dtype=torch.float32)
        return batch

model = AutoModelForSequenceClassification.from_pretrained(
    BASE, num_labels=N, problem_type="multi_label_classification")

# TrainingArguments: some kwargs may be unsupported in the installed transformers version;
# drop them dynamically instead of crashing.
base_kwargs = dict(
    output_dir="/tmp/ckpt", per_device_train_batch_size=16, gradient_accumulation_steps=2,
    num_train_epochs=2, learning_rate=2e-5, weight_decay=0.01,
    bf16=True, logging_steps=200, save_strategy="no", report_to=[], seed=42,
    dataloader_num_workers=4, remove_unused_columns=False, max_grad_norm=1.0)
for _ in range(5):
    try:
        args = TrainingArguments(**base_kwargs)
        break
    except TypeError as e:
        dropped = re.findall(r"keyword argument '(\w+)'", str(e))
        if not dropped:
            raise
        for k in dropped:
            base_kwargs.pop(k, None)
        print(f"[args] dropped unsupported kwargs: {dropped}", flush=True)

trainer = Trainer(model=model, args=args, train_dataset=tr, data_collator=Collator())
trainer.train()
model.eval()

@torch.no_grad()
def get_probs(texts, bs=64):
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i+bs], truncation=True, max_length=MAXLEN, padding=True,
                  return_tensors="pt").to(model.device)
        out.append(torch.sigmoid(model(**enc).logits.float()).cpu().numpy())
    return np.vstack(out)

def to_Y(split):
    Y = np.zeros((len(split), N), dtype=np.float32)
    for i, labs in enumerate(split["labels"]):
        for c in labs:
            Y[i, code2idx[c]] = 1.0
    return Y

# threshold calibration on validation
val, test = ds["validation"], ds["test"]
Pv, Yv = get_probs(val["text"]), to_Y(val)
thr = {}
for j, c in enumerate(codes):
    y = Yv[:, j]
    if y.sum() == 0:
        thr[c] = 0.5
        continue
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.951, 0.025):
        pred = (Pv[:, j] >= t).astype(np.float32)
        tp = float(((pred == 1) & (y == 1)).sum())
        fp = float(((pred == 1) & (y == 0)).sum())
        fn = float(((pred == 0) & (y == 1)).sum())
        f1 = 2 * tp / max(1e-9, 2 * tp + fp + fn)
        if f1 > best_f1:
            best_f1, best_t = f1, round(float(t), 3)
    thr[c] = best_t

# test evaluation with calibrated thresholds
T = np.array([thr[c] for c in codes])
Pt, Yt = get_probs(test["text"]), to_Y(test)
pred = (Pt >= T[None, :]).astype(np.float32)
tp = float((pred * Yt).sum()); fp = float((pred * (1 - Yt)).sum()); fn = float(((1 - pred) * Yt).sum())
micro_f1 = 2 * tp / max(1e-9, 2 * tp + fp + fn)
per_f1 = 2 * (pred * Yt).sum(axis=0) / np.maximum(1e-9, 2 * (pred * Yt).sum(axis=0) + (pred * (1 - Yt)).sum(axis=0) + ((1 - pred) * Yt).sum(axis=0))
pos_labels = Yt.sum(axis=0) > 0
macro_f1 = float(per_f1[pos_labels].mean())
rows_pos = Yt.sum(axis=1) > 0
top1 = float(((Pt.argmax(axis=1) == Yt.argmax(axis=1)) & rows_pos).sum() / max(1, rows_pos.sum()))
prec_at_top1 = float(((Yt[np.arange(len(Yt)), Pt.argmax(axis=1)] == 1) & rows_pos).sum() / max(1, rows_pos.sum()))
metrics = {"micro_f1": round(micro_f1, 4), "macro_f1_over_labels_with_positives": round(macro_f1, 4),
           "top1_accuracy_on_labelled_rows": round(top1, 4), "precision_at_top1": round(prec_at_top1, 4),
           "n_test_rows": int(len(test)), "labels_without_positives_in_test": int((~pos_labels).sum())}
print(f"[eval] {metrics}", flush=True)

counts = json.load(open(hf_hub_download(DS_REPO, "counts.json", repo_type="dataset", token=TOK)))
per_label = {c: {"f1": round(float(per_f1[i]), 4), "n_test": int(Yt[:, i].sum()),
                 "n_train": counts.get(c, 0), "threshold": thr[c]} for i, c in enumerate(codes)}

model.config.id2label = {str(i): c for i, c in enumerate(codes)}
model.config.label2id = {c: i for i, c in enumerate(codes)}
model.push_to_hub(MODEL_REPO, token=TOK)
tok.push_to_hub(MODEL_REPO, token=TOK)

api = HfApi(token=TOK)
for name, content in {
    "codes.json": json.dumps(codes),
    "thresholds.json": json.dumps(thr),
    "metrics.json": json.dumps(metrics, indent=2),
    "per_label_metrics.json": json.dumps(per_label, indent=2),
}.items():
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write(content)
        p = f.name
    api.upload_file(path_or_fileobj=p, path_in_repo=name, repo_id=MODEL_REPO, commit_message=f"add {name}")

card = []
card.append("---")
card.append("language: en")
card.append("license: apache-2.0")
card.append("base_model: answerdotai/ModernBERT-base")
card.append("tags:")
card.append("- multi-label-classification")
card.append("- country-detection")
card.append("- scholarly-metadata")
card.append("- openalex")
card.append("---")
card.append("")
card.append("# Publication Country Classifier (pub-country-classifier)")
card.append("")
card.append(f"Detects whether a publication (title + optional abstract) is about one of the {N} ISO 3166-1 alpha-2 country codes, via a multi-label sigmoid head over ModernBERT-base.")
card.append("")
card.append("Input: title on line 1, abstract on line 2 (title-only works too). Output: per-country probability; keep codes above the calibrated thresholds in thresholds.json.")
card.append("")
card.append("## Training data")
card.append("")
card.append("Weakly supervised from [Geraldine/pub-country-classification-dataset](https://huggingface.co/datasets/Geraldine/pub-country-classification-dataset): OpenAlex works whose title/abstract explicitly mention a country (names/demonyms), hard negatives with ambiguous tokens (Georgia, Congo, ...), plus clean topical negatives. 15% of rows are title-only.")
card.append("")
card.append("## Evaluation (test split, calibrated thresholds)")
card.append("")
card.append("| metric | value |")
card.append("|---|---|")
for k, v in metrics.items():
    card.append(f"| {k} | {v} |")
card.append("")
card.append("## Limitations")
card.append("")
card.append("- Labels are **mention-based** (weak supervision), not human-adjudicated aboutness.")
card.append("- English-only corpus; sparse labels have low reliability.")
card.append("- Inputs longer than 512 tokens are truncated.")
with tempfile.NamedTemporaryFile("w", delete=False) as f:
    f.write("\n".join(card) + "\n")
    p = f.name
api.upload_file(path_or_fileobj=p, path_in_repo="README.md", repo_id=MODEL_REPO, commit_message="model card")
print("[done] all artifacts pushed", flush=True)
