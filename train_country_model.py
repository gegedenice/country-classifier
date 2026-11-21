# /// script
# requires-python = ">=3.10"
# dependencies = ["torch>=2.3", "transformers>=4.48", "datasets>=3.0", "accelerate>=0.33",
#                 "numpy", "huggingface_hub>=0.30", "pandas", "pyarrow"]
# ///
"""Fine-tune a multilingual encoder (default jhu-clsp/mmBERT-base) as a 247-label multi-label
"country mention" classifier, evaluate it and push it to the Hub.

One definition of the target everywhere: the labels are the rule labels checked by the LLM
(column verified_labels, files verified/*.parquet written by judge_labels.py). If those files
do not exist, the raw rule labels are used. Training, validation and test use the same labels.

Decision rule: a country is predicted when its probability is >= THRESHOLD (default 0.5, the
same for every country). CALIBRATE=1 instead picks one threshold per country on the validation
split (the one maximising F1); it is optional and off by default.

Run on HF Jobs (GPU), from a clone of the repository:
  hf jobs uv run --flavor a10g-small --timeout 6h --secrets HF_TOKEN \
     -e DS_REPO=Geraldine/pub-country-classification-dataset-v2.1 \
     -e MODEL_REPO=Geraldine/pub-country-classifier-v2.1 train_country_model.py

Local smoke test (CPU, a few steps, no push):  SMOKE=1 DS_LOCAL=build python train_country_model.py
"""
import json
import os
import random
import tempfile

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

SMOKE = os.environ.get("SMOKE") == "1"
TOK = os.environ.get("HF_TOKEN")
DS_REPO = os.environ.get("DS_REPO", "Geraldine/pub-country-classification-dataset-v2.1")
MODEL_REPO = os.environ.get("MODEL_REPO", "Geraldine/pub-country-classifier-v2.1")
PRIVATE = os.environ.get("MODEL_PRIVATE", "1") == "1"
BASE = os.environ.get("BASE_MODEL", "jhu-clsp/mmBERT-base")
MAXLEN = int(os.environ.get("MAXLEN", "512"))
EPOCHS = float(os.environ.get("EPOCHS", "2"))
LR = float(os.environ.get("LR", "3e-5"))
THRESHOLD = float(os.environ.get("THRESHOLD", "0.5"))
CALIBRATE = os.environ.get("CALIBRATE") == "1"
DS_LOCAL = os.environ.get("DS_LOCAL")   # directory with train/validation/test.jsonl (smoke tests)
LABELS_DIR = os.environ.get("LABELS_DIR", "verified")   # folder of LLM-checked splits in the dataset repo
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def ds_file(name):
    if DS_LOCAL:
        return os.path.join(DS_LOCAL, name)
    return hf_hub_download(DS_REPO, name, repo_type="dataset", token=TOK)


codes = json.load(open(ds_file("codes.json")))
N = len(codes)
idx = {c: i for i, c in enumerate(codes)}

# ------------------------------------------------------------------ data
SPLITS = ("train", "validation", "test")
if DS_LOCAL:
    ds = load_dataset("json", data_files={s: os.path.join(DS_LOCAL, f"{s}.jsonl") for s in SPLITS})
    LABEL_COL = "labels"
else:
    files = set(HfApi(token=TOK).list_repo_files(DS_REPO, repo_type="dataset"))
    if all(any(f.startswith(f"{LABELS_DIR}/{s}-") for f in files) for s in SPLITS):
        ds = load_dataset(DS_REPO, data_files={s: f"{LABELS_DIR}/{s}-*.parquet" for s in SPLITS}, token=TOK)
        LABEL_COL = "verified_labels"
    else:
        ds = load_dataset(DS_REPO, token=TOK)
        LABEL_COL = "labels"
print(f"[init] {N} labels, base={BASE}, labels from column '{LABEL_COL}' ({LABELS_DIR if LABEL_COL != 'labels' else 'rules'})", flush=True)

if SMOKE:
    ds["train"] = ds["train"].shuffle(seed=0).select(range(256))
    ds["validation"] = ds["validation"].select(range(128))
    ds["test"] = ds["test"].select(range(128))
    MAXLEN, EPOCHS = 128, 1

tok = AutoTokenizer.from_pretrained(BASE)


def multi_hot(labs):
    y = np.zeros(N, dtype=np.float32)
    for c in labs:
        y[idx[c]] = 1.0
    return y


def prep(batch):
    enc = tok(batch["text"], truncation=True, max_length=MAXLEN)
    enc["labels"] = [multi_hot(l).tolist() for l in batch[LABEL_COL]]
    return enc


def tokenize(split, num_proc=None):
    d = ds[split]
    return d.map(prep, batched=True, num_proc=num_proc, remove_columns=d.column_names)


train = tokenize("train", None if SMOKE else 8)
val_tok = tokenize("validation")


class Collator:
    def __call__(self, feats):
        batch = tok.pad([{k: f[k] for k in ("input_ids", "attention_mask")} for f in feats],
                        padding=True, return_tensors="pt")
        batch["labels"] = torch.tensor([f["labels"] for f in feats], dtype=torch.float32)
        return batch


# ------------------------------------------------------------------ training
model = AutoModelForSequenceClassification.from_pretrained(
    BASE, num_labels=N, problem_type="multi_label_classification",
    id2label={i: c for i, c in enumerate(codes)}, label2id=idx)
gpu = torch.cuda.is_available()
total_steps = int(np.ceil(len(train) / 32) * EPOCHS)
args = TrainingArguments(
    output_dir=tempfile.mkdtemp(), per_device_train_batch_size=16, per_device_eval_batch_size=64,
    gradient_accumulation_steps=2, num_train_epochs=EPOCHS, learning_rate=LR, weight_decay=0.01,
    warmup_steps=max(1, int(0.05 * total_steps)), lr_scheduler_type="linear", max_grad_norm=1.0,
    bf16=gpu, seed=SEED, logging_steps=10 if SMOKE else 200, eval_strategy="epoch", save_strategy="no",
    report_to=[], dataloader_num_workers=0 if SMOKE else 4, remove_unused_columns=False,
    max_steps=8 if SMOKE else -1)
trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=val_tok, data_collator=Collator())
trainer.train()
model.eval()


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def probs_of(texts, bs=64):
    P = np.zeros((len(texts), N), dtype=np.float32)
    order = np.argsort([len(t) for t in texts])          # length-sorted batches = faster
    for i in range(0, len(texts), bs):
        sel = order[i:i + bs]
        enc = tok([texts[j] for j in sel], truncation=True, max_length=MAXLEN, padding=True,
                  return_tensors="pt").to(model.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=gpu):
            P[sel] = torch.sigmoid(model(**enc).logits.float()).cpu().numpy()
    return P


def Y_of(label_lists):
    return np.stack([multi_hot(l) for l in label_lists])


def f1(tp, fp, fn):
    return 2 * tp / max(1e-9, 2 * tp + fp + fn)


def metrics(P, Y, T, mask=None):
    if mask is not None:
        P, Y = P[mask], Y[mask]
    if len(Y) == 0:
        return {"n": 0}
    pred = (P >= T[None, :]).astype(np.float32)
    tp, fp, fn = (pred * Y).sum(), (pred * (1 - Y)).sum(), ((1 - pred) * Y).sum()
    tp_l, fp_l, fn_l = (pred * Y).sum(0), (pred * (1 - Y)).sum(0), ((1 - pred) * Y).sum(0)
    has = Y.sum(0) > 0
    pos_rows, neg_rows = Y.sum(1) > 0, Y.sum(1) == 0
    top = P.argmax(1)
    return {
        "n": int(len(Y)),
        "micro_precision": round(float(tp / max(1, tp + fp)), 4),
        "micro_recall": round(float(tp / max(1, tp + fn)), 4),
        "micro_f1": round(float(f1(tp, fp, fn)), 4),
        "macro_f1": round(float(np.mean([f1(a, b, c) for a, b, c in zip(tp_l[has], fp_l[has], fn_l[has])])), 4) if has.any() else None,
        "exact_match": round(float((pred == Y).all(1).mean()), 4),
        "precision_at_top1": round(float(Y[np.arange(len(Y)), top][pos_rows].mean()), 4) if pos_rows.any() else None,
        "false_positive_rate_on_negatives": round(float((pred[neg_rows].sum(1) > 0).mean()), 4) if neg_rows.any() else None,
        "n_negative_rows": int(neg_rows.sum()),
    }


thr = {c: THRESHOLD for c in codes}
if CALIBRATE:
    Pv, Yv = probs_of(ds["validation"]["text"]), Y_of(ds["validation"][LABEL_COL])
    grid = np.arange(0.05, 0.951, 0.025)
    for j, c in enumerate(codes):
        y = Yv[:, j]
        if y.sum() >= 3:
            thr[c] = round(float(max(grid, key=lambda t: (f1(((Pv[:, j] >= t) & (y == 1)).sum(),
                                                               ((Pv[:, j] >= t) & (y == 0)).sum(),
                                                               ((Pv[:, j] < t) & (y == 1)).sum()), -abs(t - 0.5)))), 3)
T = np.array([thr[c] for c in codes])

test = ds["test"]
Pt, Yt = probs_of(test["text"]), Y_of(test[LABEL_COL])
langs, variants = np.array(test["language"]), np.array(test["variant"])
report = {
    "base_model": BASE, "labels": LABEL_COL, "labels_dir": LABELS_DIR,
    "decision": "per-country thresholds calibrated on validation" if CALIBRATE else f"single threshold {THRESHOLD}",
    "hyperparameters": {"epochs": EPOCHS, "lr": LR, "batch_effective": 32, "max_len": MAXLEN,
                        "warmup": "5% of steps", "seed": SEED},
    "test": metrics(Pt, Yt, T),
    "test_by_language": {l: metrics(Pt, Yt, T, langs == l) for l in ["en", "fr", "es", "de", "pt"]},
    "test_by_variant": {v: metrics(Pt, Yt, T, variants == v) for v in ["full", "title_only"]},
}
print(json.dumps(report, indent=1), flush=True)

pred_t = Pt >= T[None, :]
counts = json.load(open(ds_file("counts.json")))
lexicon = json.load(open(ds_file("country_lexicon.json")))
per_label = {}
for j, c in enumerate(codes):
    tp = int((pred_t[:, j] & (Yt[:, j] == 1)).sum()); fp = int((pred_t[:, j] & (Yt[:, j] == 0)).sum())
    fn = int((~pred_t[:, j] & (Yt[:, j] == 1)).sum())
    per_label[c] = {"name": lexicon["names"][c], "f1": round(float(f1(tp, fp, fn)), 4), "tp": tp, "fp": fp, "fn": fn,
                    "n_test": int(Yt[:, j].sum()), "n_train_rules": counts.get(c, 0), "threshold": thr[c]}

if SMOKE:
    print("[smoke] OK, nothing pushed", flush=True)
    raise SystemExit(0)

# ------------------------------------------------------------------ release
api = HfApi(token=TOK)
api.create_repo(MODEL_REPO, private=PRIVATE, exist_ok=True)
model.push_to_hub(MODEL_REPO, token=TOK)
tok.push_to_hub(MODEL_REPO, token=TOK)
m = report["test"]
card = [
    "---", "language: [en, fr, es, de, pt]", "license: apache-2.0", f"base_model: {BASE}",
    f"datasets: [{DS_REPO}]", "pipeline_tag: text-classification",
    "tags: [multi-label-classification, country-detection, geographic-indexing, scholarly-metadata, openalex]", "---", "",
    f"# Publication Country Classifier ({MODEL_REPO.split('/')[-1]})",
    "",
    f"Given a scholarly publication (title, optional abstract) in English, French, Spanish, German or Portuguese, "
    f"returns the countries it **mentions** among the {N} OpenAlex countries (ISO 3166-1 alpha-2, plus XK). "
    "A mention is an explicit name, variant or demonym of the country, in its geographic or political sense "
    "(not a homonym such as *Anguilla anguilla*, *New Jersey* or *Jordan curve*).",
    f"Fine-tuned from {BASE}. Training labels: a multilingual rule labeler (OpenAlex + Wikidata lexicon), "
    "with every label checked by an LLM (Qwen3-235B) for wrong or missed mentions.",
    "", "Code and methodology (in French): https://github.com/gegedenice/country-classifier", "",
    "## Usage", "", "```python",
    "import json, torch", "from huggingface_hub import hf_hub_download",
    "from transformers import AutoTokenizer, AutoModelForSequenceClassification",
    f"repo = '{MODEL_REPO}'", "tok = AutoTokenizer.from_pretrained(repo)",
    "model = AutoModelForSequenceClassification.from_pretrained(repo).eval()",
    "codes = json.load(open(hf_hub_download(repo, 'codes.json')))",
    "thr = json.load(open(hf_hub_download(repo, 'thresholds.json')))",
    "text = 'La gestion de l’eau dans le delta du Mékong au Vietnam'",
    "with torch.no_grad():",
    "    p = torch.sigmoid(model(**tok(text, truncation=True, max_length=512, return_tensors='pt')).logits)[0]",
    "print([(c, round(float(x), 3)) for c, x in zip(codes, p) if x >= thr[c]])", "```", "",
    f"## Evaluation (test split, {report['decision']})", "",
    "| | precision | recall | F1 micro | F1 macro | P@top1 | FP rate on no-country rows | n |",
    "|---|---|---|---|---|---|---|---|",
    f"| all | {m['micro_precision']} | {m['micro_recall']} | {m['micro_f1']} | {m['macro_f1']} | "
    f"{m['precision_at_top1']} | {m['false_positive_rate_on_negatives']} | {m['n']} |",
]
for l, x in report["test_by_language"].items():
    card.append(f"| {l} | {x.get('micro_precision')} | {x.get('micro_recall')} | {x.get('micro_f1')} | {x.get('macro_f1')} | "
                f"{x.get('precision_at_top1')} | {x.get('false_positive_rate_on_negatives')} | {x.get('n')} |")
card += ["", "## Limitations", "",
         "- Weak supervision: rule labels checked by an LLM, not by human annotators.",
         "- Detects explicit mentions only: a country named only through a city or region is not detected.",
         "- Trained on en/fr/es/de/pt; inputs truncated to 512 tokens.",
         "- A text classifier: do not use it to infer authors' nationality or for decisions about people."]
files = {"codes.json": json.dumps(codes), "thresholds.json": json.dumps(thr, indent=1),
         "metrics.json": json.dumps(report, indent=1), "per_label_metrics.json": json.dumps(per_label, indent=1),
         "README.md": "\n".join(card) + "\n"}
for name, content in files.items():
    api.upload_file(path_or_fileobj=content.encode(), path_in_repo=name, repo_id=MODEL_REPO, commit_message=f"add {name}")
print(f"[done] pushed {MODEL_REPO} (private={PRIVATE})", flush=True)
