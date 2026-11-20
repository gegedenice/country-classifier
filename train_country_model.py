# /// script
# requires-python = ">=3.10"
# dependencies = ["torch>=2.3", "transformers>=4.48", "datasets>=3.0", "accelerate>=0.33",
#                 "numpy", "huggingface_hub>=0.30"]
# ///
"""Fine-tune a multilingual encoder (default jhu-clsp/mmBERT-base) as a 247-label multi-label
country classifier, calibrate one threshold per country, evaluate and push to the Hub.

Reference labels for calibration/evaluation:
  * if the dataset has a "judged" config (judge_labels.py), thresholds are calibrated on the
    judged validation split and the test split is evaluated against BOTH the judge labels
    (main result: "is the publication about this country?") and the rule labels;
  * otherwise, rule labels are used everywhere.

Run on HF Jobs (GPU):
  hf jobs uv run --flavor a10g-small --timeout 6h --secrets HF_TOKEN \
     -e DS_REPO=Geraldine/pub-country-classification-dataset-v2 \
     -e MODEL_REPO=Geraldine/pub-country-classifier-v2 \
     train_country_model.py (from a clone of the repository)

Local smoke test (CPU, a few steps, no push):  SMOKE=1 python train_country_model.py
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
DS_REPO = os.environ.get("DS_REPO", "Geraldine/pub-country-classification-dataset-v2")
MODEL_REPO = os.environ.get("MODEL_REPO", "Geraldine/pub-country-classifier-v2")
PRIVATE = os.environ.get("MODEL_PRIVATE", "1") == "1"
BASE = os.environ.get("BASE_MODEL", "jhu-clsp/mmBERT-base")
MAXLEN = int(os.environ.get("MAXLEN", "512"))
EPOCHS = float(os.environ.get("EPOCHS", "2"))
LR = float(os.environ.get("LR", "3e-5"))
SEED = 42

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DS_LOCAL = os.environ.get("DS_LOCAL")   # directory with train/validation/test.jsonl (smoke tests)


def ds_file(name):
    if DS_LOCAL:
        return os.path.join(DS_LOCAL, name)
    return hf_hub_download(DS_REPO, name, repo_type="dataset", token=TOK)


codes = json.load(open(ds_file("codes.json")))
N = len(codes)
idx = {c: i for i, c in enumerate(codes)}
print(f"[init] {N} labels, base={BASE}", flush=True)

if DS_LOCAL:
    ds = load_dataset("json", data_files={s: os.path.join(DS_LOCAL, f"{s}.jsonl") for s in ("train", "validation", "test")})
else:
    ds = load_dataset(DS_REPO, token=TOK)
try:
    if DS_LOCAL:
        raise FileNotFoundError
    judged = load_dataset(DS_REPO, "judged", token=TOK)
    print("[init] judged config found: judge labels used for calibration and evaluation", flush=True)
except Exception:  # noqa: BLE001
    judged = None
    print("[init] no judged config: rule labels used everywhere", flush=True)

if SMOKE:
    ds["train"] = ds["train"].shuffle(seed=0).select(range(256))
    ds["validation"] = ds["validation"].select(range(128))
    ds["test"] = ds["test"].select(range(128))
    judged = None
    MAXLEN, EPOCHS = 128, 1

tok = AutoTokenizer.from_pretrained(BASE)


def multi_hot(labs):
    y = np.zeros(N, dtype=np.float32)
    for c in labs:
        y[idx[c]] = 1.0
    return y


def prep(batch):
    enc = tok(batch["text"], truncation=True, max_length=MAXLEN)
    enc["labels"] = [multi_hot(l).tolist() for l in batch["labels"]]
    return enc


keep_cols = ["input_ids", "attention_mask", "labels"]
train = ds["train"].map(prep, batched=True, num_proc=None if SMOKE else 8,
                        remove_columns=[c for c in ds["train"].column_names if c not in keep_cols])
val_tok = ds["validation"].map(prep, batched=True,
                               remove_columns=[c for c in ds["validation"].column_names if c not in keep_cols])


class Collator:
    def __call__(self, feats):
        batch = tok.pad([{k: f[k] for k in ("input_ids", "attention_mask")} for f in feats],
                        padding=True, return_tensors="pt")
        batch["labels"] = torch.tensor([f["labels"] for f in feats], dtype=torch.float32)
        return batch


model = AutoModelForSequenceClassification.from_pretrained(
    BASE, num_labels=N, problem_type="multi_label_classification",
    id2label={i: c for i, c in enumerate(codes)}, label2id=idx)

gpu = torch.cuda.is_available()
total_steps = int(np.ceil(len(train) / 32) * EPOCHS)
args = TrainingArguments(
    output_dir=tempfile.mkdtemp(), per_device_train_batch_size=16, per_device_eval_batch_size=64,
    gradient_accumulation_steps=2, num_train_epochs=EPOCHS, learning_rate=LR, weight_decay=0.01,
    warmup_steps=max(1, int(0.05 * total_steps)), lr_scheduler_type="linear", max_grad_norm=1.0, bf16=gpu, seed=SEED,
    logging_steps=10 if SMOKE else 200, eval_strategy="epoch", save_strategy="no", report_to=[],
    dataloader_num_workers=0 if SMOKE else 4, remove_unused_columns=False,
    max_steps=8 if SMOKE else -1)
trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=val_tok, data_collator=Collator())
trainer.train()
model.eval()


@torch.no_grad()
def probs_of(texts, bs=64):
    out = []
    order = np.argsort([len(t) for t in texts])          # length-sorted batches = faster
    for i in range(0, len(texts), bs):
        sel = order[i:i + bs]
        enc = tok([texts[j] for j in sel], truncation=True, max_length=MAXLEN, padding=True,
                  return_tensors="pt").to(model.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=gpu):
            p = torch.sigmoid(model(**enc).logits.float()).cpu().numpy()
        out.append((sel, p))
    P = np.zeros((len(texts), N), dtype=np.float32)
    for sel, p in out:
        P[sel] = p
    return P


def Y_of(label_lists):
    return np.stack([multi_hot(l) for l in label_lists]) if len(label_lists) else np.zeros((0, N))


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
    pos_rows = Y.sum(1) > 0
    neg_rows = ~pos_rows
    top = P.argmax(1)
    return {
        "n": int(len(Y)),
        "micro_precision": round(float(tp / max(1, tp + fp)), 4),
        "micro_recall": round(float(tp / max(1, tp + fn)), 4),
        "micro_f1": round(float(f1(tp, fp, fn)), 4),
        "macro_f1": round(float(np.mean([f1(a, b, c) for a, b, c in zip(tp_l[has], fp_l[has], fn_l[has])])), 4) if has.any() else None,
        "exact_match": round(float((pred == Y).all(1).mean()), 4),
        # among rows with >= 1 true country: is the most probable country one of them?
        "precision_at_top1": round(float(Y[np.arange(len(Y)), top][pos_rows].mean()), 4) if pos_rows.any() else None,
        # among rows with no country: share where the model still predicts something
        "false_positive_rate_on_negatives": round(float((pred[neg_rows].sum(1) > 0).mean()), 4) if neg_rows.any() else None,
        "n_negative_rows": int(neg_rows.sum()),
    }


def calibrate(P, Y):
    thr = {}
    grid = np.arange(0.05, 0.951, 0.025)
    for j, c in enumerate(codes):
        y = Y[:, j]
        if y.sum() < 3:                      # too few positives to tune: fixed, prudent cut
            thr[c] = 0.5
            continue
        best = max(grid, key=lambda t: (f1(((P[:, j] >= t) & (y == 1)).sum(), ((P[:, j] >= t) & (y == 0)).sum(),
                                           ((P[:, j] < t) & (y == 1)).sum()), t))
        thr[c] = round(float(best), 3)
    return thr


ref_val = judged["validation"] if judged else ds["validation"]
ref_key = "judge_labels" if judged else "labels"
Pv = probs_of(ref_val["text"])
thr = calibrate(Pv, Y_of(ref_val[ref_key]))
T = np.array([thr[c] for c in codes])

report = {"base_model": BASE, "reference_for_thresholds": ref_key, "hyperparameters": {
    "epochs": EPOCHS, "lr": LR, "batch_effective": 32, "max_len": MAXLEN, "warmup_ratio": 0.05, "seed": SEED}}
test = ds["test"]
Pt = probs_of(test["text"])
Yr = Y_of(test["labels"])
report["test_vs_rules"] = metrics(Pt, Yr, T)
langs = np.array(test["language"])
variants = np.array(test["variant"])
report["test_vs_rules_by_language"] = {l: metrics(Pt, Yr, T, langs == l) for l in ["en", "fr", "es", "de", "pt"]}
report["test_vs_rules_by_variant"] = {v: metrics(Pt, Yr, T, variants == v) for v in ["full", "title_only"]}
if judged:
    jt = judged["test"]
    Pj = probs_of(jt["text"])
    Yj = Y_of(jt["judge_labels"])
    report["test_vs_judge"] = metrics(Pj, Yj, T)
    report["rules_vs_judge_on_test"] = metrics(Y_of(jt["labels"]), Yj, np.full(N, 0.5))
    jl = np.array(jt["language"])
    report["test_vs_judge_by_language"] = {l: metrics(Pj, Yj, T, jl == l) for l in ["en", "fr", "es", "de", "pt"]}
print(json.dumps(report, indent=1), flush=True)

pred_t = (Pt >= T[None, :])
per_label = {}
counts = json.load(open(ds_file("counts.json")))
for j, c in enumerate(codes):
    tp, fp, fn = (pred_t[:, j] & (Yr[:, j] == 1)).sum(), (pred_t[:, j] & (Yr[:, j] == 0)).sum(), (~pred_t[:, j] & (Yr[:, j] == 1)).sum()
    per_label[c] = {"name": None, "f1_vs_rules": round(float(f1(tp, fp, fn)), 4), "n_test": int(Yr[:, j].sum()),
                    "n_train": counts.get(c, 0), "threshold": thr[c]}

if SMOKE:
    print("[smoke] OK, nothing pushed", flush=True)
    raise SystemExit(0)

api = HfApi(token=TOK)
api.create_repo(MODEL_REPO, private=PRIVATE, exist_ok=True)
model.push_to_hub(MODEL_REPO, token=TOK)
tok.push_to_hub(MODEL_REPO, token=TOK)
lexicon = json.load(open(ds_file("country_lexicon.json")))
for c in per_label:
    per_label[c]["name"] = lexicon["names"][c]
files = {"codes.json": json.dumps(codes), "thresholds.json": json.dumps(thr, indent=1),
         "metrics.json": json.dumps(report, indent=1), "per_label_metrics.json": json.dumps(per_label, indent=1)}


def row(name, m):
    return (f"| {name} | {m.get('micro_precision')} | {m.get('micro_recall')} | {m.get('micro_f1')} | "
            f"{m.get('macro_f1')} | {m.get('precision_at_top1')} | {m.get('false_positive_rate_on_negatives')} | {m.get('n')} |")


card = [
    "---", "language: [en, fr, es, de, pt]", "license: apache-2.0", f"base_model: {BASE}",
    f"datasets: [{DS_REPO}]", "pipeline_tag: text-classification",
    "tags: [multi-label-classification, country-detection, geographic-indexing, scholarly-metadata, openalex]", "---", "",
    "# Publication Country Classifier v2",
    "",
    f"Multi-label classifier: given a scholarly publication (title, optional abstract) in English, French, Spanish, "
    f"German or Portuguese, returns which of the {N} OpenAlex countries (ISO 3166-1 alpha-2, plus XK) it is about.",
    "Fine-tuned from " + BASE + " on weakly-supervised labels (multilingual lexicon from OpenAlex + Wikidata, "
    "with blockers for traps such as *New Jersey*, *Gulf of Mexico* or *Jordan curve*), with one decision threshold per country.",
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
    "## Evaluation (test split, calibrated thresholds)", "",
    "| reference | micro P | micro R | micro F1 | macro F1 | P@top1 | FP rate on no-country rows | n |",
    "|---|---|---|---|---|---|---|---|",
    row("rule labels", report["test_vs_rules"]),
]
if judged:
    card.append(row("LLM-judge labels", report["test_vs_judge"]))
    card.append(row("(rules themselves vs judge)", report["rules_vs_judge_on_test"]))
card += ["", "By language (vs rule labels):", "", "| lang | micro F1 | FP rate on negatives | n |", "|---|---|---|---|"]
for l, m in report["test_vs_rules_by_language"].items():
    card.append(f"| {l} | {m.get('micro_f1')} | {m.get('false_positive_rate_on_negatives')} | {m.get('n')} |")
card += ["", "## Limitations", "",
         "- Labels are produced by rules (and checked by an LLM judge on validation/test only when available), not by human annotators.",
         "- Only en/fr/es/de/pt were trained on. Countries are mostly detected when named (names, demonyms); "
         "inference from cities or regions alone is limited.",
         "- Inputs are truncated to 512 tokens.",
         "- A text-topic classifier: do not use it to infer authors' nationality or for decisions about people."]
files["README.md"] = "\n".join(card) + "\n"
for name, content in files.items():
    api.upload_file(path_or_fileobj=content.encode(), path_in_repo=name, repo_id=MODEL_REPO, commit_message=f"add {name}")
print(f"[done] pushed {MODEL_REPO} (private={PRIVATE})", flush=True)
