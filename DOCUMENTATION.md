# Publication Country Classifier — Code & Documentation

This bundle contains the full source code and documentation for the **publication-country detection** pipeline: a small decision model that reads a scholarly publication (title + abstract, or title alone) and decides which of the 249 ISO 3166-1 alpha-2 countries the text is about.

Repos:
- Dataset: https://huggingface.co/datasets/Geraldine/pub-country-classification-dataset (CC0-1.0)
- Model: https://huggingface.co/Geraldine/pub-country-classifier (Apache-2.0)

---

## 1. Goal

Given the text of a publication, answer: **is this publication talking about country X?** for every X in the ISO 3166-1 standard set. This is a 249-way *multi-label* decision problem: one publication can mention several countries, or none. Typical uses: enriching bibliometric corpora with geographic focus, filtering literature by study geography, routing abstracts.

## 2. Nature of the model

- **Architecture**: answerdotai/ModernBERT-base (~149M parameters) with a linear classification head (768 to 249) and a sigmoid per output — independent per-country probabilities (problem_type=multi_label_classification, BCEWithLogitsLoss).
- **Input contract**: line 1 = title, line 2 = abstract (a title-only string also works). Truncated to 512 wordpiece tokens.
- **Output contract**: 249 probabilities. A country is *detected* when p >= threshold[code] (per-label thresholds in thresholds.json, calibrated on the validation split).
- **Artifacts in the model repo**: codes.json (index to alpha-2 mapping), thresholds.json, metrics.json, per_label_metrics.json, config.json (id2label/label2id set), model card (README.md).

## 3. Pipeline logic

OpenAlex API -> per-country relevance fetch -> alias labeling + ban rules -> hard negatives (ambiguous tokens) -> clean topical negatives -> 90/5/5 split -> ModernBERT fine-tune (BCE) -> per-label threshold calibration (validation) -> test evaluation -> Hub push.

- build_country_dataset.py implements the first half (data).
- train_country_model.py implements the second half (training, calibration, evaluation, release).

## 4. Data building — build_country_dataset.py

Weak supervision recipe (no human annotation):

1. **Fetching**: for each of the 249 codes, query OpenAlex /works with title_and_abstract.search:"<country term>" (English only), cursor-paginated at 200/page, up to 800 works per country, with retry/backoff and an API key. Abstracts are reconstructed from abstract_inverted_index.
2. **Positive labeling** — a code is attached when the title/abstract matches any alias: names (common/official via pycountry), abbreviations (USA, UK, UAE, DRC, DPRK, East Timor, Ivory Coast, ...), demonyms (French, Kenyan, Taiwanese, Saudi, ...).
3. **Ban rules** to stop obvious false positives: prefix bans (New Mexico does not label MX, North Korea does not label KR, South Sudan does not label SD, Guinea pig does not label GN); postfix bans (American Samoa does not label US, French horn does not label FR, British Virgin Islands does not label VG); the bare word Georgia labels GE only with regional context (Tbilisi, Abkhazia, Batumi, ...).
4. **Hard negatives**: works containing ambiguous tokens (Congo, Macedonia, Georgia, New Guinea, Congolese, guinea worm) are excluded from the negative pool entirely.
5. **Clean negatives**: works from ~30 topical queries (deep learning, clinical trials, batteries, ...) that contain no alias of any code and no ambiguous token -> empty label list.
6. **Title-only robustness**: 15% of abstract rows are duplicated as title-only rows (source=openalex_titleonly).
7. **Splits**: 90/5/5 by OpenAlex work id (all variants of a work stay in the same split). Pushed as a DatasetDict; codes.json (canonical label order) and counts.json (per-code train counts) uploaded alongside.

Result: 190.8K rows (train 171.8K / val 9.5K / test 9.5K), 245/249 codes with >=25 training examples; zero-count codes: FK, MF, SX, VA; sparsest: CD 269, SH 394, GS 574, ST 599.

## 5. Training — train_country_model.py

- Tokenize with truncation to 512; labels are multi-hot vectors in canonical codes.json order.
- Fine-tune ModernBERT-base with the BCE multi-label head.
- Hyperparameters: batch 16 x grad-accum 2 (effective 32), lr 2e-5, 2 epochs, weight decay 0.01, bf16, linear LR schedule, grad-norm clip 1.0, seed 42.
- **Threshold calibration**: after training, sigmoid probabilities on the validation split; for each label pick the cut in 0.05..0.95 (step 0.025) maximizing F1 (0.5 default for labels with no validation positives).
- **Evaluation** on the test split with calibrated thresholds (micro/macro F1) plus top-1 accuracy and precision@top-1 on rows carrying >=1 positive label.
- **Release**: push model + tokenizer with id2label/label2id, then codes.json, thresholds.json, metrics.json, per_label_metrics.json, and a generated model card.
- Note: TrainingArguments kwargs are filtered dynamically (unsupported args in the installed transformers version are dropped with a log line instead of crashing).

## 6. Results (test split, calibrated thresholds)

| metric | value |
|---|---|
| micro_f1 | 0.9572 |
| macro_f1_over_labels_with_positives | 0.9565 |
| top1_accuracy_on_labelled_rows | 0.8680 |
| precision_at_top1 | 0.9927 |
| n_test_rows | 9528 |
| labels_without_positives_in_test | 4 |
| frac_test_rows_exceeding_512_tokens_sampled | 0.0377 |

Reading: mention-detection is largely lexical, so F1 is high and uniform across labels (macro ~ micro). Top-1 accuracy is lower mostly on multi-country publications (argmax picks one of several true labels). Precision at argmax is near-perfect (0.993): when the model is most confident, it is almost always right. Per-label details: per_label_metrics.json in the model repo.

## 7. How to run

```bash
pip install -r requirements_build.txt
export OPENALEX_API_KEY=...   # https://openalex.org
export HF_TOKEN=...
python build_country_dataset.py        # ~30-60 min, modest API budget
```

```bash
pip install -r requirements_train.txt
export HF_TOKEN=...
python train_country_model.py          # needs a >=16GB GPU; ~1-2 h on A10G
```

Both scripts are the exact ones run on HF Jobs (uv runtime, a10g-small flavor for training).

## 8. Inference

```bash
python inference_example.py --title "Water quality governance in rural India" --abstract "This study examines village-level water policy in India ..."
```

or in Python: see the model card on the Hub.

## 9. Limitations & responsible-use notes

- Labels are **mention-based** (text explicitly names/derives the country). The model detects that a text *talks about* a country; it is not a human-adjudicated aboutness judgment, and implicit references without the country name can be missed.
- English-only training corpus; non-English titles/abstracts are out of distribution.
- Inputs longer than 512 tokens are truncated (~3.8% of test rows).
- Sparse labels: tiny territories (FK, MF, SX, VA zero-count; CD, SH, GS, ST, ... under 1000) will have lower reliability.
- Do **not** use for attributing authors' nationality/affiliation, or any decision with legal/sanctions consequences about individuals or organizations — this is a text-topics classifier, nothing more.

## 10. Bundle inventory

| file | role |
|---|---|
| DOCUMENTATION.md | this document |
| build_country_dataset.py | dataset builder (OpenAlex -> weak labels) |
| train_country_model.py | training, calibration, evaluation, release |
| inference_example.py | minimal inference CLI |
| requirements_build.txt / requirements_train.txt | dependencies |
