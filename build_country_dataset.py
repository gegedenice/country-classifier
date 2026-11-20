# build_country_dataset.py
# Builds Geraldine/pub-country-classification-dataset from the OpenAlex API.
# Env: HF_TOKEN (dataset upload), OPENALEX_API_KEY (OpenAlex).
# Deps: requests pycountry datasets huggingface_hub
import os, re, json, time, random
import requests
import pycountry
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

HF_TOKEN = os.environ["HF_TOKEN"]
API_KEY = os.environ["OPENALEX_API_KEY"]
DS_REPO = "Geraldine/pub-country-classification-dataset"
MAILTO = "geraldine+openalex@huggingface.co"
API = "https://api.openalex.org/works"
HEADERS = {"User-Agent": "pub-country-classifier-build/1.0 (mailto:geraldine+openalex@huggingface.co)"}
PER_PAGE = 200
MAX_WORKS = 800
random.seed(42)

countries = list(pycountry.countries)
codes = sorted(c.alpha_2 for c in countries)
by_code = {c.alpha_2: c for c in countries}
print(f"[init] {len(codes)} ISO 3166-1 alpha-2 codes", flush=True)

r = requests.get(API, params={"per-page": 1, "mailto": MAILTO, "api_key": API_KEY}, headers=HEADERS, timeout=30)
print(f"[conn] status={r.status_code}", flush=True)
r.raise_for_status()

EXTRA = {
 "US": [r"\bU\.S\.(?=$|[\s,;:])", r"\bU\.S\.A\.(?=$|[\s,;:])", r"\bUSA\b"],
 "GB": [r"\bUK\b", r"\bBritain\b", r"\bGreat Britain\b"],
 "VN": [r"\bViet ?Nam\b"], "AE": [r"\bUAE\b"], "CD": [r"\bDRC\b", r"\bDR Congo\b"],
 "KP": [r"\bDPRK\b"], "TW": [r"\bFormosa\b"], "MM": [r"\bBurma\b"], "CI": [r"\bIvory Coast\b"],
 "CV": [r"\bCape Verde\b"], "CZ": [r"\bCzech Republic\b"], "TL": [r"\bEast Timor\b"],
 "SZ": [r"\bSwaziland\b"], "IR": [r"\bPersia\b"], "NL": [r"\bHolland\b"],
 "SS": [r"\bSouthern Sudan\b"], "BZ": [r"\bBritish Honduras\b"], "VI": [r"U\.S\. Virgin Islands"],
}

DEMONYMS = {
 "US": ["American"], "GB": ["British"], "FR": ["French"], "DE": ["German"], "IT": ["Italian"],
 "ES": ["Spanish"], "PT": ["Portuguese"], "NL": ["Dutch"], "BE": ["Belgian"], "CH": ["Swiss"],
 "AT": ["Austrian"], "SE": ["Swedish"], "NO": ["Norwegian"], "DK": ["Danish"], "FI": ["Finnish"],
 "GR": ["Greek"], "PL": ["Polish"], "CZ": ["Czech"], "SK": ["Slovak"], "SI": ["Slovenian", "Slovene"],
 "HU": ["Hungarian"], "RO": ["Romanian"], "UA": ["Ukrainian"], "BY": ["Belarusian"], "RU": ["Russian"],
 "RS": ["Serbian"], "HR": ["Croatian"], "BA": ["Bosnian"], "AL": ["Albanian"], "BG": ["Bulgarian"],
 "TR": ["Turkish"], "IL": ["Israeli"], "PS": ["Palestinian"], "SA": ["Saudi"], "AE": ["Emirati"],
 "QA": ["Qatari"], "KW": ["Kuwaiti"], "BH": ["Bahraini"], "OM": ["Omani"], "JO": ["Jordanian"],
 "LB": ["Lebanese"], "SY": ["Syrian"], "IQ": ["Iraqi"], "IR": ["Iranian"], "AF": ["Afghan"],
 "PK": ["Pakistani"], "IN": ["Indian"], "NP": ["Nepalese", "Nepali"], "BD": ["Bangladeshi"],
 "LK": ["Sri Lankan"], "MM": ["Burmese"], "TH": ["Thai"], "VN": ["Vietnamese"], "KH": ["Cambodian"],
 "LA": ["Laotian"], "MY": ["Malaysian"], "SG": ["Singaporean"], "ID": ["Indonesian"], "PH": ["Filipino"],
 "CN": ["Chinese"], "TW": ["Taiwanese"], "JP": ["Japanese"], "KR": ["Korean"], "KP": ["North Korean"],
 "AU": ["Australian"], "NZ": ["New Zealander"], "CA": ["Canadian"], "MX": ["Mexican"],
 "BR": ["Brazilian"], "AR": ["Argentine"], "CL": ["Chilean"], "CO": ["Colombian"], "PE": ["Peruvian"],
 "VE": ["Venezuelan"], "EC": ["Ecuadorian"], "BO": ["Bolivian"], "PY": ["Paraguayan"], "UY": ["Uruguayan"],
 "ZA": ["South African"], "NG": ["Nigerian"], "NE": ["Nigerien"], "EG": ["Egyptian"], "ET": ["Ethiopian"],
 "KE": ["Kenyan"], "TZ": ["Tanzanian"], "UG": ["Ugandan"], "GH": ["Ghanaian"], "CM": ["Cameroonian"],
 "SN": ["Senegalese"], "ML": ["Malian"], "TG": ["Togolese"], "BJ": ["Beninese"], "BF": ["Burkinabe"],
 "CI": ["Ivorian"], "MA": ["Moroccan"], "DZ": ["Algerian"], "TN": ["Tunisian"], "LY": ["Libyan"],
 "AO": ["Angolan"], "MZ": ["Mozambican"], "ZM": ["Zambian"], "ZW": ["Zimbabwean"], "BW": ["Botswanan"],
 "NA": ["Namibian"], "RW": ["Rwandan"], "MU": ["Mauritian"], "MG": ["Malagasy"],
 "KZ": ["Kazakh", "Kazakhstani"], "UZ": ["Uzbek"], "TM": ["Turkmen"], "TJ": ["Tajik"], "KG": ["Kyrgyz"],
 "MN": ["Mongolian"], "AM": ["Armenian"], "AZ": ["Azerbaijani"], "MD": ["Moldovan"],
 "LT": ["Lithuanian"], "LV": ["Latvian"], "EE": ["Estonian"], "IS": ["Icelandic"], "IE": ["Irish"],
 "CU": ["Cuban"], "HT": ["Haitian"], "DO": ["Dominican"], "JM": ["Jamaican"], "GY": ["Guyanese"],
 "SR": ["Surinamese"], "PA": ["Panamanian"], "CR": ["Costa Rican"], "HN": ["Honduran"],
 "NI": ["Nicaraguan"], "SV": ["Salvadoran"], "GT": ["Guatemalan"], "FJ": ["Fijian"],
 "PG": ["Papua New Guinean"], "TD": ["Chadian"], "MV": ["Maldivian"], "BT": ["Bhutanese"],
 "BN": ["Bruneian"], "GM": ["Gambian"], "GN": ["Guinean"], "LR": ["Liberian"], "SO": ["Somali"],
 "DJ": ["Djiboutian"], "ER": ["Eritrean"], "SS": ["South Sudanese"], "CF": ["Central African"],
 "GA": ["Gabonese"], "HK": ["Hong Kong"], "MO": ["Macanese"], "PR": ["Puerto Rican"],
 "GE": ["Georgian"], "CY": ["Cypriot"], "MT": ["Maltese"], "LU": ["Luxembourgish"],
}

BAN_PRE = {
 "US": ["North", "South", "Latin", "Central"], "KR": ["North"], "SD": ["South"],
 "WS": ["American"], "GN": ["Papua", "New", "Equatorial", "Bissau"],
 "MX": ["New"], "IE": ["Northern"], "HN": ["British"],
}
BAN_POST = {
 "GB": ["Isles", "Virgin"], "FR": ["horn", "fries", "press"],
 "GN": ["Bissau", "worm", "pig"], "US": ["Virgin", "Samoa"], "IN": ["Ocean"],
}

SEARCH = {
 "US": "United States", "GB": "United Kingdom", "KR": "South Korea", "KP": "North Korea",
 "AE": "United Arab Emirates", "CD": "Democratic Republic of the Congo", "CG": "Republic of the Congo",
 "CI": "Ivory Coast", "CV": "Cape Verde", "CZ": "Czechia", "TL": "East Timor", "SZ": "Eswatini",
 "MM": "Myanmar", "FM": "Micronesia", "MK": "North Macedonia", "VA": "Vatican", "PS": "Palestine",
 "SS": "South Sudan", "GW": "Guinea-Bissau", "GQ": "Equatorial Guinea", "PG": "Papua New Guinea",
 "SB": "Solomon Islands", "MH": "Marshall Islands", "MP": "Northern Mariana Islands",
 "VG": "British Virgin Islands", "VI": "U.S. Virgin Islands", "BQ": "Caribbean Netherlands",
 "TF": "French Southern Territories", "HM": "Heard Island", "AQ": "Antarctica", "BV": "Bouvet Island",
 "IO": "British Indian Ocean Territory", "GS": "South Sandwich Islands", "UM": "Minor Outlying Islands",
 "SJ": "Svalbard", "GE": "Georgian",
}

GE_CTX = ["Tbilisi", "Abkhazia", "Ossetia", "Batumi", "Kutaisi", "Rustavi", "Gori", "Mtskheta", "Kartli", "Caucasus"]

ALIASES = {}
def add(code, pat, ctx=None):
    ALIASES.setdefault(code, []).append({"re": re.compile(pat), "ctx": ctx})

def name_pat(name):
    return r"\b" + re.escape(name) + r"\b"

for c in countries:
    code = c.alpha_2
    names = {c.name}
    for k in ("official_name", "common_name"):
        v = getattr(c, k, None)
        if v:
            names.add(v)
    for n in names:
        add(code, name_pat(n))
    for e in EXTRA.get(code, []):
        add(code, e)
for code, dms in DEMONYMS.items():
    for d in dms:
        add(code, r"\b" + d + r"\b")
add("GE", r"\bGeorgia\b", ctx=GE_CTX)

for code in ALIASES:
    pre = BAN_PRE.get(code, [])
    post = BAN_POST.get(code, [])
    for e in ALIASES[code]:
        e["pre"] = set(pre)
        e["post"] = set(post)

AMB = ["Congo", "Macedonia", "Georgia", "New Guinea", "Congolese", "guinea worm"]
AMB_RE = [re.compile(r"\b" + re.escape(a) + r"\b", re.I) for a in AMB]

def labels_for(text):
    labs = set()
    for code, entries in ALIASES.items():
        for e in entries:
            if e["ctx"] is not None and not any(t in text for t in e["ctx"]):
                continue
            ok = False
            for m in e["re"].finditer(text):
                s = m.start()
                pre = " ".join(text[max(0, s - 30):s].split()[-2:])
                post = text[m.end():m.end() + 12].lstrip().lower()
                if any(pre.endswith(b) for b in e["pre"]):
                    continue
                if any(post.startswith(p.lower()) for p in e["post"]):
                    continue
                ok = True
                break
            if ok:
                labs.add(code)
                break
    return labs

def has_any_alias(text):
    for code, entries in ALIASES.items():
        for e in entries:
            if e["re"].search(text):
                return True
    return False

def is_clean_negative(text):
    if has_any_alias(text):
        return False
    for r in AMB_RE:
        if r.search(text):
            return False
    return True

def inv2txt(inv):
    if not inv:
        return ""
    pos = []
    for w, idxs in inv.items():
        for i in idxs:
            pos.append((i, w))
    pos.sort()
    return " ".join(w for _, w in pos)

def api_search(term, max_works):
    out = {}
    cursor = "*"
    filt = f'title_and_abstract.search:"{term}",language:en'
    while len(out) < max_works:
        params = {"filter": filt, "per-page": PER_PAGE, "cursor": cursor, "mailto": MAILTO,
                  "api_key": API_KEY, "select": "id,title,abstract_inverted_index"}
        j = None
        for attempt in range(7):
            try:
                r = requests.get(API, params=params, timeout=60, headers=HEADERS)
                if r.status_code in (401, 429, 500, 502, 503):
                    wait = min(2 * (attempt + 1), 30)
                    print(f"[http] {r.status_code} on '{term}', retry in {wait}s", flush=True)
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                j = r.json()
                break
            except Exception:
                if attempt == 6:
                    raise
                time.sleep(min(2 * (attempt + 1), 30))
        results = j.get("results", [])
        if not results:
            break
        for w in results:
            out[w["id"]] = w
        if len(results) < PER_PAGE:
            break
        cursor = j.get("meta", {}).get("next_cursor")
        if not cursor:
            break
        time.sleep(0.12)
    return list(out.values())

def ingest(ws):
    for w in ws:
        title = (w.get("title") or "").strip()
        abstract = inv2txt(w.get("abstract_inverted_index"))
        if not title:
            continue
        text = title + ("\n" + abstract if abstract else "")
        labs = labels_for(text)
        if not labs:
            continue
        prev = works.get(w["id"], {"labels": set()})
        prev.update({"title": title, "abstract": abstract, "labels": prev["labels"] | labs})
        works[w["id"]] = prev

works = {}
fail_codes = []
for n, code in enumerate(codes):
    term = SEARCH.get(code, by_code[code].name)
    try:
        ingest(api_search(term, MAX_WORKS))
    except Exception as ex:
        print(f"[fetch] {code} FAILED: {ex}", flush=True)
        fail_codes.append(code)
    if (n + 1) % 25 == 0:
        print(f"[fetch] {n+1}/{len(codes)} codes done, works={len(works)}, failed={fail_codes}", flush=True)

for round_ in range(3):
    if not fail_codes:
        break
    print(f"[retry-round {round_+1}] codes: {fail_codes}", flush=True)
    time.sleep(30)
    for code in list(fail_codes):
        term = SEARCH.get(code, by_code[code].name)
        try:
            ingest(api_search(term, MAX_WORKS))
            fail_codes.remove(code)
            print(f"[retry] {code} ok", flush=True)
        except Exception as ex:
            print(f"[retry] {code} still failed: {ex}", flush=True)

NEG_QUERIES = ["graph neural networks", "protein structure prediction", "climate change impacts",
 "quantum error correction", "randomized clinical trial", "lithium ion battery",
 "antibiotic resistance genes", "word embeddings", "supply chain optimization",
 "cancer immunotherapy", "soil organic carbon", "deep reinforcement learning",
 "computer vision", "photonic crystals", "microbiome", "carbon nanotubes",
 "statistical physics", "machine translation", "drug delivery", "seismic tomography",
 "transformer architecture", "federated learning", "precision agriculture", "ocean acidification",
 "cognitive behavioral therapy", "quantum entanglement", "crowdsourcing", "photovoltaics",
 "natural language inference", "object detection"]

negatives = []
for q in NEG_QUERIES:
    try:
        ws = api_search(q, 600)
    except Exception as ex:
        print(f"[neg] {q} failed: {ex}", flush=True)
        continue
    for w in ws:
        title = (w.get("title") or "").strip()
        abstract = inv2txt(w.get("abstract_inverted_index"))
        if not title:
            continue
        text = title + ("\n" + abstract if abstract else "")
        if is_clean_negative(text):
            negatives.append({"title": title, "abstract": abstract, "id": w["id"]})
    time.sleep(0.1)

negatives = list({n["id"]: n for n in negatives}.values())
print(f"[neg] {len(negatives)} clean negatives", flush=True)

ids = sorted(works.keys())
random.shuffle(ids)
n = len(ids)
split_of = {}
for i, wid in enumerate(ids):
    split_of[wid] = "test" if i < int(0.05 * n) else ("validation" if i < int(0.05 * n) + int(0.05 * n) else "train")

data = {"train": [], "validation": [], "test": []}
for wid, rec in works.items():
    sp = split_of[wid]
    labs = sorted(rec["labels"])
    text = rec["title"] + ("\n" + rec["abstract"] if rec["abstract"] else "")
    data[sp].append({"text": text, "labels": labs, "has_abstract": rec["abstract"] != "", "source": "openalex"})
    if rec["abstract"] and random.random() < 0.15:
        data[sp].append({"text": rec["title"], "labels": labs, "has_abstract": False, "source": "openalex_titleonly"})
for nrec in negatives:
    sp = "train" if random.random() < 0.9 else ("validation" if random.random() < 0.5 else "test")
    text = nrec["title"] + ("\n" + nrec["abstract"] if nrec["abstract"] else "")
    data[sp].append({"text": text, "labels": [], "has_abstract": nrec["abstract"] != "", "source": "openalex_neg"})
for sp in data:
    random.shuffle(data[sp])
    print(f"[split] {sp}: {len(data[sp])} rows", flush=True)

from collections import Counter
cnt_train = Counter()
for row in data["train"]:
    for c in row["labels"]:
        cnt_train[c] += 1
zero_codes = [c for c in codes if cnt_train[c] == 0]
print(f"[coverage] zero-count codes: {len(zero_codes)} {zero_codes[:20]}", flush=True)

dsdict = DatasetDict({"train": Dataset.from_list(data["train"]),
                      "validation": Dataset.from_list(data["validation"]),
                      "test": Dataset.from_list(data["test"])})
dsdict.push_to_hub(DS_REPO, token=HF_TOKEN)
print("[push] dataset uploaded", flush=True)

import tempfile
api = HfApi(token=HF_TOKEN)
with tempfile.NamedTemporaryFile("w", delete=False) as f:
    json.dump({c: cnt_train.get(c, 0) for c in codes}, f)
    p = f.name
api.upload_file(path_or_fileobj=p, path_in_repo="counts.json", repo_id=DS_REPO, repo_type="dataset", commit_message="per-code train counts")
with tempfile.NamedTemporaryFile("w", delete=False) as f:
    json.dump(codes, f)
    p = f.name
api.upload_file(path_or_fileobj=p, path_in_repo="codes.json", repo_id=DS_REPO, repo_type="dataset", commit_message="canonical code order")
print("[done]", flush=True)
