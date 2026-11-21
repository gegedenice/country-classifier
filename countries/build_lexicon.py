"""Build data/country_lexicon.json: the alias lexicon used to label publications.

Sources, in this order:
  1. OpenAlex country vocabulary (https://api.openalex.org/countries): the 247 codes,
     display_name and display_name_alternatives.
  2. Wikidata (through each country's QID given by OpenAlex): labels, aliases and
     demonyms (P1549) in en, fr, es, de, pt.
  3. countries/overrides.py: manual additions, removals and ambiguity rules.

Automatic aliases are filtered (no 2-3 letter codes, no lowercase English aliases,
no aliases shared by several countries) because they are the main source of noise.

Usage:  python countries/build_lexicon.py            (network: OpenAlex + Wikidata)
Env:    OPENALEX_API_KEY (optional, sent as Bearer header)
"""
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import overrides as ov  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "country_lexicon.json"
LANGS = ["en", "fr", "es", "de", "pt"]
UA = {"User-Agent": "country-classifier-lexicon/2.0 (https://github.com/gegedenice/country-classifier)"}


def openalex_countries():
    headers = dict(UA)
    if os.environ.get("OPENALEX_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['OPENALEX_API_KEY']}"
    out = []
    for page in (1, 2):
        r = requests.get("https://api.openalex.org/countries", params={"per-page": 200, "page": page},
                         headers=headers, timeout=60)
        r.raise_for_status()
        out += r.json()["results"]
    return out


def wikidata_terms(qids):
    """Return {qid: [(kind, lang, text)]} for labels, aliases and demonyms."""
    res = defaultdict(list)
    for i in range(0, len(qids), 60):
        chunk = qids[i:i + 60]
        q = """SELECT ?item ?kind ?label WHERE { VALUES ?item { %s }
          { ?item rdfs:label ?label . BIND("label" AS ?kind) }
          UNION { ?item skos:altLabel ?label . BIND("alias" AS ?kind) }
          UNION { ?item wdt:P1549 ?label . BIND("demonym" AS ?kind) }
          FILTER(LANG(?label) IN (%s)) }""" % (
            " ".join("wd:" + x for x in chunk), ",".join(f'"{l}"' for l in LANGS))
        for attempt in range(5):
            r = requests.post("https://query.wikidata.org/sparql", data={"query": q, "format": "json"},
                              headers=UA, timeout=180)
            if r.status_code == 200:
                break
            time.sleep(5 * (attempt + 1))
        r.raise_for_status()
        for b in r.json()["results"]["bindings"]:
            qid = b["item"]["value"].rsplit("/", 1)[1]
            res[qid].append((b["kind"]["value"], b["label"]["xml:lang"], b["label"]["value"]))
        time.sleep(1)
    return res


def inflect(word, lang):
    """Gender/number variants for demonym adjectives in fr/es/pt/de (singular masculine input only)."""
    w = {word}
    lw = word
    if lang == "fr":
        if lw.endswith(("ais", "ois")):
            w |= {lw + "e", lw + "es"}
        elif lw.endswith(("s", "x", "es")):
            pass                                    # already plural / invariable
        elif lw.endswith("ien"):
            w |= {lw + "ne", lw + "nes", lw + "s"}
        elif lw.endswith(("ain", "an", "in")):
            w |= {lw + "e", lw + "es", lw + "s"}
        elif lw.endswith("e"):
            w |= {lw + "s"}
        else:
            w |= {lw + "e", lw + "s", lw + "es"}
    elif lang in ("es", "pt"):
        if lw.endswith("s") and not lw.endswith(("ês", "és")):
            pass
        elif lw.endswith("o"):
            w |= {lw[:-1] + "a", lw + "s", lw[:-1] + "as"}
        elif lw.endswith(("ês", "és")):
            w |= {lw[:-2] + "esa", lw[:-2] + "eses", lw[:-2] + "esas"}
        elif lw.endswith(("a", "e")):
            w |= {lw + "s"}
        else:
            w |= {lw + "es"}
    elif lang == "de":
        if lw.endswith("er"):
            w |= {lw + "in", lw + "innen"}
    return w


def keep_alias(text, kind, lang):
    t = text.strip()
    if len(t) < 4 or t in ov.GLOBAL_DROP or t.lower() in ov.GENERIC:
        return False
    core = re.sub(r"^(?:the|die|der|les|la|le|los|las|l'|el|o|a|os|as)\s+", "", t, flags=re.I)
    if core.lower() in ov.GENERIC or core in ov.GLOBAL_DROP:
        return False
    if re.fullmatch(r"[A-Z.\s-]{2,8}", t):        # ISO-like codes and initials: only via overrides
        return False
    if re.search(r"\d", t) or re.fullmatch(r"\w{1,5}\.", t):  # subdivision codes, abbreviations ("All.")
        return False
    if all(len(tok.strip(".")) <= 2 for tok in t.split()):
        return False
    if kind == "alias":
        if not t[0].isupper():
            return False
        if any(ch in t for ch in "()[]/;:'’\""):
            return False
        if len(t.split()) > 7:
            return False
        if " " not in t and len(t) < 5:              # short one-word nicknames ("Cook", "Saba")
            return False
    if kind == "demonym" and lang != "en" and len(t) < 5:
        return False
    return True


def main():
    oa = openalex_countries()
    codes = sorted(c["country_code"] for c in oa)
    assert len(codes) == 247, len(codes)
    by_code = {c["country_code"]: c for c in oa}
    qid = {c["country_code"]: c["ids"]["wikidata"].rsplit("/", 1)[1] for c in oa}
    wd = wikidata_terms(list(qid.values()))

    # code -> {(text, case_sensitive): source}
    entries = defaultdict(dict)

    def put(code, text, cs, src):
        text = re.sub(r"\s+", " ", text.strip())
        if text:
            entries[code].setdefault((text, cs), src)

    for code in codes:
        c = by_code[code]
        if c["display_name"] not in ov.GLOBAL_DROP and c["display_name"] not in ov.AMBIGUOUS:
            put(code, c["display_name"], True, "openalex")
        for a in c.get("display_name_alternatives") or []:
            if keep_alias(a, "alias", "en"):
                put(code, a, True, "openalex_alt")
        for kind, lang, text in wd.get(qid[code], []):
            if not keep_alias(text, kind, lang):
                continue
            if kind == "demonym":
                if lang == "en":
                    put(code, text, True, "wikidata_demonym_en")
                else:
                    for v in inflect(text, lang):
                        if code in ov.LANGUAGE_DEMONYM_CODES:
                            # demonym == language name ("le français"): people only, capitalised
                            if v[0].isupper():
                                put(code, v, True, f"wikidata_demonym_{lang}")
                        else:
                            put(code, v, False, f"wikidata_demonym_{lang}")
            else:
                put(code, text, True, f"wikidata_{kind}_{lang}")

    # Aliases claimed by several countries are ambiguous: drop them (overrides may re-add).
    owners = defaultdict(set)
    for code, d in entries.items():
        for (text, cs) in d:
            owners[text.lower()].add(code)
    dropped = defaultdict(set)
    for code, d in entries.items():
        for key in list(d):
            if len(owners[key[0].lower()]) > 1:
                if not d[key].startswith("wikidata_alias"):
                    dropped[key[0]] |= owners[key[0].lower()]
                del d[key]
    print("[lexicon] dropped (shared by several countries):",
          {t: sorted(c) for t, c in sorted(dropped.items())})

    # Manual overrides.
    for code, texts in ov.REMOVE.items():
        for key in list(entries[code]):
            if key[0] in texts:
                del entries[code][key]
    for code, texts in ov.ADD.items():
        for t in texts:
            entries[code][(t, True)] = "manual"
    for code, texts in ov.ADD_CI.items():
        for t in texts:
            entries[code][(t, False)] = "manual"
    for code, texts in ov.PT_ES_DEMONYMS.items():
        for t in texts:
            for v in inflect(t, "pt"):
                entries[code][(v, False)] = "manual_demonym_pt_es"

    for code in codes:
        for key in list(entries[code]):
            if key[0] in ov.AMBIGUOUS:
                del entries[code][key]

    lex = {
        "version": 2,
        "codes": codes,
        "names": {c: by_code[c]["display_name"] for c in codes},
        "wikidata": qid,
        "aliases": {c: sorted([{"text": t, "case_sensitive": cs, "source": s}
                               for (t, cs), s in entries[c].items()], key=lambda x: x["text"])
                    for c in codes},
        "blockers": ov.BLOCKERS,
        "ambiguous": ov.AMBIGUOUS,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(lex, ensure_ascii=False, indent=1))
    n = sum(len(v) for v in lex["aliases"].values())
    print(f"[lexicon] {len(codes)} codes, {n} aliases -> {OUT}")
    thin = [c for c in codes if len(lex["aliases"][c]) < 3]
    print(f"[lexicon] codes with <3 aliases: {thin}")


if __name__ == "__main__":
    main()
