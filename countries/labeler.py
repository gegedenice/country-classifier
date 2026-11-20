"""Rule-based country labeler (weak supervision) built on data/country_lexicon.json.

Matching rules:
  * aliases match on Unicode word boundaries, case-sensitive unless flagged otherwise;
  * all candidate spans (aliases, blockers, ambiguous words) are collected, then overlaps
    are resolved by keeping the longest span: "Papua New Guinea" beats "New Guinea" and
    "Guinea", "Guinea-Bissau" beats "Guinea", "New Mexico" (blocker) beats "Mexico";
  * a blocker span produces no label ("Gulf of Mexico", "American Journal", "Jordan curve");
  * ambiguous words ("Congo", "Georgia", "Macedonia") are resolved with context rules
    evaluated on the whole text.

    from countries.labeler import Labeler
    lab = Labeler()
    lab.labels("Ebola in the Democratic Republic of the Congo")   # {'CD'}
    lab.explain("...")                                            # spans + decisions
"""
import json
import re
import unicodedata
from pathlib import Path

DEFAULT_LEXICON = Path(__file__).resolve().parents[1] / "data" / "country_lexicon.json"
_B = r"(?<![\w-])"   # left boundary: not preceded by a letter/digit or hyphen
_E = r"(?![\w])"     # right boundary


def _norm(text):
    text = unicodedata.normalize("NFC", text or "")
    return text.replace("’", "'").replace(" ", " ")


class Labeler:
    def __init__(self, lexicon_path=DEFAULT_LEXICON):
        lex = json.loads(Path(lexicon_path).read_text())
        self.codes = lex["codes"]
        self.cs, self.ci = {}, {}
        for code, items in lex["aliases"].items():
            for a in items:
                t = _norm(a["text"])
                if a["case_sensitive"]:
                    self.cs[t] = code
                else:
                    self.ci[t.lower()] = code
        self.ambiguous = lex["ambiguous"]

        def alt(words):
            return "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))

        self.re_cs = re.compile(_B + "(?:" + alt(list(self.cs) + list(self.ambiguous)) + ")" + _E)
        self.re_ci = re.compile(_B + "(?:" + alt(self.ci) + ")" + _E, re.I)
        self.re_block = re.compile("|".join(f"(?:{b})" for b in lex["blockers"]))

    def _candidates(self, text):
        cands = []  # (start, end, kind, value)
        for m in self.re_block.finditer(text):
            if m.end() > m.start():
                cands.append((m.start(), m.end(), "block", m.group(0)))
        for m in self.re_cs.finditer(text):
            w = m.group(0)
            if w in self.ambiguous:
                cands.append((m.start(), m.end(), "ambiguous", w))
            else:
                cands.append((m.start(), m.end(), "alias", self.cs[w]))
        for m in self.re_ci.finditer(text):
            cands.append((m.start(), m.end(), "alias", self.ci[m.group(0).lower()]))
        return cands

    @staticmethod
    def _resolve_overlaps(cands):
        kept, taken = [], []
        # longest first; on equal length, blockers win (prudence)
        for c in sorted(cands, key=lambda c: (-(c[1] - c[0]), c[2] != "block", c[0])):
            if any(c[0] < e and s < c[1] for s, e in taken):
                continue
            kept.append(c)
            taken.append((c[0], c[1]))
        return sorted(kept)

    def _resolve_ambiguous(self, word, text):
        for rule in self.ambiguous[word]:
            req = rule.get("require_any")
            forb = rule.get("forbid_any")
            if req and not any(r in text for r in req):
                continue
            if forb and any(f in text for f in forb):
                continue
            return rule["code"]
        return None

    def explain(self, text):
        text = _norm(text)
        out = []
        for s, e, kind, val in self._resolve_overlaps(self._candidates(text)):
            if kind == "ambiguous":
                code = self._resolve_ambiguous(val, text)
                out.append({"span": text[s:e], "start": s, "kind": kind, "code": code})
            elif kind == "alias":
                out.append({"span": text[s:e], "start": s, "kind": kind, "code": val})
            else:
                out.append({"span": text[s:e], "start": s, "kind": kind, "code": None})
        return out

    def labels(self, text):
        return {d["code"] for d in self.explain(text) if d["code"]}

    def has_trap(self, text):
        """True when the text contains a blocker or an unresolved ambiguous word (hard negative)."""
        return any(d["code"] is None for d in self.explain(text))
