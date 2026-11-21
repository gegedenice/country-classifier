"""Regression tests for the rule labeler. Each case comes from an observed labelling error or a known trap.

Run:  python -m pytest tests/
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from countries.labeler import Labeler  # noqa: E402

LAB = Labeler()

CASES = [
    # names that must be recognised
    ("Economic growth in Russia", {"RU"}),
    ("Earthquakes in Turkey", {"TR"}),
    ("Health system of Korea", {"KR"}),
    ("Oil revenues in Brunei", {"BN"}),
    ("Gaza and Palestine", {"PS"}),
    ("The Vatican archives", {"VA"}),
    ("Ebola in the Democratic Republic of the Congo", {"CD"}),
    ("Kinshasa, DR Congo", {"CD"}),
    ("Brazzaville, Republic of the Congo", {"CG"}),
    ("Malaria in Guinea-Bissau", {"GW"}),
    ("Papua New Guinea languages", {"PG"}),
    ("Equatorial Guinea oil", {"GQ"}),
    ("Kosovo independence", {"XK"}),
    ("Scotland and Wales", {"GB"}),
    ("Nuclear program of North Korea", {"KP"}),
    ("Falkland Islands fisheries", {"FK"}),
    # traps that must not be labelled
    ("Soil survey of New Jersey", set()),
    ("Atlanta, Georgia public schools", set()),
    ("Jordan curve theorem in topology", set()),
    ("Chinese hamster ovary cells expression", set()),
    ("Spanish flu pandemic of 1918", set()),
    ("Published in the American Journal of Physics", set()),
    ("Gulf of Mexico oil spill", set()),
    ("Indian Ocean dipole", set()),
    ("German shepherd dogs", set()),
    ("Guinea pig model of asthma", set()),
    ("Democracy in Latin America", set()),
    ("New Mexico drought", set()),
    ("Ancient Macedonia and Philip II", set()),
    # positives that must still work
    ("Wine regions of Georgia: Kakheti and Tbilisi", {"GE"}),
    ("Rice farming in Vietnam", {"VN"}),
    ("Comparing France and Germany", {"DE", "FR"}),
    ("Study in the U.S. Midwest", {"US"}),
    ("Côte d'Ivoire cocoa", {"CI"}),
    ("Hong Kong protests", {"HK"}),
    ("Migration from Mexico to the United States", {"MX", "US"}),
    ("Skopje, North Macedonia", {"MK"}),
    # multilingual
    ("La pauvreté rurale au Sénégal et au Mali", {"SN", "ML"}),
    ("Les agriculteurs sénégalais face au changement climatique", {"SN"}),
    ("Políticas de salud en Colombia", {"CO"}),
    ("A agricultura familiar no Brasil", {"BR"}),
    ("Migration nach Deutschland", {"DE"}),
    ("Les États-Unis et la Chine", {"US", "CN"}),
    ("La République démocratique du Congo", {"CD"}),
    ("Enseignement du français langue étrangère", set()),
]


@pytest.mark.parametrize("text,expected", CASES)
def test_labels(text, expected):
    assert LAB.labels(text) == expected, LAB.explain(text)


def test_codes_are_openalex_247():
    assert len(LAB.codes) == 247 and "XK" in LAB.codes and "UM" not in LAB.codes


def test_overrides_have_no_duplicate_keys():
    import ast
    src = (Path(__file__).resolve().parents[1] / "countries" / "overrides.py").read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Dict):
            keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
            assert len(keys) == len(set(keys)), sorted(k for k in set(keys) if keys.count(k) > 1)


@pytest.mark.parametrize("text,expected", [
    ("British conduct during the Mau Mau rebellion in Kenya", {"GB", "KE"}),
    ("Trade between China and Taiwan", {"CN", "TW"}),
    ("French Polynesia: Moorea and Papeete", {"PF"}),
    ("Congolese refugees in Goma, North Kivu", {"CD"}),
])
def test_shared_aliases(text, expected):
    assert LAB.labels(text) == expected, LAB.explain(text)


@pytest.mark.parametrize("text,expected", [
    ("Dominique Saint-Martin", set()),
    ("University of the West Indies, Trinidad-Tobago", {"TT"}),
    ("Tourism in Saint-Martin and Sint Maarten", {"MF", "SX"}),
])
def test_names_and_places(text, expected):
    assert LAB.labels(text) == expected, LAB.explain(text)


@pytest.mark.parametrize("text,expected", [
    # homonyms found by the LLM check
    ("Circadian rhythm in the eel Anguilla anguilla", set()),
    ("Habitat use by the European eel Anguilla anguilla in Irish waters", {"IE"}),
    ("Tourism in Anguilla", {"AI"}),
    ("Urban violence in Salvador, Bahia", set()),
    ("Elections in El Salvador", {"SV"}),
    ("Turismo activo en Granada y Sevilla", set()),
    ("Nutmeg production in Grenada", {"GD"}),
    ("Salomon Maimons Lebensgeschichte", set()),
    ("Excursions into Inauthenticity: The Chamorros of Guam", {"GU"}),
    # Portuguese / Spanish demonyms
    ("A agricultura familiar no semiárido brasileiro", {"BR"}),
    ("Mulheres angolanas e a guerra civil", {"AO"}),
    ("La economía mexicana tras el TLCAN", {"MX"}),
])
def test_homonyms_and_demonyms(text, expected):
    assert LAB.labels(text) == expected, LAB.explain(text)
