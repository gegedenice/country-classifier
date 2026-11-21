# country-classifier : détecter les pays mentionnés dans une publication scientifique

Ce projet produit un modèle qui lit une publication scientifique et indique **quels pays elle mentionne**. Il prend en entrée le titre, avec le résumé s'il existe, et répond parmi les **247 pays du vocabulaire OpenAlex** ([api.openalex.org/countries](https://api.openalex.org/countries)) : codes ISO 3166-1 alpha-2, plus `XK` pour le Kosovo.

L'objectif est de **compenser le manque d'indexation géographique** des bases bibliographiques. Le champ « couverture géographique » est souvent vide ou mal rempli, alors que le titre et le résumé contiennent l'information.

**Ce que le modèle détecte : une mention explicite.** Le texte nomme le pays, par son nom, une variante ou un gentilé (« France », « USA », « Côte d'Ivoire », « Kenyan », « brasileiro »), au sens du pays lui-même, de sa population ou de quelque chose qui en vient. Les homonymes ne comptent pas : une espèce (*Anguilla anguilla*), une ville ou un État du même nom (Salvador de Bahia, New Jersey, Georgia aux États-Unis), un prénom, ou un nom de langue employé seulement comme langue. Le modèle **ne juge pas** si la publication « porte sur » le pays : c'est un choix délibéré, car cette notion documentaire est ambiguë et difficile à évaluer sans annotation humaine.

| | v1 (ML Intern) | v2.1 (ce dépôt) |
|---|---|---|
| Liste de pays | 249 codes `pycountry` | **247 codes OpenAlex** |
| Langues | anglais | **anglais, français, espagnol, allemand, portugais** |
| Modèle de base | ModernBERT-base | **mmBERT-base** (multilingue) |
| Étiquettes | regex sur les noms `pycountry` | **lexique OpenAlex + Wikidata** et bloqueurs, puis **chaque étiquette vérifiée par un LLM** |
| Négatifs | 30 thèmes STEM | **échantillon aléatoire d'OpenAlex** + négatifs difficiles |
| Décision | un seuil par pays | **un seuil unique, 0,5** |

- Dataset : `Geraldine/pub-country-classification-dataset-v2.1` (privé pour l'instant)
- Modèle : `Geraldine/pub-country-classifier-v2.1` (privé, à entraîner)
- Versions précédentes : voir l'historique (§ 11)

---

## 1. Formulation du problème

Un texte peut mentionner plusieurs pays (« comparaison France–Allemagne ») ou aucun (« une nouvelle architecture de réseau de neurones »). Il s'agit donc d'une **classification multi-label** :

- en entrée, un texte : le titre sur la ligne 1, le résumé, facultatif, sur la ligne 2 ;
- en sortie, **un score entre 0 et 1 pour chacun des 247 pays**, calculés indépendamment ;
- **un pays est retenu si son score est supérieur ou égal à 0,5.**

Exemple :

> « Urban agriculture in Kenya and Uganda » → KE 0,99 · UG 0,97 · TZ 0,08 · FR 0,001 … → **KE, UG**

Une liste vide est une réponse normale : la plupart des publications ne mentionnent aucun pays.

## 2. Vue d'ensemble

```
OpenAlex /countries ─┐
Wikidata (5 langues) ─┼─► lexique des pays ─► labeler à règles
overrides manuels ───┘    (data/country_lexicon.json)     │
                                                           ▼
OpenAlex /works : requêtes par pays (5 langues) + échantillon aléatoire
                                                           │
                                                           ▼
       dataset : étiquettes des règles (train 90 % / validation 5 % / test 5 %)
                                                           │
                                                           ▼
   vérification par un LLM de chaque étiquette, sur TOUTES les notices (judge_labels.py)
   → verified_labels = étiquettes des règles − fausses mentions + mentions oubliées
                                                           │
                                                           ▼
   fine-tuning mmBERT-base sur verified_labels ─► évaluation sur verified_labels (test) ─► Hub
```

**Une seule définition des étiquettes, de bout en bout :** l'entraînement, la validation et le test utilisent tous `verified_labels`.

| Fichier | Rôle |
|---|---|
| `countries/build_lexicon.py` | construit le lexique des alias de pays |
| `countries/overrides.py` | corrections manuelles, documentées |
| `countries/labeler.py` | étiqueteur à règles |
| `tests/test_labeler.py` | tests de non-régression (pièges connus) |
| `build_country_dataset.py` | collecte OpenAlex, étiquetage, découpage, publication |
| `judge_labels.py` | vérification des mentions par un LLM |
| `train_country_model.py` | fine-tuning, évaluation, publication |
| `inference_example.py` | exemple d'utilisation |

## 3. Le lexique des pays

### 3.1 Sources

1. **OpenAlex** (`/countries`) fournit les 247 codes, le nom d'affichage, quelques variantes et l'identifiant Wikidata de chaque pays.
2. **Wikidata**, via ces identifiants, apporte en anglais, français, espagnol, allemand et portugais :
   - les libellés (« Allemagne », « Deutschland », « Alemania »…) ;
   - les alias (« Burma », « Zaïre », « Holland »…) ;
   - les gentilés (propriété P1549 : « Kenyan », « sénégalais », « brasileño »…).
   Pour les gentilés, des formes fléchies sont générées (féminin, pluriel).
   Wikidata a peu de gentilés portugais et espagnols : environ 70 ont été ajoutés à la main (« brasileiro », « angolano », « mexicano »…).
3. **`countries/overrides.py`** contient les corrections manuelles.

Les alias automatiques sont filtrés, car c'est la principale source de bruit. Sont écartés :
- les codes de 2 à 4 lettres et les abréviations (« All. », « Col. ») ;
- les mots génériques (« States », « Union ») ;
- les surnoms courts (« Cook », « Heard ») ;
- les alias revendiqués par plusieurs pays : « China » est revendiqué par CN et TW, « Congolese » par CD et CG. Ces alias sont réattribués à la main quand c'est pertinent.

Résultat : **environ 4 840 alias** pour 247 pays, dans `data/country_lexicon.json`.

### 3.2 Règles d'étiquetage (`countries/labeler.py`)

- **Correspondance sur les frontières de mots Unicode**, sensible à la casse. Les adjectifs en minuscules (« sénégalais ») sont la seule exception.
- **La plus longue correspondance l'emporte.** On collecte tous les candidats puis on garde le plus long en cas de chevauchement : « Papua New Guinea » bat « New Guinea » et « Guinea », « Guinea-Bissau » bat « Guinea », « Democratic Republic of the Congo » bat « Congo ».
- **Les bloqueurs** sont des expressions qui ressemblent à un pays sans en être un. Elles « consomment » le texte et ne produisent aucune étiquette. Exemples :
  - lieux : « New Jersey », « New Mexico », « Gulf of Mexico », « Atlanta, Georgia », « Latin America », « Indian Ocean » ;
  - espèces, maladies, termes techniques : « Guinea pig », « Spanish flu », « Chinese hamster », « Jordan curve », « French horn », *« Anguilla anguilla »* ;
  - organisations et revues : « American Journal of… », « British Medical Journal », « German Research Foundation » ;
  - noms de personnes : « Dominique », « Jean Saint-Martin ».
- **Les mots ambigus** sont résolus par le contexte :
  - « Congo » donne `CD` si le texte cite Kinshasa, Kivu, DRC…, et `CG` s'il cite Brazzaville, Pointe-Noire… ;
  - « Georgia » donne `GE`, sauf en contexte américain (Atlanta, USA, county…) ;
  - « Macedonia » donne `MK`, sauf en contexte grec ou antique.
- **Gentilés et noms de langue.** En français, espagnol et portugais, « le français » ou « el japonés » désignent aussi la langue. Pour les pays concernés, seule la forme à majuscule (« les Français ») est retenue en dehors de l'anglais.

Certains homonymes trop fréquents ont été retirés du lexique : « Salvador » (Salvador de Bahia), « Granada » / « Grenade », « Salomon », « Santa Elena ». On garde à la place les formes sans ambiguïté (« El Salvador », « Grenada », « Îles Salomon »).

Chaque erreur relevée, lors de la revue de la v1 puis par la vérification LLM de la v2, est devenue un test (`tests/test_labeler.py`, 66 tests) :

| Texte | v1 | v2 |
|---|---|---|
| « Economic growth in Russia » | rien | `RU` |
| « Earthquakes in Turkey » | rien | `TR` |
| « Ebola in the Democratic Republic of the Congo » | `CG` | `CD` |
| « Soil survey of New Jersey » | `JE` | rien |
| « Atlanta, Georgia public schools » | `GE` | rien |
| « Malaria in Guinea-Bissau » | `GN` + `GW` | `GW` |
| « Published in the American Journal of Physics » | `US` | rien |
| « Les agriculteurs sénégalais… » | *(hors périmètre)* | `SN` |
| « Enseignement du français langue étrangère » | *(hors périmètre)* | rien |
| « Circadian rhythm in the eel Anguilla anguilla » | `AI` | rien (v2.1) |
| « A agricultura familiar no semiárido brasileiro » | `BR` | `BR` (v2.1) |
| « Urban violence in Salvador, Bahia » | `SV` | rien (v2.1) |

```bash
python -m pytest tests/
```

## 4. Le dataset (`build_country_dataset.py`)

### 4.1 Collecte OpenAlex

Pour chacun des 247 pays :
- **jusqu'à 400 publications anglophones**, cherchées sur le nom anglais et quelques variantes (« Russia » OR « Russian Federation »…) ;
- **jusqu'à 200 publications en fr/es/de/pt**, avec une seule requête booléenne sur les noms du pays dans ces langues (« Kenia » OR « Quénia » OR « Kénya »…).

S'y ajoute un **échantillon aléatoire d'OpenAlex** (paramètre `sample`) : 24 000 publications dans les 5 langues, plus 4 000 par langue hors anglais. Cet échantillon reflète une vraie base bibliographique, toutes disciplines confondues : **seules 16 % de ces publications nomment un pays**. Ce sont des négatifs réalistes.

Seuls les types `article`, `book`, `book-chapter`, `dissertation`, `preprint`, `review` et `report` sont retenus. Les fiches de données (GBIF…), les pages de menus et autres scories sont écartées. Les textes qui nomment plus de 12 pays (listes, formulaires) sont aussi supprimés : il y en a 672.

Les requêtes ne servent qu'à **choisir** les publications : **les étiquettes viennent toujours du labeler**. Une requête large comme « Georgia » ou « Jordan » ramène donc aussi des textes sans le pays (l'État américain, la courbe de Jordan…). Ces textes sont conservés avec une liste vide : ce sont des **négatifs difficiles**, et il y en a environ 7 000.

### 4.2 Notices sans résumé

15 % des publications avec résumé sont dupliquées en version **titre seul**. Contrairement à la v1, ces copies sont **réétiquetées à partir du titre seul** : le modèle n'apprend pas à deviner un pays absent du texte.

### 4.3 Découpage

Le découpage 90/5/5 se fait par **hachage de l'identifiant OpenAlex**. Il est stable et reproductible, et une publication reste toujours dans la même partition que sa copie « titre seul ».

### 4.4 Chiffres

| | train | validation | test |
|---|---|---|---|
| lignes | 145 781 | 8 061 | 7 954 |
| part sans pays | 27 % | 28 % | 28 % |

- Langues (train) : en 91 260, fr 18 294, es 17 239, de 11 132, pt 7 856.
- **Les 247 codes ont des exemples.** En v1, `FK`, `MF`, `SX` et `VA` n'en avaient aucun.
- Seuls trois codes ont moins de 50 exemples : `PM` (Saint-Pierre-et-Miquelon, 16), `MP` (îles Mariannes du Nord, 28) et `WF` (Wallis-et-Futuna, 31).
- Exemples de corrections par rapport à la v1 : RD Congo 572 exemples (269 en v1, et la plupart étiquetés `CG` à tort), Russie 1 570, Brésil 2 318.
- Ces chiffres sont ceux des **règles**, avant la vérification par le LLM (§ 5).

Colonnes : `id`, `text`, `labels`, `language`, `year`, `type`, `origin` (`country_query` ou `random`), `variant` (`full` ou `title_only`), `has_abstract` et `has_trap` (le texte contient un bloqueur ou un mot ambigu non résolu).

Le budget OpenAlex est d'environ **0,80 $** (clé gratuite, 1 $/jour). Les pages brutes sont mises en cache (`.cache/`), donc une relance ne coûte rien.

## 5. La vérification des mentions par un LLM (`judge_labels.py`)

Les règles font des erreurs que le lexique ne peut pas toutes prévoir. Elles prennent un homonyme pour un pays, ou ratent un pays écrit d'une façon inattendue. Un LLM relit donc **chaque notice, des trois partitions**, avec les pays trouvés par les règles, et répond à deux questions seulement :

1. **Quelles étiquettes sont fausses ?** Le mot qui les a déclenchées désigne autre chose que le pays : une espèce, une ville ou un État homonyme, un prénom, un nom de langue employé comme langue…
2. **Quels pays sont nommés explicitement mais ont été oubliés ?** Pour chacun, le LLM doit **citer les mots exacts du texte**, et le script vérifie que ces mots y figurent vraiment. Un pays sans citation vérifiable est ignoré.

Le LLM **ne juge pas la pertinence** : une mention correcte reste, même en passant. Il **n'ajoute jamais un pays à partir de la langue du texte**, d'une ville, d'une région ou d'une institution.

```
verified_labels = (étiquettes des règles − étiquettes rejetées) + mentions oubliées (citation vérifiée)
```

| | |
|---|---|
| Modèle | `Qwen/Qwen3-235B-A22B-Instruct-2507` via HF Inference Providers (modifiable avec `JUDGE_MODEL`) |
| Volume | les ~162 000 notices (train, validation et test) |
| Durée | 3 à 4 h sur HF Jobs `cpu-basic` (32 requêtes en parallèle) |
| Reprise | les résultats sont écrits par blocs de 5 000 notices (`verified/*.parquet`) : un job interrompu reprend là où il s'est arrêté |
| En cas d'erreur d'appel | les étiquettes des règles sont gardées telles quelles, et l'erreur est notée dans `judge_error` |

Colonnes ajoutées : `verified_labels`, `judge_rejected`, `judge_added`, `judge_quotes` (les citations), `judge_error`, `judge_model` et `judge_prompt`.

Le LLM n'est pas infaillible. Pour connaître la qualité réelle, il faut relire à la main un échantillon de 100 à 200 notices (§ 12).

## 6. Le fine-tuning (`train_country_model.py`)

| Paramètre | Valeur |
|---|---|
| Modèle de base | [`jhu-clsp/mmBERT-base`](https://huggingface.co/jhu-clsp/mmBERT-base) : encodeur de type ModernBERT pré-entraîné sur plus de 1 800 langues, licence MIT |
| Étiquettes | `verified_labels`, ou les étiquettes des règles si la vérification n'a pas été faite |
| Tête | linéaire 768 → 247, sigmoïde, entropie croisée binaire (`BCEWithLogitsLoss`) |
| Longueur maximale | 512 tokens |
| Époques | 2 |
| Batch | 16 × accumulation 2, soit **32 effectif** |
| Taux d'apprentissage | 3e-5, warmup sur 5 % des pas, décroissance linéaire |
| Weight decay / clipping | 0,01 / 1,0 |
| Précision | bf16 |
| Graine | 42 |
| Matériel | 1 GPU A10G (HF Jobs `a10g-small`), environ 1 h 30 |
| Décision | **seuil unique de 0,5** pour tous les pays |

**Pourquoi un seuil unique ?** Le modèle est entraîné et évalué sur les mêmes étiquettes, et ses scores sont presque toujours très tranchés (proches de 0 ou de 1). Un seuil par pays n'apporterait presque rien et compliquerait l'explication. Cette calibration reste disponible en option (`CALIBRATE=1`) : pour chaque pays, on essaie les seuils de 0,05 à 0,95 sur la partition validation et on garde celui qui fait le moins d'erreurs (F1 maximal).

**Évaluation** sur la partition test, contre `verified_labels` (`metrics.json`) :
- précision : parmi les pays prédits, combien sont justes ;
- rappel : parmi les pays à trouver, combien ont été trouvés ;
- F1 micro et F1 macro (moyenne par pays) ;
- précision du pays le plus probable, sur les notices qui ont au moins un pays ;
- **taux de faux positifs sur les notices sans pays** : la part des notices sans aucun pays qui en reçoivent un par erreur ;
- ventilation par langue et par variante (complet / titre seul) ;
- détail par pays dans `per_label_metrics.json`.

## 7. Résultats

*À compléter après l'entraînement de la v2.1.* Les résultats de la v2 sont dans l'historique (§ 11).

## 8. Lancer le pipeline

```bash
pip install -r requirements_build.txt
python -m pytest tests/
```

```bash
export OPENALEX_API_KEY=...  HF_TOKEN=...
python countries/build_lexicon.py
python build_country_dataset.py        # 30 à 45 min, environ 0,80 $ d'OpenAlex la première fois
```

Puis, sur [HF Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs), depuis un clone du dépôt, **dans cet ordre** :

```bash
hf jobs uv run --flavor cpu-basic --timeout 8h --secrets HF_TOKEN \
  -e DS_REPO=Geraldine/pub-country-classification-dataset-v2.1 \
  judge_labels.py
```

```bash
hf jobs uv run --flavor a10g-small --timeout 6h --secrets HF_TOKEN \
  -e DS_REPO=Geraldine/pub-country-classification-dataset-v2.1 \
  -e MODEL_REPO=Geraldine/pub-country-classifier-v2.1 \
  train_country_model.py
```

Si le premier job s'interrompt, il suffit de le relancer : les blocs déjà vérifiés sont sautés. Le modèle est publié en privé par défaut (`MODEL_PRIVATE=0` pour le rendre public). Pour tester l'entraînement sur CPU, quelques pas sans publication : `SMOKE=1 DS_LOCAL=build python train_country_model.py`.

## 9. Utilisation

```bash
python inference_example.py --repo Geraldine/pub-country-classifier-v2.1 \
  --title "La gestion de l'eau dans le delta du Mékong au Vietnam"
```

```python
import json, torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification

repo = "Geraldine/pub-country-classifier-v2.1"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForSequenceClassification.from_pretrained(repo).eval()
codes = json.load(open(hf_hub_download(repo, "codes.json")))
thr = json.load(open(hf_hub_download(repo, "thresholds.json")))   # 0,5 pour tous les pays

text = "Titre\nRésumé (facultatif)"
with torch.no_grad():
    p = torch.sigmoid(model(**tok(text, truncation=True, max_length=512, return_tensors="pt")).logits)[0]
print([(c, round(float(x), 3)) for c, x in zip(codes, p) if x >= thr[c]])
```

## 10. Limites

- **Supervision faible.** Les étiquettes viennent des règles, vérifiées par un LLM, pas par des annotateurs humains.
- **Mentions explicites seulement.** Un texte sur « Nairobi » qui ne nomme pas le Kenya n'est pas étiqueté `KE`. C'est voulu, mais pour l'indexation, c'est un manque.
- **Mention ≠ sujet.** Un pays cité en passant est détecté au même titre que le terrain d'étude. Un tri éventuel (« porte sur ») est à faire en aval.
- **Langues.** Seules les 5 langues d'entraînement sont couvertes. D'autres langues fonctionneront parfois grâce à mmBERT, sans garantie.
- **Territoires rares.** `PM`, `MP` et `WF` ont peu d'exemples, leur fiabilité est plus faible.
- **Troncature** à 512 tokens.
- **Usage responsable.** C'est un classifieur de texte. Il ne doit pas servir à déduire la nationalité des auteurs ni à prendre des décisions sur des personnes.

## 11. Historique

**v1** (ML Intern) : ModernBERT-base, anglais, 249 codes `pycountry`, F1 0,957 contre ses propres règles. La revue du code a montré des erreurs d'étiquetage systématiques : « Russia », « Turkey » et « Korea » ignorés, la RD Congo étiquetée Congo-Brazzaville, New Jersey étiqueté Jersey. Elles étaient invisibles dans le score, parce que le test était étiqueté par les mêmes règles.

**v2** (20/11/2026), modèle `Geraldine/pub-country-classifier-v2` : le même pipeline que la v2.1, avec deux différences.
- Le LLM jouait le rôle de juge **« porte sur »**, et seulement sur la validation et le test.
- Les seuils par pays étaient calibrés sur ses réponses, alors que le modèle avait appris les règles (« mentionne »).

Résultats sur le test : F1 0,982 contre les règles, avec 0,6 % de faux positifs sur les notices sans pays ; F1 0,850 contre le juge, au même niveau que les règles elles-mêmes (0,861). Ce mélange de deux définitions rendait les résultats difficiles à interpréter. Le juge déduisait aussi parfois un pays de la langue du texte, et les seuils en ont été faussés (Brésil à 0,175, par exemple). La v2 a toutefois révélé les homonymes corrigés dans la v2.1 : *Anguilla anguilla*, Salvador, Granada…

## 12. Pistes

1. Relire à la main 100 à 200 notices pour mesurer la qualité réelle des `verified_labels`.
2. Ajouter au lexique les capitales et les grandes villes (GeoNames ou Wikidata P36), avec des bloqueurs.
3. Ajouter, en aval, un classement des pays mentionnés par importance (titre ou résumé, fréquence) pour approcher la notion de « porte sur ».
4. Comparer avec `mmBERT-small`, plus rapide, et avec XLM-RoBERTa.

## Licence

Code : MIT. Données OpenAlex et Wikidata : CC0.
