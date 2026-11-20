# country-classifier : détecter les pays dont parle une publication scientifique

Ce projet produit un modèle qui lit une publication scientifique et indique **de quels pays elle parle**. Il prend en entrée le titre, avec le résumé s'il existe, et répond parmi les **247 pays du vocabulaire OpenAlex** ([api.openalex.org/countries](https://api.openalex.org/countries)) : codes ISO 3166-1 alpha-2, plus `XK` pour le Kosovo.

L'objectif est de **compenser le manque d'indexation géographique** des bases bibliographiques. Le champ « couverture géographique » est souvent vide ou mal rempli, alors que le titre et le résumé contiennent l'information.

| | v1 (ML Intern) | v2 (ce dépôt) |
|---|---|---|
| Liste de pays | 249 codes `pycountry` | **247 codes OpenAlex** |
| Langues | anglais | **anglais, français, espagnol, allemand, portugais** |
| Modèle de base | ModernBERT-base | **mmBERT-base** (multilingue) |
| Étiquetage | regex sur les noms `pycountry` | **lexique OpenAlex + Wikidata**, bloqueurs, règles de contexte, tests |
| Négatifs | 30 thèmes STEM | **échantillon aléatoire d'OpenAlex** + négatifs difficiles |
| Évaluation | contre les règles | contre les règles **et contre un LLM juge** |

- Dataset v2 : `Geraldine/pub-country-classification-dataset-v2` (privé pour l'instant)
- Modèle v2 : `Geraldine/pub-country-classifier-v2` (privé pour l'instant), entraîné le 20/11/2026

**En bref.** Sur le jeu de test, le modèle reproduit l'étiquetage par règles avec un **F1 de 0,98** et une précision de 0,99. Moins de **1 %** des notices sans pays reçoivent un pays par erreur. Jugé par un LLM sur la question « la publication porte-t-elle sur ce pays ? », il obtient un **F1 de 0,85**, au même niveau que les règles elles-mêmes (0,86). Les résultats détaillés et leur lecture sont au § 6.1.

**Statut.** Le modèle est utilisable pour de l'**indexation assistée** : suggestion de pays, validée par un humain ou filtrée par le score. Il n'est pas encore prêt pour une indexation automatique sans contrôle (voir § 9).
- v1 : [modèle](https://huggingface.co/Geraldine/pub-country-classifier) et [dataset](https://huggingface.co/datasets/Geraldine/pub-country-classification-dataset)

---

## 1. Formulation du problème

On ne cherche pas *le* pays d'une publication : un article peut porter sur plusieurs pays (« comparaison France–Allemagne ») ou sur aucun (« une nouvelle architecture de réseau de neurones »). Il s'agit donc d'une **classification multi-label** :

- en entrée, un texte : le titre sur la ligne 1, le résumé, facultatif, sur la ligne 2 ;
- en sortie, 247 probabilités indépendantes, une par pays ;
- un pays est retenu si sa probabilité dépasse **un seuil propre à ce pays**, calibré sur la partition de validation.

Une liste vide est une réponse normale : la plupart des publications ne parlent d'aucun pays.

## 2. Vue d'ensemble

```
OpenAlex /countries ─┐
Wikidata (5 langues) ─┼─► lexique des pays ─► labeler à règles ─┐
overrides manuels ───┘    (data/country_lexicon.json)          │
                                                                ▼
OpenAlex /works : requêtes par pays (5 langues) + échantillon aléatoire
                                                                │
                                                                ▼
                    dataset v2 (train 90 % / validation 5 % / test 5 %)
                                                                │
                      LLM juge sur validation + test (judge_labels.py)
                                                                │
                                                                ▼
          fine-tuning mmBERT-base ─► seuils par pays ─► évaluation ─► Hub
```

| Fichier | Rôle |
|---|---|
| `countries/build_lexicon.py` | construit le lexique des alias de pays |
| `countries/overrides.py` | corrections manuelles, documentées |
| `countries/labeler.py` | étiqueteur à règles |
| `tests/test_labeler.py` | tests de non-régression (pièges connus) |
| `build_country_dataset.py` | collecte OpenAlex, étiquetage, découpage, publication |
| `judge_labels.py` | validation des étiquettes par un LLM juge |
| `train_country_model.py` | fine-tuning, calibration, évaluation, publication |
| `inference_example.py` | exemple d'utilisation |

## 3. Le lexique des pays

### 3.1 Sources

1. **OpenAlex** (`/countries`) fournit les 247 codes, le nom d'affichage, quelques variantes et l'identifiant Wikidata de chaque pays.
2. **Wikidata**, via ces identifiants, apporte en anglais, français, espagnol, allemand et portugais :
   - les libellés (« Allemagne », « Deutschland », « Alemania »…) ;
   - les alias (« Burma », « Zaïre », « Holland »…) ;
   - les gentilés (propriété P1549 : « Kenyan », « sénégalais », « brasileño »…).
   Pour les gentilés, des formes fléchies sont générées (féminin, pluriel).
3. **`countries/overrides.py`** contient les corrections manuelles.

Les alias automatiques sont filtrés, car c'est la principale source de bruit. Sont écartés :
- les codes de 2 à 4 lettres et les abréviations (« All. », « Col. ») ;
- les mots génériques (« States », « Union ») ;
- les surnoms courts (« Cook », « Heard ») ;
- les alias revendiqués par plusieurs pays : « China » est revendiqué par CN et TW, « Congolese » par CD et CG. Ces alias sont réattribués à la main quand c'est pertinent.

Résultat : **environ 4 700 alias** pour 247 pays, dans `data/country_lexicon.json`.

### 3.2 Règles d'étiquetage (`countries/labeler.py`)

- **Correspondance sur les frontières de mots Unicode**, sensible à la casse. Les adjectifs en minuscules (« sénégalais ») sont la seule exception.
- **La plus longue correspondance l'emporte.** On collecte tous les candidats puis on garde le plus long en cas de chevauchement : « Papua New Guinea » bat « New Guinea » et « Guinea », « Guinea-Bissau » bat « Guinea », « Democratic Republic of the Congo » bat « Congo ».
- **Les bloqueurs** sont des expressions qui ressemblent à un pays sans en être un. Elles « consomment » le texte et ne produisent aucune étiquette. Exemples :
  - lieux : « New Jersey », « New Mexico », « Gulf of Mexico », « Atlanta, Georgia », « Latin America », « Indian Ocean » ;
  - espèces, maladies, termes techniques : « Guinea pig », « Spanish flu », « Chinese hamster », « Jordan curve », « French horn » ;
  - organisations et revues : « American Journal of… », « British Medical Journal », « German Research Foundation » ;
  - noms de personnes : « Dominique », « Jean Saint-Martin ».
- **Les mots ambigus** sont résolus par le contexte :
  - « Congo » donne `CD` si le texte cite Kinshasa, Kivu, DRC…, et `CG` s'il cite Brazzaville, Pointe-Noire… ;
  - « Georgia » donne `GE`, sauf en contexte américain (Atlanta, USA, county…) ;
  - « Macedonia » donne `MK`, sauf en contexte grec ou antique.
- **Gentilés et noms de langue.** En français, espagnol et portugais, « le français » ou « el japonés » désignent aussi la langue. Pour les pays concernés, seule la forme à majuscule (« les Français ») est retenue en dehors de l'anglais.

Chaque erreur relevée lors de la revue de la v1 est devenue un test (`tests/test_labeler.py`, 54 cas) :

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

```bash
python -m pytest tests/
```

## 4. Le dataset (`build_country_dataset.py`)

### 4.1 Collecte OpenAlex

Pour chacun des 247 pays :
- **jusqu'à 400 publications anglophones**, cherchées sur le nom anglais et quelques variantes (« Russia » OR « Russian Federation »…) ;
- **jusqu'à 200 publications en fr/es/de/pt**, avec une seule requête booléenne sur les noms du pays dans ces langues (« Kenia » OR « Quénia » OR « Kénya »…).

S'y ajoute un **échantillon aléatoire d'OpenAlex** (paramètre `sample`) : 24 000 publications dans les 5 langues, plus 4 000 par langue hors anglais. Cet échantillon reflète une vraie base bibliographique, toutes disciplines confondues : **seules 16 % de ces publications nomment un pays**. Ce sont des négatifs réalistes.

Seuls les types `article`, `book`, `book-chapter`, `dissertation`, `preprint`, `review` et `report` sont retenus. Les fiches de données (GBIF…), les pages de menus et autres scories sont écartées. Les textes qui nomment plus de 12 pays (listes, formulaires) sont aussi supprimés : il y en a 673.

Les requêtes ne servent qu'à **choisir** les publications : **les étiquettes viennent toujours du labeler**. Une requête large comme « Georgia » ou « Jordan » ramène donc aussi des textes sans le pays (l'État américain, la courbe de Jordan…). Ces textes sont conservés avec une liste vide : ce sont des **négatifs difficiles**, et il y en a environ 7 000.

### 4.2 Notices sans résumé

15 % des publications avec résumé sont dupliquées en version **titre seul**. Contrairement à la v1, ces copies sont **réétiquetées à partir du titre seul** : le modèle n'apprend pas à deviner un pays absent du texte.

### 4.3 Découpage

Le découpage 90/5/5 se fait par **hachage de l'identifiant OpenAlex**. Il est stable et reproductible, et une publication reste toujours dans la même partition que sa copie « titre seul ».

### 4.4 Chiffres

| | train | validation | test |
|---|---|---|---|
| lignes | 146 258 | 8 058 | 8 021 |
| part sans pays | 27 % | 28 % | 28 % |

- Langues (train) : en 91 208, fr 18 421, es 17 545, de 11 201, pt 7 883.
- **Les 247 codes ont des exemples.** En v1, `FK`, `MF`, `SX` et `VA` n'en avaient aucun.
- Seuls deux codes ont moins de 50 exemples : `PM` (Saint-Pierre-et-Miquelon, 17) et `WF` (Wallis-et-Futuna, 29).
- Exemples de corrections par rapport à la v1 : RD Congo 567 exemples (269 en v1, et la plupart étiquetés `CG` à tort), Russie 1 567, Turquie 760, Palestine 752, Kosovo 420.

Colonnes : `id`, `text`, `labels`, `language`, `year`, `type`, `origin` (`country_query` ou `random`), `variant` (`full` ou `title_only`), `has_abstract` et `has_trap` (le texte contient un bloqueur ou un mot ambigu non résolu).

Le budget OpenAlex est d'environ **0,80 $** (clé gratuite, 1 $/jour). Les pages brutes sont mises en cache (`.cache/`), donc une relance ne coûte rien.

## 5. Le LLM juge (`judge_labels.py`)

Des règles détectent des **mentions**. Or, pour l'indexation, on veut savoir si la publication **porte sur** un pays. Un LLM sert de juge sur les partitions validation et test. Pour chaque notice, il voit le titre, le résumé et les pays proposés par les règles. Il doit :
- **garder** les pays dont la publication traite réellement : terrain, données, cas, population, politique publique, comparaison ;
- **rejeter** les simples mentions : revue, organisation, financeur, langue, espèce, maladie, État américain, exemple en passant ;
- **ajouter** les pays manqués, par exemple nommés via une ville ou une région.

Le résultat est publié comme configuration `judged` du dataset, avec les colonnes `judge_labels`, `judge_removed` et `judge_added`.

- Modèle par défaut : `Qwen/Qwen3-235B-A22B-Instruct-2507` via HF Inference Providers (modifiable avec `JUDGE_MODEL`).
- Durée : environ 20 minutes sur HF Jobs (`cpu-basic`) pour 16 079 notices, sans aucune erreur d'appel.

**Ce que le juge a trouvé**, sur 8 021 notices de test :

| | |
|---|---|
| Accord exact règles / juge (même liste de pays) | 85 % |
| Pays des règles **rejetés** par le juge | 7 %, soit une précision des règles d'environ 0,93 |
| Pays **ajoutés** par le juge | ~1 800, soit un rappel des règles d'environ 0,80 |

Les rejets sont pour la plupart justifiés. Ils ont révélé de vrais défauts du lexique : *Anguilla anguilla* (l'anguille) pris pour Anguilla, Salvador de Bahia pris pour le Salvador, Granada / Grenade (la ville espagnole), « Salomon » ou « Santa Elena » pris pour des pays, et des gentilés hors sujet (« British » dans un texte sur l'Eswatini colonial). Les ajouts sont plus discutables. Environ un quart correspond au **pays de la langue du texte** : un texte en portugais devient « Brésil » sans indice explicite. Le juge en ajoute aussi parfois trop : pour un article sur l'anguille européenne, il ajoute dix pays européens.

Le juge **ne remplace pas une annotation humaine**. Il est lui-même un modèle et peut se tromper. Avant de citer les chiffres contre le juge comme « vérité », il est recommandé d'en contrôler à la main un échantillon de 100 à 200 notices.

## 6. Le fine-tuning (`train_country_model.py`)

| Paramètre | Valeur |
|---|---|
| Modèle de base | [`jhu-clsp/mmBERT-base`](https://huggingface.co/jhu-clsp/mmBERT-base) : encodeur de type ModernBERT pré-entraîné sur plus de 1 800 langues, licence MIT |
| Tête | linéaire 768 → 247, sigmoïde, entropie croisée binaire (`BCEWithLogitsLoss`) |
| Longueur maximale | 512 tokens |
| Époques | 2 |
| Batch | 16 × accumulation 2, soit **32 effectif** |
| Taux d'apprentissage | 3e-5, warmup 5 %, décroissance linéaire |
| Weight decay / clipping | 0,01 / 1,0 |
| Précision | bf16 |
| Graine | 42 |
| Matériel | 1 GPU A10G (HF Jobs, `a10g-small`) : 9 142 pas, **1 h 32** au total |

**Calibration.** Pour chaque pays, on cherche sur la partition validation le seuil (de 0,05 à 0,95, par pas de 0,025) qui maximise le F1. La référence est `judge_labels` si la configuration `judged` existe, sinon les étiquettes des règles. Les pays qui ont moins de 3 exemples positifs en validation gardent un seuil fixe de 0,5.

**Évaluation**, sur la partition test (`metrics.json`) :
- précision, rappel, F1 micro et F1 macro ;
- **précision du pays le plus probable**, sur les notices qui ont au moins un pays (la métrique « top-1 accuracy » de la v1, mal définie, a été supprimée) ;
- **taux de faux positifs sur les notices sans pays**, une mesure essentielle pour un usage réel ;
- ventilation par langue et par variante (complet / titre seul) ;
- comparaison avec les règles **et** avec le juge, ainsi que l'accord entre les règles et le juge.

### 6.1 Résultats (partition test, 8 021 notices, seuils calibrés)

| Référence | Précision | Rappel | F1 micro | F1 macro | Précision du 1er pays | Faux positifs sur les notices sans pays |
|---|---|---|---|---|---|---|
| Étiquettes des règles | **0,994** | 0,970 | **0,982** | 0,973 | 0,999 | **0,6 %** |
| Étiquettes du LLM juge | 0,930 | 0,782 | 0,850 | 0,846 | 0,899 | 5,6 % |
| *Pour comparaison : les règles elles-mêmes face au juge* | *0,930* | *0,802* | *0,861* | *0,862* | – | *5,1 %* |

Par langue (contre les règles) :

| | en | fr | es | de | pt |
|---|---|---|---|---|---|
| F1 micro | 0,988 | 0,970 | 0,974 | 0,974 | 0,957 |
| Faux positifs sur les notices sans pays | 0,5 % | 1,0 % | 0,7 % | 0,7 % | 0,5 % |

Les notices « titre seul » sont aussi bien traitées que les notices complètes (F1 0,992 contre 0,981).

**Comment lire ces chiffres :**
- **Contre les règles**, le modèle a appris l'étiquetage à règles presque parfaitement et de façon très prudente. Il se trompe rarement par excès : 0,6 % de faux positifs sur les notices sans pays.
- **Contre le juge**, le modèle fait à peu près jeu égal avec les règles dont il a appris. Il hérite donc de leurs limites, surtout un rappel d'environ 0,78. Ce rappel est en partie sous-estimé, parce que le juge ajoute des pays déduits de la langue (§ 5).
- On ne peut pas comparer ces chiffres directement au F1 de 0,957 de la v1 : le jeu de test, la liste de pays et les étiquettes ont changé. La comparaison qualitative ci-dessous est plus parlante.
- Les pays les moins bien reconnus sont des territoires rares : `MP`, `PM`, `KN`, `WF` et `IO`. Ils n'ont que 0 à 6 exemples dans le jeu de test, donc leurs scores sont peu fiables.

**Comparaison v1 / v2 sur des cas concrets**, aux seuils calibrés :

| Texte | v1 | v2 |
|---|---|---|
| Ebola outbreak response in the Democratic Republic of the Congo | `CG` ✗ | `CD` ✓ |
| Soil survey of New Jersey | `JE` ✗ | rien ✓ |
| Earthquake damage assessment in Turkey | rien ✗ | `TR` ✓ |
| Published in the American Journal of Physics: a new laser cooling method | `US` ✗ | rien ✓ |
| La pauvreté rurale au Sénégal et au Mali | `ML` | `SN` + `ML` ✓ |
| Enseignement du français langue étrangère à l'université | `FR` ✗ | rien ✓ |
| Migrationspolitik in Deutschland und Österreich | `ES` + `NL` ✗ | `DE` + `AT` ✓ |
| Bibliothèques universitaires et science ouverte en Afrique de l'Ouest | `FR` ✗ | rien ✓ |
| Políticas de salud pública en Colombia durante la pandemia | `CO` ✓ | `CO` ✓ |
| Circadian rhythm in the eel *Anguilla anguilla* | `AI` ✗ | `AI` ✗ |
| A agricultura familiar no semiárido brasileiro | `BR` (0,56) | rien ✗ |
| Urban heat islands in Nairobi | rien ✗ | rien ✗ |

Les trois échecs de la v2 viennent du lexique : un bloqueur *Anguilla anguilla* manque, le gentilé portugais « brasileiro » est absent de Wikidata, et il n'y a pas de villes dans le lexique. Ils sont traités dans les pistes (§ 10).

## 7. Lancer le pipeline

```bash
pip install -r requirements_build.txt
python -m pytest tests/
```

```bash
export OPENALEX_API_KEY=...  HF_TOKEN=...
python countries/build_lexicon.py
python build_country_dataset.py        # 30 à 45 min, environ 0,80 $ OpenAlex
```

Puis, sur [HF Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs), depuis un clone du dépôt :

```bash
hf jobs uv run --flavor cpu-basic --timeout 4h --secrets HF_TOKEN \
  -e DS_REPO=Geraldine/pub-country-classification-dataset-v2 \
  judge_labels.py
```

```bash
hf jobs uv run --flavor a10g-small --timeout 6h --secrets HF_TOKEN \
  -e DS_REPO=Geraldine/pub-country-classification-dataset-v2 \
  -e MODEL_REPO=Geraldine/pub-country-classifier-v2 \
  train_country_model.py
```

Le modèle est publié en privé par défaut (`MODEL_PRIVATE=0` pour le rendre public). Pour tester le script d'entraînement sur CPU (quelques pas, sans publication), on peut lancer `SMOKE=1 DS_LOCAL=build python train_country_model.py`.

## 8. Utilisation

```bash
python inference_example.py --title "La gestion de l'eau dans le delta du Mékong au Vietnam"
```

```python
import json, torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification

repo = "Geraldine/pub-country-classifier-v2"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForSequenceClassification.from_pretrained(repo).eval()
codes = json.load(open(hf_hub_download(repo, "codes.json")))
thr = json.load(open(hf_hub_download(repo, "thresholds.json")))

text = "Titre\nRésumé (facultatif)"
with torch.no_grad():
    p = torch.sigmoid(model(**tok(text, truncation=True, max_length=512, return_tensors="pt")).logits)[0]
print([(c, round(float(x), 3)) for c, x in zip(codes, p) if x >= thr[c]])
```

## 9. Limites

- **Supervision faible.** Les étiquettes d'entraînement viennent des règles, qui détectent des mentions. Le juge n'intervient que sur la validation et le test, sauf si on le lance aussi sur le train (`JUDGE_SPLITS=train`, voir § 10).
- **Peu d'inférence implicite.** Le lexique ne contient pas de villes ni de régions. Un texte sur « Nairobi » sans « Kenya » n'est pas étiqueté `KE` par les règles, même si le juge peut l'ajouter.
- **Langues.** Seules les 5 langues d'entraînement sont couvertes. D'autres langues fonctionneront parfois grâce à mmBERT, sans garantie.
- **Gentilés anglais.** « French » reste ambigu (le peuple ou la langue) : c'est au juge de trier.
- **Territoires rares.** `PM` et `WF` ont peu d'exemples, leur fiabilité est plus faible.
- **Erreurs connues du lexique**, repérées par le juge et pas encore corrigées : *Anguilla anguilla* → `AI` ; Salvador (Bahia) → `SV` ; Granada / Grenade → `GD` ; « Salomon » → `SB` ; « Santa Elena » → `SH` ; « Savage » → `NU`. Il manque aussi des gentilés portugais (« brasileiro/brasileira »…), ce qui explique le score plus faible en portugais.
- **Troncature** à 512 tokens.
- **Usage responsable.** C'est un classifieur de sujets de texte. Il ne doit pas servir à déduire la nationalité des auteurs ni à prendre des décisions sur des personnes.

## 10. Pistes

0. **v2.1** : corriger les erreurs connues du lexique (§ 9), ajouter les gentilés portugais, durcir la consigne du juge (jamais d'ajout déduit de la langue seule), relancer le juge, recalibrer les seuils puis réentraîner.
1. Faire annoter à la main 300 à 500 notices tirées de votre base (HAL, SUDOC…), pour une vraie mesure de qualité et pour valider le juge.
2. Lancer le juge sur une partie du train et entraîner sur `judge_labels`, pour apprendre « porte sur » plutôt que « mentionne ».
3. Ajouter au lexique les capitales et les grandes régions (GeoNames ou Wikidata P36/P150), avec des bloqueurs.
4. Comparer avec `mmBERT-small`, plus rapide, et avec XLM-RoBERTa.

## Licence

Code : MIT. Données OpenAlex et Wikidata : CC0.
