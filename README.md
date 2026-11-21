# country-classifier : détecter les pays mentionnés dans une publication scientifique

Ce projet produit un modèle qui lit une publication scientifique et indique **quels pays elle mentionne**. Il prend en entrée le titre, et le résumé quand il existe. Il répond parmi les **247 pays du vocabulaire OpenAlex** ([api.openalex.org/countries](https://api.openalex.org/countries)) : les codes ISO 3166-1 alpha-2, plus `XK` pour le Kosovo.

L'objectif est de **compenser le manque d'indexation géographique** des bases bibliographiques. Le champ « couverture géographique » y est souvent vide ou mal rempli, alors que le titre et le résumé contiennent l'information.

- **Modèle** : [`Geraldine/pub-country-classifier`](https://huggingface.co/Geraldine/pub-country-classifier)
- **Dataset** : [`Geraldine/pub-country`](https://huggingface.co/datasets/Geraldine/pub-country)
- Langues : anglais, français, espagnol, allemand, portugais
- Modèle de base : [`jhu-clsp/mmBERT-base`](https://huggingface.co/jhu-clsp/mmBERT-base), un encodeur multilingue

**En bref.** Sur le jeu de test (7 954 notices, seuil de 0,5), la **précision est de 0,98, le rappel de 0,94 et le F1 de 0,96**. Quand le modèle place un pays en tête, ce pays est juste dans 99 % des cas. Le modèle convient à l'**indexation assistée** : il suggère des pays, qu'une personne valide ou qu'on filtre par le score (§ 7).

### Ce que le modèle détecte : une mention explicite

Le texte nomme le pays par son nom, une variante ou un gentilé : « France », « USA », « Côte d'Ivoire », « Kenyan », « brasileiro ». Le mot doit désigner le pays lui-même, sa population ou quelque chose qui en vient. Les formes historiques ou coloniales comptent (« British India », « le Congo belge »), de même que les territoires dépendants (Martinique, Porto Rico…).

Ne comptent pas :
- les homonymes : une espèce (*Anguilla anguilla*), une ville ou un État du même nom (Salvador de Bahia, New Jersey, la Georgia américaine) ;
- les prénoms ;
- les noms de revue ou de société savante (« *American Journal of…* ») ;
- les noms de langue (« Romanian dialects »).

Le modèle **ne juge pas** si la publication « porte sur » le pays. C'est un choix délibéré : cette notion documentaire est ambiguë, et on ne peut pas l'évaluer sans annotation humaine.

---

## 1. Formulation du problème

Un texte peut mentionner plusieurs pays (« comparaison France–Allemagne ») ou aucun (« une nouvelle architecture de réseau de neurones »). Il s'agit donc d'une **classification multi-label** :

- en entrée, un texte : le titre sur la ligne 1, puis le résumé, facultatif, sur la ligne 2 ;
- en sortie, **un score entre 0 et 1 pour chacun des 247 pays**, calculé indépendamment des autres ;
- **un pays est retenu si son score est supérieur ou égal à 0,5.**

> « Urban agriculture in Kenya and Uganda » → KE 0,99 · UG 0,97 · TZ 0,08 · FR 0,001 … → **KE, UG**

Une liste vide est une réponse normale : la plupart des publications ne mentionnent aucun pays.

## 2. Vue d'ensemble

```
OpenAlex /countries ─┐
Wikidata (5 langues) ─┼─► lexique des pays ─► étiqueteur à règles
corrections manuelles ┘   (data/country_lexicon.json)     │
                                                           ▼
OpenAlex /works : requêtes par pays (5 langues) + échantillon aléatoire
                                                           │
                                                           ▼
          dataset de travail : étiquettes des règles (train 90 % / validation 5 % / test 5 %)
                                                           │
                                                           ▼
   vérification par LLM, étape 1 : chaque notice, chaque étiquette (judge_labels.py)
   vérification par LLM, étape 2 : les étiquettes rejetées à l'étape 1, une à une (recheck_tags.py)
                                                           │
                                                           ▼
   dataset final : étiquettes des règles − mentions fausses + mentions oubliées (finalize_labels.py)
                                                           │
                                                           ▼
          fine-tuning de mmBERT-base ─► évaluation sur le test ─► Hub (train_country_model.py)
```

L'entraînement, la validation et le test utilisent **les mêmes étiquettes**, celles du dataset final.

| Fichier | Rôle |
|---|---|
| `countries/build_lexicon.py` | construit le lexique des alias de pays |
| `countries/overrides.py` | corrections manuelles, documentées |
| `countries/labeler.py` | étiqueteur à règles |
| `tests/test_labeler.py` | tests de non-régression (pièges connus) |
| `build_country_dataset.py` | collecte OpenAlex, étiquetage par les règles, découpage |
| `judge_labels.py` | vérification par LLM, étape 1 |
| `recheck_tags.py` | vérification par LLM, étape 2 |
| `finalize_labels.py` | prépare l'étape 2, fusionne les verdicts, publie le dataset final |
| `train_country_model.py` | fine-tuning, évaluation, publication du modèle |
| `inference_example.py` | exemple d'utilisation |
| `data/eval/judge_sample_reviewed.json` | échantillon de 120 étiquettes ayant servi à évaluer la vérification par LLM |

## 3. Le lexique des pays

### 3.1 Sources

1. **OpenAlex** (`/countries`) fournit, pour chaque pays, le code, le nom d'affichage, quelques variantes et l'identifiant Wikidata. Il y a 247 pays.
2. **Wikidata**, interrogé avec ces identifiants, fournit en anglais, français, espagnol, allemand et portugais :
   - les libellés (« Allemagne », « Deutschland », « Alemania »…) ;
   - les alias (« Burma », « Zaïre », « Holland »…) ;
   - les gentilés (propriété P1549 : « Kenyan », « sénégalais », « brasileño »…).

   Les formes fléchies des gentilés (féminin, pluriel) sont générées automatiquement. Wikidata contient peu de gentilés portugais et espagnols : environ 70 ont été ajoutés à la main (« brasileiro », « angolano », « mexicano »…).
3. **`countries/overrides.py`** contient les corrections manuelles.

Les alias automatiques sont filtrés, car ils sont la principale source de bruit. Sont écartés :
- les codes de 2 à 4 lettres et les abréviations (« All. », « Col. ») ;
- les mots génériques (« States », « Union ») ;
- les surnoms courts (« Cook », « Heard ») ;
- les alias revendiqués par plusieurs pays. Par exemple, « China » est revendiqué par CN et TW, et « Congolese » par CD et CG. Ces alias sont réattribués à la main quand c'est pertinent.

Certains homonymes trop fréquents sont aussi retirés : « Salvador », « Granada » / « Grenade », « Salomon », « Santa Elena ». Les formes sans ambiguïté sont conservées (« El Salvador », « Grenada », « Îles Salomon »).

Le lexique final compte **environ 4 840 alias**. Il est enregistré dans `data/country_lexicon.json`.

### 3.2 Règles d'étiquetage (`countries/labeler.py`)

- **Correspondance sur les frontières de mots Unicode**, sensible à la casse. La seule exception concerne les adjectifs en minuscules (« sénégalais »).
- **La correspondance la plus longue l'emporte.** Tous les candidats sont collectés, puis on garde le plus long en cas de chevauchement. Ainsi, « Papua New Guinea » bat « New Guinea » et « Guinea », « Guinea-Bissau » bat « Guinea », et « Democratic Republic of the Congo » bat « Congo ».
- **Bloqueurs.** Ce sont des expressions qui ressemblent à un pays sans en être un. Elles « consomment » le texte et ne produisent aucune étiquette :
  - lieux : « New Jersey », « New Mexico », « Gulf of Mexico », « Atlanta, Georgia », « Latin America », « Indian Ocean » ;
  - espèces, maladies, termes techniques : « Guinea pig », « Spanish flu », « Chinese hamster », « Jordan curve », « French horn », *« Anguilla anguilla »* ;
  - organisations et revues : « American Journal of… », « British Medical Journal », « German Research Foundation » ;
  - noms de personnes : « Dominique », « Jean Saint-Martin ».
- **Mots ambigus.** Ils sont résolus d'après le contexte :
  - « Congo » donne `CD` si le texte cite Kinshasa, le Kivu, « DRC »…, et `CG` s'il cite Brazzaville, Pointe-Noire… ;
  - « Georgia » donne `GE`, sauf en contexte américain (Atlanta, USA, county…) ;
  - « Macedonia » donne `MK`, sauf en contexte grec ou antique.
- **Gentilés et noms de langue.** En français, en espagnol et en portugais, « le français » ou « el japonés » désignent aussi la langue. Pour les pays concernés, seule la forme avec majuscule (« les Français ») est retenue en dehors de l'anglais.

Les pièges connus sont couverts par des tests (`tests/test_labeler.py`, 66 tests). Quelques exemples :

| Texte | Étiquette produite |
|---|---|
| « Economic growth in Russia » | `RU` |
| « Ebola in the Democratic Republic of the Congo » | `CD` |
| « Malaria in Guinea-Bissau » | `GW` |
| « Les agriculteurs sénégalais… » | `SN` |
| « A agricultura familiar no semiárido brasileiro » | `BR` |
| « Soil survey of New Jersey » | rien |
| « Atlanta, Georgia public schools » | rien |
| « Published in the American Journal of Physics » | rien |
| « Circadian rhythm in the eel Anguilla anguilla » | rien |
| « Urban violence in Salvador, Bahia » | rien |
| « Enseignement du français langue étrangère » | rien |

```bash
python -m pytest tests/
```

## 4. Le dataset

### 4.1 Collecte OpenAlex (`build_country_dataset.py`)

Pour chacun des 247 pays, le script récupère :
- **jusqu'à 400 publications anglophones**, recherchées sur le nom anglais et quelques variantes (« Russia » OR « Russian Federation »…) ;
- **jusqu'à 200 publications en français, espagnol, allemand ou portugais**, avec une seule requête booléenne sur les noms du pays dans ces langues (« Kenia » OR « Quénia » OR « Kénya »…).

S'y ajoute un **échantillon aléatoire d'OpenAlex** (paramètre `sample`) : 24 000 publications dans les cinq langues, plus 4 000 par langue autre que l'anglais. Cet échantillon reflète une vraie base bibliographique, toutes disciplines confondues : **seules 16 % de ces publications nomment un pays**. Ce sont des négatifs réalistes.

Seuls les types `article`, `book`, `book-chapter`, `dissertation`, `preprint`, `review` et `report` sont conservés. Les textes qui nomment plus de 12 pays (listes, formulaires) sont écartés.

Les requêtes servent seulement à **choisir** les publications : **les étiquettes viennent toujours de l'étiqueteur**. Une requête large comme « Georgia » ou « Jordan » ramène donc aussi des textes qui ne parlent pas du pays (l'État américain, la courbe de Jordan…). Ces textes sont conservés avec une liste vide. Ce sont des **négatifs difficiles** : plus de 5 000 dans le train.

**Notices sans résumé.** 15 % des publications qui ont un résumé sont dupliquées en version **titre seul**. Ces copies sont réétiquetées à partir du titre uniquement, pour que le modèle n'apprenne pas à deviner un pays absent du texte.

**Découpage.** Les partitions train / validation / test (90/5/5) sont tirées par **hachage de l'identifiant OpenAlex**. Le découpage est donc stable et reproductible, et une publication reste toujours dans la même partition que sa copie « titre seul ».

Le coût OpenAlex est d'environ **0,80 $**, avec une clé gratuite plafonnée à 1 $/jour. Les pages brutes sont mises en cache dans `.cache/` : relancer le script ne coûte rien.

### 4.2 Vérification par LLM, en deux étapes

Les règles font des erreurs que le lexique ne peut pas toutes prévoir. Elles prennent un homonyme pour un pays, ou ratent un pays écrit d'une façon inattendue, par exemple un titre en majuscules. Un LLM (`Qwen/Qwen3-235B-A22B-Instruct-2507`, via HF Inference Providers) vérifie donc les étiquettes. Il ne juge jamais la pertinence : une mention correcte reste, même en passant.

**Étape 1 : toutes les notices (`judge_labels.py`).** Pour chaque notice, le LLM voit le texte et les pays trouvés par les règles. Il répond à deux questions :
1. quelles étiquettes ne sont **pas** des mentions du pays (homonymes, etc.) ?
2. quels pays sont **nommés explicitement mais oubliés** ? Pour chacun, il doit citer les mots exacts du texte, et le script vérifie qu'ils y figurent. Un pays sans citation vérifiable est ignoré. Le LLM n'ajoute jamais un pays à partir de la langue du texte, d'une ville ou d'une institution.

**Étape 2 : les étiquettes rejetées, une à une (`recheck_tags.py`).** L'étape 1 est fiable pour ce qu'elle garde et ce qu'elle ajoute, mais **trop sévère dans ses rejets**. Elle écarte des territoires (« ce n'est pas un pays »), des noms étrangers ou historiques qu'elle ne relie pas au code (« Écosse » → GB, « British raj »), et des mentions en passant. Chaque étiquette rejetée est donc revue avec un prompt plus précis, qui :
- donne le mot qui a déclenché l'étiquette ;
- précise que les territoires, les anciens noms et les formes historiques ou coloniales comptent ;
- limite les motifs de rejet aux espèces, prénoms, autres lieux du même nom, noms de revue ou de société, noms de langue, noms d'un autre territoire et homonymes ;
- demande un verdict et une courte justification par étiquette.

**Une étiquette n'est retirée que si les deux étapes la rejettent** (`finalize_labels.py`) :

```
étiquettes finales = étiquettes des règles − rejetées par les deux étapes + mentions oubliées (citation vérifiée)
```

**Évaluation de la vérification.** J'ai rejugé à l'aveugle 120 étiquettes de validation et de test : 80 rejetées et 40 gardées par l'étape 1. Ce jugement a été fait par un autre LLM (Claude), pas par un annotateur humain. Sur les 110 cas non ambigus :
- l'étape 1 seule prend la bonne décision dans **72 %** des cas : ses 39 décisions « garder » sont justes, mais environ 40 % de ses rejets sont à tort ;
- avec l'étape 2, on monte à **91 %**.

Les erreurs restantes sont des cas limites (« Bermuda grass », « Mozambique Channel ») et des noms de langue. L'échantillon annoté est dans `data/eval/judge_sample_reviewed.json`.

| | train | validation | test |
|---|---|---|---|
| notices dont les étiquettes ont changé | 10,0 % | 10,3 % | 10,2 % |
| étiquettes des règles retirées | 3,9 % | 3,6 % | 3,8 % |
| pays ajoutés | 11 192 | 655 | 656 |

- **Ajouts.** Environ un quart viennent de textes écrits **en majuscules** (« ISLE OF MAN », « BRASIL », « DR CONGO »), que les règles, sensibles à la casse, ratent. Les autres sont surtout des graphies absentes du lexique.
- **Retraits.** Ce sont surtout des homonymes : espèces (*Gossypium barbadense*), personnes (« Maurice Blanchot », « Edna St. Vincent Millay »), autres lieux (« Saint Helena Parish, Louisiana », « Gulf of Guinea »), sociétés savantes (« Société entomologique de France ») et langues.

### 4.3 Le dataset final

| | train | validation | test |
|---|---|---|---|
| notices | 145 781 | 8 061 | 7 954 |
| part sans pays | 24,1 % | 23,8 % | 24,3 % |

- Répartition par langue (train) : en 91 260, fr 18 294, es 17 239, de 11 132, pt 7 856.
- **Les 247 codes ont des exemples d'entraînement.** Les plus rares sont `PM` (Saint-Pierre-et-Miquelon, 12), `MP` (îles Mariannes du Nord, 35) et `WF` (Wallis-et-Futuna, 36).
- Colonnes :
  - `id`, `text`, `labels` (étiquettes finales) ;
  - `rule_labels`, `llm_removed`, `llm_added`, `llm_added_quotes` ;
  - `language`, `year`, `type`, `origin`, `variant`, `has_abstract`, `has_trap`.

  Leur description détaillée figure dans la fiche du dataset.

## 5. Le fine-tuning (`train_country_model.py`)

| Paramètre | Valeur |
|---|---|
| Modèle de base | [`jhu-clsp/mmBERT-base`](https://huggingface.co/jhu-clsp/mmBERT-base) : encodeur de type ModernBERT pré-entraîné sur plus de 1 800 langues, licence MIT |
| Étiquettes | colonne `labels` du dataset final |
| Tête | linéaire 768 → 247, sigmoïde, entropie croisée binaire (`BCEWithLogitsLoss`) |
| Longueur maximale | 512 tokens |
| Époques | 2 (environ 9 100 pas) |
| Batch | 16 × accumulation 2, soit **32 effectif** |
| Taux d'apprentissage | 3e-5, warmup sur 5 % des pas, décroissance linéaire |
| Weight decay / clipping | 0,01 / 1,0 |
| Précision | bf16 |
| Graine | 42 |
| Matériel | 1 GPU A10G (HF Jobs `a10g-small`), 1 h 24 |
| Décision | **seuil unique de 0,5** pour tous les pays |

**Pourquoi un seuil unique ?** Le modèle est entraîné et évalué sur les mêmes étiquettes, et ses scores sont presque toujours très tranchés, proches de 0 ou de 1. Un seuil par pays n'apporterait presque rien et compliquerait l'explication. La calibration reste disponible en option (`CALIBRATE=1`) : pour chaque pays, on essaie les seuils de 0,05 à 0,95 sur la partition validation et on garde celui qui fait le moins d'erreurs (F1 maximal).

**Évaluation**, sur la partition test (`metrics.json`) :
- précision : parmi les pays prédits, combien sont justes ;
- rappel : parmi les pays à trouver, combien ont été trouvés ;
- F1 micro, et F1 macro (moyenne par pays) ;
- précision du pays le plus probable, sur les notices qui ont au moins un pays ;
- **taux de faux positifs sur les notices sans pays** : part des notices sans pays qui en reçoivent un par erreur ;
- ventilation par langue et par variante (notice complète / titre seul) ;
- détail par pays dans `per_label_metrics.json`.

## 6. Lancer le pipeline

```bash
pip install -r requirements_build.txt
python -m pytest tests/
```

```bash
export OPENALEX_API_KEY=...  HF_TOKEN=...
python countries/build_lexicon.py
python build_country_dataset.py          # 30 à 45 min, environ 0,80 $ OpenAlex la première fois
```

Puis, sur [HF Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs), depuis un clone du dépôt, **dans cet ordre** :

```bash
hf jobs uv run --flavor cpu-basic --timeout 8h --secrets HF_TOKEN judge_labels.py
```

```bash
python finalize_labels.py prepare
```

```bash
hf jobs uv run --flavor cpu-basic --timeout 2h --secrets HF_TOKEN -e INPUT=recheck/input-000.parquet -e OUTPUT=recheck/output-000.parquet recheck_tags.py
```

```bash
python finalize_labels.py merge
```

```bash
python finalize_labels.py publish
```

```bash
hf jobs uv run --flavor a10g-small --timeout 6h --secrets HF_TOKEN train_country_model.py
```

Quelques précisions :
- La commande `recheck_tags.py` est à lancer une fois par fichier `recheck/input-NNN.parquet`.
- L'étape 1 prend 1 h 30 à 2 h sur CPU, et l'étape 2 quelques minutes par fichier. Si un job de l'étape 1 s'interrompt, il suffit de le relancer : les blocs déjà vérifiés sont sautés.
- Le modèle est publié en privé par défaut (`MODEL_PRIVATE=0` pour le rendre public).
- Pour tester l'entraînement sur CPU (quelques pas, sans publication) : `SMOKE=1 DS_LOCAL=build python train_country_model.py`.

## 7. Résultats

Partition test : 7 954 notices, évaluées contre les étiquettes finales avec un seuil unique de 0,5.

| | précision | rappel | F1 micro | F1 macro | précision du 1er pays | faux positifs sur les notices sans pays | n |
|---|---|---|---|---|---|---|---|
| **tout** | **0,977** | **0,943** | **0,960** | 0,955 | 0,992 | 4,7 % | 7 954 |
| anglais | 0,985 | 0,960 | 0,973 | 0,965 | 0,996 | 4,2 % | 4 899 |
| français | 0,964 | 0,909 | 0,936 | 0,903 | 0,991 | 6,3 % | 1 070 |
| espagnol | 0,963 | 0,919 | 0,940 | 0,902 | 0,975 | 6,1 % | 933 |
| allemand | 0,965 | 0,913 | 0,938 | 0,887 | 0,988 | 3,2 % | 618 |
| portugais | 0,949 | 0,902 | 0,925 | 0,864 | 0,980 | 5,6 % | 434 |
| titre seul | 0,978 | 0,955 | 0,967 | 0,962 | 0,988 | 3,1 % | 534 |

**Comment lire ces chiffres :**
- Une **précision de 0,98** signifie que, sur 100 pays proposés, 98 sont des mentions correctes au sens des étiquettes finales. Un **rappel de 0,94** signifie que, sur 100 mentions à trouver, 94 sont trouvées.
- Les étiquettes de référence ne sont pas parfaites : environ 9 % d'erreurs sur l'échantillon évalué (§ 4.2). Une partie des « erreurs » comptées ci-dessus sont donc des prédictions justes que la référence a manquées.
- Le français, l'espagnol et le portugais sont un peu en dessous de l'anglais, qui représente 60 % des données.
- Sur les 246 pays présents dans le jeu de test, 228 ont un F1 ≥ 0,9. Les plus faibles sont des territoires rares, avec 0 à 20 exemples de test (`MP`, `PM`, `JE`, `WF`, `MF`, `IO`, `KN`, `GE`) : leurs scores sont peu fiables.

**Cas concrets** (seuil de 0,5) :

| Texte | Prédiction |
|---|---|
| Ebola outbreak response in the Democratic Republic of the Congo | `CD` |
| Soil survey of New Jersey | rien |
| Published in the American Journal of Physics… | rien |
| Circadian rhythm in the eel Anguilla anguilla | rien |
| Urban violence in Salvador, Bahia | rien |
| MEDICAL LEGISLATION FOR THE ISLE OF MAN | `IM` |
| BRASIL E A ÁFRICA DO SUL NAS PÁGINAS DA REVISTA VEJA | `BR` + `ZA` |
| A agricultura familiar no semiárido brasileiro | `BR` |
| Mulheres angolanas e a guerra civil | `AO` |
| La pauvreté rurale au Sénégal et au Mali | `SN` + `ML` |
| Migrationspolitik in Deutschland und Österreich | `DE` + `AT` |
| Napoleon's exile on Saint Helena | `SH` |
| Enseignement du français langue étrangère à l'université | rien |
| Urban heat islands in Nairobi | `KE` (0,87) |
| Grammar of Romanian dialects | `RO` (0,91) ✗ |

Pour « Nairobi », le modèle généralise à partir de ses connaissances, alors que le Kenya n'est pas nommé. C'est hors de la définition, mais plutôt utile pour l'indexation. « Romanian dialects » est une erreur typique : les noms de langue restent un point faible.

**Verdict.** Le modèle est utilisable pour de l'**indexation assistée**. Pour une indexation entièrement automatique, il faudrait d'abord mesurer la qualité réelle sur un échantillon annoté par des humains.

## 8. Utilisation

```bash
python inference_example.py --title "La gestion de l'eau dans le delta du Mékong au Vietnam"
```

```python
import json, torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification

repo = "Geraldine/pub-country-classifier"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForSequenceClassification.from_pretrained(repo).eval()
codes = json.load(open(hf_hub_download(repo, "codes.json")))
thr = json.load(open(hf_hub_download(repo, "thresholds.json")))   # 0,5 pour tous les pays

text = "Titre\nRésumé (facultatif)"
with torch.no_grad():
    p = torch.sigmoid(model(**tok(text, truncation=True, max_length=512, return_tensors="pt")).logits)[0]
print([(c, round(float(x), 3)) for c, x in zip(codes, p) if x >= thr[c]])
```

## 9. Limites

- **Supervision faible.** Les étiquettes viennent des règles, vérifiées par un LLM. Il n'y a eu aucune annotation humaine.
- **Mentions explicites seulement.** Un texte sur « Nairobi » qui ne nomme pas le Kenya n'est pas étiqueté `KE`. Le modèle le détecte parfois quand même, mais sans garantie.
- **Mention ≠ sujet.** Un pays cité en passant est détecté au même titre que le terrain d'étude. Un éventuel tri « porte sur » est à faire en aval.
- **Points faibles connus.** Les noms de langue et certains homonymes limites.
- **Langues.** Seules les cinq langues d'entraînement sont couvertes. D'autres langues fonctionneront parfois grâce à mmBERT, sans garantie.
- **Territoires rares.** `PM`, `MP` et `WF` ont peu d'exemples.
- **Troncature.** Les textes sont coupés à 512 tokens.
- **Usage responsable.** C'est un classifieur de texte. Il ne doit pas servir à déduire la nationalité des auteurs, ni à prendre des décisions sur des personnes.

## 10. Pistes

1. Faire relire à la main 100 à 200 notices, pour mesurer la qualité réelle des étiquettes et du modèle.
2. Ajouter au lexique les capitales et les grandes villes (GeoNames ou Wikidata P36), avec des bloqueurs.
3. Ajouter en aval un classement des pays mentionnés par importance (présence dans le titre ou le résumé, fréquence), pour se rapprocher de la notion de « porte sur ».
4. Comparer avec `mmBERT-small`, plus rapide, et avec XLM-RoBERTa.

## Licence

Code : MIT. Données OpenAlex et Wikidata : CC0.
