"""Manual corrections applied on top of the automatic OpenAlex + Wikidata lexicon.

Each entry below fixes a problem observed in the v1 dataset or in labeler tests
(tests/test_labeler.py). Keep them small and documented.
"""

# Never use these as aliases, whatever the source (too ambiguous in scholarly text).
GLOBAL_DROP = {
    "America", "Americas", "Columbia", "Guinea", "Congo", "Georgia", "Macedonia", "Jordan",
    "Chad", "Niger", "Turkey", "Dominica", "Man", "Jersey", "Guernsey", "Indies", "Holland",
    "Formosa", "Persia", "Pacific", "China, People's Republic of", "Republic of China",
    "Ceylon", "Soviet Union", "USSR", "Yugoslavia", "Union", "Lao", "Mainland",
    "Korea", "Ireland", "Samoa", "Virgin Islands", "Sudan", "Timor", "Antilles",
    "Saint Martin", "Micronesia",
}

# Generic words that Wikidata lists as aliases ("the States") but that are useless alone.
GENERIC = {
    "states", "staaten", "estados", "états", "etats", "republic", "république", "república", "republik",
    "kingdom", "royaume", "reino", "königreich", "union", "islands", "island", "îles", "islas", "ilhas",
    "federation", "fédération", "federación", "land", "country", "nation", "empire", "commonwealth",
    "america", "amérique", "américa", "amerika", "este", "indio", "india", "georgie", "heard", "saba", "cook",
    "silla", "stria", "liber", "mann", "palpa", "balbo", "balpo", "moris", "viti", "naija", "whallaland",
    "afrance", "franquistan", "murica", "merica", "oncle sam", "pays de l’oncle sam", "uncle sam",
    "dane", "finn", "polin", "ungar", "lette", "farre", "ilaod", "thaïl", "gwada", "vincy", "aussie", "aussies",
    "straya", "stralia", "down under", "new holland", "aussieland", "nouvelle-hollande", "new hollandia",
    "nova hollandia", "brite", "deutsch", "inder", "türke",
}

# Code -> aliases to add (case-sensitive, matched on word boundaries).
ADD = {
    "US": ["United States", "United States of America", "USA", "U.S.A.", "U.S.", "US-American", "America"],
    "GB": ["United Kingdom", "UK", "U.K.", "Great Britain", "Britain", "England", "Scotland", "Wales",
           "Northern Ireland", "Scottish", "Welsh", "Royaume-Uni", "Reino Unido", "Vereinigtes Königreich",
           "Angleterre", "Écosse", "Inglaterra", "Escocia", "Escócia", "Schottland", "British", "Britannique",
           "Britanniques", "Briten"],
    "RU": ["Russia", "Russian Federation", "Russie", "Rusia", "Rússia", "Russland"],
    "TR": ["Turkey", "Türkiye", "Turkiye", "Turquie", "Turquía", "Turquia", "Türkei"],
    "KR": ["South Korea", "Republic of Korea", "Korea", "ROK", "Corée du Sud", "Corea del Sur", "Coreia do Sul",
           "Südkorea", "Corée"],
    "KP": ["North Korea", "DPRK", "Democratic People's Republic of Korea", "Corée du Nord", "Corea del Norte",
           "Coreia do Norte", "Nordkorea"],
    "CD": ["Democratic Republic of the Congo", "Democratic Republic of Congo", "DR Congo", "DR-Congo", "D.R. Congo",
           "DRC", "Congo-Kinshasa", "Zaire", "République démocratique du Congo", "RD Congo", "RDC",
           "República Democrática del Congo", "República Democrática do Congo", "Demokratische Republik Kongo"],
    "CG": ["Republic of the Congo", "Republic of Congo", "Congo-Brazzaville", "Congo Brazzaville",
           "République du Congo", "República del Congo", "República do Congo", "Republik Kongo"],
    "IE": ["Ireland", "Republic of Ireland", "Irlande", "Irlanda", "Irland"],
    "WS": ["Samoa"],
    "SD": ["Sudan", "Soudan", "Sudán", "Sudão"],
    "SS": ["South Sudan", "Southern Sudan", "Soudan du Sud", "Sudán del Sur", "Sudão do Sul", "Südsudan"],
    "TL": ["Timor-Leste", "East Timor", "Timor Leste", "Timor oriental", "Osttimor"],
    "PS": ["Palestine", "State of Palestine", "Palestinian territories", "Occupied Palestinian Territory",
           "West Bank", "Gaza Strip", "Palästina", "Palestina"],
    "VA": ["Vatican", "Vatican City", "Holy See", "Vatican City State", "Vaticano", "Vatikan", "Vatikanstadt"],
    "BN": ["Brunei", "Brunei Darussalam", "Brunéi"],
    "FK": ["Falkland Islands", "Falklands", "Malvinas", "Îles Malouines", "Islas Malvinas", "Ilhas Malvinas",
           "Falklandinseln"],
    "FM": ["Federated States of Micronesia", "FSM"],
    "MF": ["Saint-Martin", "Saint Martin (French part)", "Collectivity of Saint Martin"],
    "SX": ["Sint Maarten"],
    "SJ": ["Svalbard", "Jan Mayen", "Spitsbergen"],
    "GS": ["South Georgia", "South Sandwich Islands", "Géorgie du Sud"],
    "HM": ["Heard Island", "McDonald Islands", "Heard and McDonald Islands"],
    "BQ": ["Bonaire", "Sint Eustatius", "Saba", "Caribbean Netherlands"],
    "VG": ["British Virgin Islands"],
    "VI": ["U.S. Virgin Islands", "US Virgin Islands", "United States Virgin Islands"],
    "XK": ["Kosovo", "Kosova"],
    "AE": ["UAE", "United Arab Emirates", "Emirates"],
    "LA": ["Laos", "Lao PDR", "Lao People's Democratic Republic"],
    "SY": ["Syria"],
    "VN": ["Vietnam", "Viet Nam", "Viêt Nam", "Viêt-Nam"],
    "MM": ["Myanmar", "Burma"],
    "CI": ["Ivory Coast", "Côte d'Ivoire", "Côte d’Ivoire", "Cote d'Ivoire", "Costa de Marfil", "Costa do Marfim",
           "Elfenbeinküste"],
    "CZ": ["Czech Republic", "Czechia"],
    "MK": ["North Macedonia", "Republic of North Macedonia", "Macédoine du Nord", "Macedonia del Norte",
           "Macedônia do Norte", "Nordmazedonien", "FYROM"],
    "SZ": ["Eswatini", "Swaziland"],
    "CV": ["Cape Verde", "Cabo Verde", "Cap-Vert", "Kap Verde"],
    "TW": ["Taiwan"],
    "IR": ["Iran", "Islamic Republic of Iran"],
    "NL": ["Netherlands", "the Netherlands", "Holland"],
    "GN": ["Guinea", "Republic of Guinea", "Guinée", "Guiné", "Guinea-Conakry", "Guinée-Conakry"],
    "GW": ["Guinea-Bissau", "Guinea Bissau", "Guinée-Bissau", "Guiné-Bissau"],
    "GQ": ["Equatorial Guinea", "Guinée équatoriale", "Guinea Ecuatorial", "Guiné Equatorial", "Äquatorialguinea"],
    "PG": ["Papua New Guinea", "Papouasie-Nouvelle-Guinée", "Papúa Nueva Guinea", "Papua-Neuguinea"],
    "NE": ["Niger", "Nigerien", "Nigérien", "Nigériens", "Nigérienne"],
    "TD": ["Chad", "Tchad", "Tschad"],
    "TT": ["Trinidad and Tobago", "Trinidad & Tobago", "Trinidad-Tobago", "Trinidad", "Trinité-et-Tobago", "Trinidad y Tobago"],
    "JO": ["Jordan", "Jordanie", "Jordania", "Jordânia", "Jordanien"],
    "DM": ["Dominica", "Commonwealth of Dominica"],
    "JE": ["Jersey"],
    "GG": ["Guernsey"],
    "IM": ["Isle of Man"],
    "PL": ["Polen"],  # German name, case-sensitive (lowercase "polen" = pollen in Spanish)
    "CN": ["China", "Chinese", "People's Republic of China", "PR China", "P.R. China", "PRC", "Mainland China"],
    "DO": ["Dominican", "Dominican Republic", "République dominicaine", "República Dominicana"],
    "GE": ["Géorgie", "Georgien", "Geórgia", "Sakartvelo"],  # bare "Georgia": see AMBIGUOUS
}

# Code -> aliases to add, case-insensitive (adjectives in fr/es/pt are lowercase).
ADD_CI = {
    "CD": ["congolais de la RDC"],
    "GN": ["guinéen", "guinéenne", "guinéens", "guinéennes"],
}

# Countries whose demonym adjective is also a language name in fr/es/pt/de
# ("le français", "el japonés", "das Deutsche"). For them, non-English demonyms are only
# kept capitalised and case-sensitive (people: "les Français"), never as lowercase adjectives.
LANGUAGE_DEMONYM_CODES = {
    "FR", "ES", "DE", "PT", "IT", "GB", "US", "NL", "GR", "TR", "PL", "RU", "CN", "JP", "KR", "SE", "DK",
    "NO", "FI", "HU", "CZ", "SK", "RO", "BG", "HR", "RS", "AL", "UA", "TH", "VN", "IS", "EE", "LV", "LT",
    "SI", "MN", "AM", "GE", "AZ", "KZ", "UZ", "IR", "IL", "IN", "PK", "BD", "ID", "MY", "PH", "LA", "KH",
    "MM", "NP", "LK", "ET", "SO",
}

# Code -> aliases to remove from the automatic sources.
REMOVE = {
    "PL": ["polen"],
    "CG": ["DRC", "Zaire", "Congo, Democratic Republic of the"],
    "CD": ["Congo"],
    "US": ["Americans", "'Murica", "Amerika", "Unión Americana", "américain", "américaine",
           "americano", "americana", "americanos", "americanas", "Amerikaner", "Amerikanerin", "Amerikanerinnen"],
    "GB": ["English", "anglais", "anglaise", "inglés", "inglesa", "inglês"],
    "CN": ["cn", "Red China", "Communist China"],
    "JP": ["Jap", "Jap.", "Iapan", "Iapon"],
    "IN": ["indio", "india", "indios", "indias", "Inde"],  # "indio/india" ambiguous in es/pt
    "ES": ["español", "española", "españoles", "españolas", "espagnol", "espagnole", "espanhol", "espanhola",
           "Spanier"],
    "FR": ["français", "française", "françaises", "francés", "francesa", "francês"],
    "DE": ["allemand", "allemande", "alemán", "alemana", "alemão", "alemã"],
    "PT": ["portugais", "portugaise", "portugués", "portuguesa", "português"],
    "IT": ["italien", "italienne", "italiano", "italiana"],
}
# Rationale for the fr/es/pt REMOVE entries: in those languages the lowercase adjective is
# also the name of the language ("le français", "el español"), which would label every
# paper about language teaching with a country. English demonyms ("French") keep the same
# ambiguity: the LLM judge is there to catch it.

# Spans that look like a country but are not one. They are matched first and
# "consume" the text, so no country alias can match inside them. Regex, case-sensitive.
BLOCKERS = [
    # US states and other places named after countries
    r"New Mexico", r"New Jersey", r"New South Wales", r"New England", r"New Guinea(?! [A-Z])",
    r"Gulf of Mexico", r"Mexico City(?=, (?:Missouri|MO))",
    r"(?:University of|Georgia Institute of Technology|Georgia Tech|Georgia State|Medical College of|"
    r"Atlanta,?|Savannah,?|Athens,?) ?Georgia(?: (?:Tech|State|Southern))?",
    r"Georgia,? (?:USA|U\.S\.A?\.|United States)", r"Georgian Bay", r"Georgian era", r"Georgian architecture",
    r"Lebanon,? (?:New Hampshire|NH|Pennsylvania|PA|Ohio|Oregon|Tennessee)",
    r"Holland,? (?:Michigan|MI)", r"Peru,? (?:Indiana|Illinois)", r"Paris,? Texas",
    r"Jersey (?:cows?|cattle|heifers?|breed|milk)", r"Guernsey (?:cows?|cattle|breed)",
    # Person names and French place names
    r"Dominique(?! \(|, île| island)", r"Saint-Martin(?=[- ](?:d[eu']|en|le|la|sur|de|du|des|aux?|[A-Z]))",
    r"(?:Dominique|Jean|Marie|Pierre|Paul|Louis|Jacques|Michel|Anne|François|Philippe|Christophe|Nicolas|"
    r"Sophie|Catherine|Isabelle|Claude|André|Henri|Yves|Alain|Bernard|Daniel|Laurent|Olivier)[- ]Saint-Martin",
    # Oceans, regions and continents that contain a country name
    r"Indian Ocean", r"Indian subcontinent", r"(?:Latin|South|North|Central|Pan|Ibero)[- ]Americ(?:a|an|ans|as)",
    r"(?:Amérique|América|Amerika) (?:latine|Latina|del Sur|du Sud|do Sul|del Norte|du Nord|do Norte|Central|centrale)",
    r"Lateinamerika", r"Südamerika", r"Nordamerika", r"Mittelamerika",
    r"(?:the )?Americas", r"Congo Basin", r"Congo River", r"(?:West|East) Indies", r"South China Sea",
    r"East China Sea", r"Irish Sea", r"Dead Sea", r"Arabian (?:Sea|Peninsula)", r"Persian Gulf",
    r"Macedonia,? Greece", r"Greek Macedonia", r"Central Macedonia",
    # Species, diseases, foods and technical terms
    r"[Gg]uinea[- ](?:pigs?|fowls?|worms?|grass)", r"Chinese hamsters?", r"Chinese medicine",
    r"Traditional Chinese Medicine", r"Spanish flu", r"Spanish influenza", r"German measles",
    r"German shepherds?", r"French (?:horns?|fries|press|kiss|toast)", r"Japanese (?:encephalitis|quail|knotweed|beetles?|"
    r"cedar|white-eye|eel|flounder|medaka|macaques?)", r"Brazil nuts?", r"Canada (?:geese|goose|lynx|thistle)",
    r"Swiss (?:cheese|roll|army knife)", r"Dutch (?:oven|elm|auction)", r"Turkish delight", r"Mexican hat",
    r"Chil[ei] (?:peppers?|con carne|powder|sauce)", r"Panama disease", r"Russian doll", r"Russian roulette", r"Indian (?:ink|summer)",
    r"Jordan(?:'s)? (?:curve|normal form|algebras?|blocks?|decomposition|canonical form|measure|lemma|"
    r"product|chains?|form|matrix|matrices|theorem|type)", r"Jordan[- ]H[öo]lder", r"Jordan[- ]Wigner",
    r"Jordan[- ]Brans[- ]Dicke", r"Jordan frame",
    r"Greek (?:letters?|alphabet|yogh?urt)",
    # Organisations and journals (a mention, not the topic of the paper)
    r"American (?:Journal|Society|Chemical Society|Psychological Association|Heart Association|College|"
    r"Academy|Medical Association|Physical Society|Mathematical Society|Statistical Association|Cancer Society|"
    r"Diabetes Association|Thoracic Society|Association|Institute of Physics|Economic Review|Sociological Review|"
    r"Political Science Review|Psychiatric Association|Geophysical Union|Meteorological Society)",
    r"British (?:Journal|Medical Journal|Society|Academy|Library|Museum)",
    r"Indian Journal", r"Chinese Journal", r"Japanese Journal", r"German Research Foundation", r"French National",
]

# Bare words that may or may not mean a given country. Rules are tested in order;
# the first rule whose condition holds gives the label, else no label.
# "require_any": one of these strings must appear in the text; "forbid_any": none may appear.
AMBIGUOUS = {
    "Congo": [
        {"code": "CD", "require_any": ["Kinshasa", "Kivu", "Katanga", "Kasai", "Ituri", "Lubumbashi", "Goma",
                                        "Bukavu", "Kisangani", "Mbuji", "Zaire", "DRC", "RDC", "Democratic"]},
        {"code": "CG", "require_any": ["Brazzaville", "Pointe-Noire", "Republic of the Congo", "République du Congo"]},
    ],
    "Congolese": [
        {"code": "CD", "require_any": ["Kinshasa", "Kivu", "Katanga", "Kasai", "Ituri", "Lubumbashi", "Goma",
                                        "Bukavu", "Kisangani", "Zaire", "DRC", "Democratic"]},
        {"code": "CG", "require_any": ["Brazzaville", "Pointe-Noire", "Republic of the Congo"]},
    ],
    "Congolais": [
        {"code": "CD", "require_any": ["Kinshasa", "Kivu", "Katanga", "Kasaï", "Lubumbashi", "Goma", "Bukavu", "RDC", "démocratique"]},
        {"code": "CG", "require_any": ["Brazzaville", "Pointe-Noire", "République du Congo"]},
    ],
    "Macedonia": [
        {"code": "MK", "forbid_any": ["Greece", "Greek", "Thessaloniki", "Grèce", "Grecia", "Griechenland",
                                       "Alexander", "ancient", "Ancient", "Hellenistic", "Philip"]},
    ],
    "Georgia": [
        {"code": "GE", "forbid_any": ["Atlanta", "USA", "U.S.", "United States", "Alabama", "Florida",
                                       "South Carolina", "Tennessee", "Savannah", "Augusta", "Athens, GA", "state of Georgia",
                                       "State of Georgia", "Georgia State", "southeastern", "Southeastern", "county", "County"]},
    ],
}
