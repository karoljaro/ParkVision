# Occupied/free: rozszerzenie treningu o ACPDS train

2026-11-08. Kontynuacja Dataset Preparation + Model Improvement v1.
Dotychczasowy model trenował wyłącznie na jednym parkingu UFPR. Nowy zbiór
zachowuje te dane i dodaje **cały oficjalny train ACPDS**. Format zadania nie
zmienia się: obraz konkretnego ROI + maska jego obrysu + źródłowa etykieta
empty/occupied. Nie wykorzystujemy anotacji linii do etykietowania zajętości.

Artefakty są poza repozytorium, w:

`/home/karoljaron/Dokumenty/ChatGPT/System parkingowy/dataset-model-improvement-2026-11-08-v1/expansion-multisite-v1/`

## Rzeczywiście przygotowane dane

| Rola / źródło | Obrazy | Empty | Occupied | ROI |
| --- | ---: | ---: | ---: | ---: |
| train, dotychczasowy PKLot UFPR04/05 | 48 | 1067 | 755 | 1822 |
| train, nowy oficjalny ACPDS train | 231 | 3902 | 3940 | 7842 |
| **train łącznie** | **279** | **4969** | **4695** | **9664** |
| val, zachowany ACPDS valid | 35 | 1073 | 831 | 1904 |

Pozyskano 230 brakujących obrazów; jeden oryginalny obraz train był już
zapisany w rezerwie poprzedniego audytu. Walidacyjne obrazy, tablice i rekordy
manifestu odziedziczono bez zmiany bajtów. Archiwów Roboflow nie pobierano.
Nie pobierano ani nie dekodowano żadnego obrazu ACPDS test ani PUCPR.

ACPDS pochodzi z [oficjalnego wydania Martina Marka](https://github.com/martin-marek/parking-space-occupancy).
Autor [w publikacji](https://arxiv.org/abs/2107.12207) wprost udostępnia dane,
kod i modele na MIT oraz opisuje rozłączne parkingi oficjalnych train/valid/test.
Ta deklaracja jest podstawą grupowania ACPDS, bez losowania obrazów między
rolami. Sam JSON nie podaje pełnych parking/session IDs. Nie utożsamiamy 231
obrazów z 231 parkingami ani identyfikatora `ACPDS-train` z odrębnym fizycznym
urządzeniem: to zachowawcza grupa widoków danego oficjalnego splitu.
Geometria każdego ROI jest związana z konkretnym zdjęciem; brak potwierdzonego
globalnego mapowania miejsc między widokami (`physical_slot_id=null`).

Przypięte wejścia:

- anotacje ACPDS SHA-256:
  `a3c1be0dff4eb6b6ab42e64adad650f6ab245463363b8f9b58436d7d16ac0156`;
- lokalny indeks ZIP SHA-256:
  `337605404f88fd7d9cc7bb2b9024a2fad16a0eeddef7c46557c9cbc7bc2ec8d3`;
- archiwum 380796292 bajtów, ETag `14c649b0b87549b3ab31d1c7d0ca875d-46`;
- nowy manifest SHA-256:
  `30e7adc215dbdbf9501b27ea3d554857716eb5ffdd3373087e08d25463c9ec74`.

Pobierane są wyłącznie człony train, zakresami z tego samego publicznego ZIP-a.
Każdy człon przechodzi kontrolę nazwy, rozmiaru i CRC z przypiętego indeksu,
a lokalne bajty, wycinek i jego piksele mają SHA-256.
`expand.py` ponownie sprawdza oryginalne anotacje, role i integralność całej puli.
Dokładne duplikaty bajtów i zdekodowanych obrazów są odrzucane globalnie,
także gdy plik został zakodowany ponownie. Identyczne wycinki są blokowane.

Przegląd 314 obrazów wykazał wyłącznie EXIF orientation=1. Nie ma kandydatów
bliskich duplikatów między train i val według 256-bit dHash, Hamming <=6.
Nie jest to dowód pełnej niezależności lokalizacji; zachowujemy osobno
deklarację wydawcy o fizycznych parkingach i ograniczenia metadanych.
Test chronią allowlista, kontrola rozłączności nazw wszystkich oficjalnych
splitów oraz znane hashe wcześniejszych testowych obrazów, bez ich dekodowania.

Kontrola wizualna 18 deterministycznie dobranych wycinków train (9 na klasę,
z różnych zdjęć) jest zapisana w `acpds-train-annotation-check.jpg` i towarzyszącym
JSON-ie. Pokazuje rzeczywiste stanowiska z autami i bez aut, a także silny skos
i częściowe zasłonięcia. Nie jest pełnym audytem GT i nie zmienia źródłowych
etykiet. W szczególności puste miejsce ukryte pod roślinnością nie jest wizualnym
potwierdzeniem free: `visibility=unassessed` pozostaje przy każdym rekordzie.

## Trening i odtwarzanie

Wybrana architektura nadal to ConvNeXt Tiny ImageNet1K V1, RGB + kanał maski ROI.
Nowy run rozpoczyna fine-tuning z oficjalnych wag ImageNet, z nowym optymalizatorem
i seed=42. Nie jest wznowieniem stanu optymalizatora poprzedniego runu na innym
zbiorze. Wcześniejsze checkpointy UFPR, wyniki i manifesty pozostają zachowane.

Konfiguracja `multisite.json`: 20 epok, batch 16, AdamW, backbone LR=2e-5,
head LR=5e-4, weight decay 0.05, label smoothing 0.02. Każda epoka ma 604 kroki,
pełny backbone jest uczony. Wagi klas wynikają wyłącznie z nowego train.
Kryterium wyboru to najmniejsza walidacyjna cross entropy. Stałych progów
0.1/0.9 nie stroimy podczas treningu; model nie ma automatycznie potwierdzonej
widoczności ani kalibracji. Kontrakt `unknown` i obserwator pozostają zachowane.

```bash
cd /home/karoljaron/Projekty/ParkVision
python_ml="$PWD/research/paint-segmentation/.venv/bin/python"
workspace='/home/karoljaron/Dokumenty/ChatGPT/System parkingowy'
work="$workspace/dataset-model-improvement-2026-11-08-v1"

# Ponowne wywołanie sprawdza istniejący wynik, bez ponownego pobierania.
"$python_ml" -m research.occupancy.training_v1.expand \
  --workspace "$workspace" --base-data "$work/data" \
  --output "$work/expansion-multisite-v1/data"

"$python_ml" -m research.occupancy.training_v1.data \
  --output "$work/expansion-multisite-v1/data" --verify-only

# Odtworzenie wymaga nowego katalogu runu; nie nadpisuje istniejących wag.
CUBLAS_WORKSPACE_CONFIG=:4096:8 "$python_ml" -m research.occupancy.training_v1.train \
  --config research/occupancy/training_v1/multisite.json \
  --data "$work/expansion-multisite-v1/data" \
  --output "$work/expansion-multisite-v1/runs/reproduction-20ep" --device cuda

"$python_ml" -m pytest -q research/occupancy/training_v1
"$python_ml" -m flake8 -j 1 --config apps/engine/.flake8 research/occupancy/training_v1
```

W ograniczonym procesie GPU może być niewidoczne. Wykonany run korzysta
z rzeczywistej RTX 4070 przez dostęp do urządzenia poza tym ograniczeniem.
Nie zmieniamy architektury pod CPU. Metadane runu zapisują nazwę GPU i wersje.
Nowe testy obejmują m.in. zakaz importu test, oficjalne role train/valid,
ponownie zakodowane duplikaty i zachowanie odziedziczonych danych.

Wyniki tego przebiegu dotyczą walidacji ACPDS, na której wybierany jest
checkpoint. Nie są niezależnym testem ani porównaniem z zamrożonym baseline’em.
R1/R2/R3, Family Dollar C/T-01, temporal v1 i zasady C/D pozostają poza tuningiem.

## Ukończony wynik

Ukończono **20 epok na RTX 4070**, 12 080 kroków optymalizatora. Łączny czas
epok z walidacją wyniósł 1377.8 s (22 min 58 s). Wybrany przez najmniejszą
walidacyjną cross entropy checkpoint pochodzi z **epoki 13**, `last.pt` z epoki 20.
Nie zmieniano kryterium ani progów po obejrzeniu wyników.

| ACPDS valid, etykiety źródłowe | Wybrana epoka 13 |
| --- | ---: |
| Cross entropy | 0.033724 |
| Accuracy, próg 0.5 | 99.26% |
| Recall empty/free, próg 0.5 | 99.25% |
| Recall occupied, próg 0.5 | 99.28% |

Przy stałych progach score <=0.1 / >=0.9 zapisany model daje:

- recall źródłowego empty/free **98.88%** (1061/1073);
- recall źródłowego occupied **99.04%** (823/831);
- **3** sygnały free na źródłowo zajętych ROI;
- **7** sygnałów occupied na źródłowo pustych ROI;
- **10/1904** wyników w paśmie abstencji, pokrycie score 99.47%.

To wyniki tej samej walidacji używanej do wyboru checkpointu. Nie dowodzą
przewagi nad zamrożonym baseline’em ani jakości na docelowych kamerach.
Nie wykonano kalibracji; stan widoczności pozostaje osobnym warunkiem decyzji.
Wysoki source score nie znosi reguł `unknown`.

`runs/convnext-roi4-multisite-20ep/` zawiera wagi, snapshot kodu, konfigurację,
historię, `training-summary.json`, odczyt score dla 1904 ROI walidacji oraz
`training-curves.png`. Ponowny odczyt wag odtworzył val loss i metryki.
Ścieżki, hashe i wyniki wiąże `multisite-reference.json` w katalogu kodu.

SHA-256 `best.pt`:
`5f5672cfdea28cdb53942675d748d5f9d6c3abda5bc7fc18abe209d1d1331a23`.

SHA-256 `last.pt`:
`2b0725f57b742fbb024bea38b7b37c838f4069bdcfe73dc2d4fb6847e8121cba`.

**60 testów, Black i Flake8 przechodzą.** Sprawdzono także ponowne wywołanie
rozszerzania z zablokowaną funkcją pobierania: wykorzystuje istniejący wynik
i nie wykonuje żadnego downloadu. Zachowano 13 wcześniejszych artefaktów
(w tym poprzednie checkpointy), 93 chronione pliki wcześniejszych etapów,
oryginalne ZIP-y Roboflow i niezmienione rekordy walidacji. Nie wykonano
commita ani pushu.
