# Dataset Preparation + Model Improvement v1

**Końcowe porównanie jest wykonane:** [wyniki i zakres](final-comparison.md).
Zamrożony kandydat został oceniony na ACPDS test i CNR Pisa po zakończeniu
treningu, bez późniejszego strojenia. Starsze reference JSON opisują stan
wcześniejszych etapów; końcowy pomiar ma osobny `final-comparison-reference.json`.

**Aktualna kontynuacja:** [trening wieloparkingowy UFPR + ACPDS train](multisite.md)
ma 9664 ROI train i tę samą walidację 1904 ROI. Poniżej zachowano opis pierwszego
etapu na samym UFPR oraz jego ukończonego przebiegu 20 epok.

2026-11-08. Powstał działający importer danych, nowy klasyfikator per ROI
**ConvNeXt Tiny ImageNet + kanał maski stanowiska**, pipeline fine-tuningu
i rzeczywiście wytrenowany pierwszy checkpoint. Nie uruchomiono kolejnego
porównania starego baseline’u. Nowy model jest kandydatem badawczym;
nie zastępuje `observe.py`, reguł C/D ani temporal v1.

Wszystkie dane, dowody źródłowe, logi i wagi są poza repozytorium:

`/home/karoljaron/Dokumenty/ChatGPT/System parkingowy/dataset-model-improvement-2026-11-08-v1/`

## Dane użyte do treningu

| Rola | Źródło | Klatki | ROI empty / occupied |
| --- | --- | ---: | ---: |
| train | PKLot/MetaPKLot UFPR04 + UFPR05 | 48 | 1067 / 755 |
| val | ACPDS, wyłącznie oficjalny `valid` | 35 | 1073 / 831 |
| razem | | 83 | 2140 / 1586 |

UFPR04/05 przedstawiają **jeden fizyczny parking** i pozostają razem w train.
Zablokowano całe dni wszystkich wcześniejszych pilotaży tego obiektu, także
prób konfliktów i weak-free-veto, oraz dni znanych zdjęć farby. Manifest zapisuje
62 wykluczone grupy parking–dzień i hashe wcześniejszych manifestów.
Nowe klatki dobierano deterministycznie (seed 42), najwyżej dwie na widok/dzień,
z odstępem co najmniej godziny, bez zaglądania do etykiet podczas wyboru.

Pierwotnie rozważano CNR jako val. Rozszerzony audyt wykazał, że starsze próby
objęły wszystkie dostępne dni CNR przez różne kamery. **Żaden obraz CNR nie
wszedł do końcowego manifestu treningowego.** Walidację przeniesiono do ACPDS.
ACPDS `valid` ma zmienne widoki, nie sekwencję jednej kamery. Autor deklaruje
rozłączne parkingi w oficjalnych splitach; JSON nie ujawnia kompletnych site IDs.
Dlatego wszystkie jego lokalizacje, kamera i sesje są zachowawczo grupowane
razem w val, bez wymyślania liczby parkingów i mapowania fizycznych stanowisk.
Nie pobierano obrazów ACPDS `train` ani wcześniejszego `test`.

Źródła i prawa:

- [PKLot, UFPR](https://web.inf.ufpr.br/vri/databases/parking-lot-database/):
  **CC BY 4.0**, Almeida i in., *Parking lot database*, ESWA 42(11), 2015.
- [MetaPKLot](https://github.com/DSBD-Research/MetaPKLot-Dataset/tree/9218800c4eb2b7c1b17b51c8cdb07ad51686f8a2):
  przypięty commit `9218800c4eb2b7c1b17b51c8cdb07ad51686f8a2`;
  obrazy i paczki anotacji sprawdzono z Git blobami lokalnego przypiętego drzewa.
  Cytować de Almeida i in., 2026, DOI `10.1007/s00521-026-12398-0`.
  Nie przenosimy jednej licencji na cały mieszany MetaPKLot.
- [ACPDS, Martin Marek](https://arxiv.org/abs/2107.12207): autor wprost deklaruje
  **MIT dla danych, kodu i modeli**, a nie tylko kodu repozytorium.
  Wykorzystano lokalne, przypięte anotacje i indeks ZIP. Pobrano zakresami tylko
  35 brakujących JPEG-ów `valid`, sprawdzając ETag, rozmiary i CRC członków.
- [ConvNeXt Tiny, Torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html):
  oficjalne wagi `IMAGENET1K_V1`, SHA-256
  `983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d`.

Archiwów anotacji MetaPKLot, indeksu i anotacji ACPDS nie pobierano ponownie.
Kopie deklaracji, URL-e, SHA-256 i błędy dostępu zapisano w `evidence/`.
Przed otrzymaniem lokalnych ZIP-ów sprawdzono również publiczny eksport
Roboflow; bezpośredni link zwrócił 403. Nie proszono o klucz. Po wskazaniu
lokalnych archiwów nie pobierano żadnego zbioru Roboflow.

## Lokalne ZIP-y Roboflow: rzeczywiste zawartości

Archiwa w `/home/karoljaron/Datasets/ParkVision/` pozostawiono bez zmiany bajtów.
Każdy obraz sprawdzono przez CRC ZIP, SHA-256, dekodowanie i zgodność wymiarów.
Wynik importu jest w `roboflow-prepared-v1/`: 4960 oryginalnych obrazów,
osobne oryginalne COCO, README, `images.jsonl`, `audit.json`, `duplicates.json`.
Podział eksportera zapisano jako pochodzenie, nie jako zatwierdzony split ML.

| Archiwum | train / valid / test | Anotacje | Znaczenie |
| --- | --- | ---: | --- |
| Parking Space Detection v3 | 2000 / 400 / 0 | 88 689 | `parking_space` 86 289, `blocked_space` 2400; geometria stanowisk |
| Parking lines v5 | 192 / 33 / 0 | 786 | `ParkingLines`; dodatnie poligony farby |
| parking lot v1 | 2133 / 116 / 86 | 69 515 | `car` 43 625, `free` 25 890; bboxy różnych obiektów/obszarów |

Klasy nadrzędne COCO (`objects`, `cars-free` oraz ID 0 `ParkingLines`) nie mają
anotacji. W v5 nie ma klasy `d` widocznej w ogólnym opisie innej wersji projektu.
`parking lot` ma 2335 wyeksportowanych obrazów, a nie 913 z galerii; README
potwierdza trzy warianty po augmentacjach oraz rozciągnięcie do 640×640.

Ustalenia audytu:

- **0** identycznych bajtowo obrazów i **0** identycznych zdekodowanych obrazów.
  To nie dowodzi niezależności źródeł.
- **28 grup tych samych nazw źródłowych** ma warianty w train i valid `parking lot`.
  Są to dowody przecieku po augmentacji. Nie wolno używać tych splitów wprost.
- **43 grupy nazw sugerujących wspólne źródło** przecinają splity, w tym
  12 prefiksów w Parking Space Detection i 31 rodzin w parking lot.
  Prefiksy liczbowe mogą oznaczać dni; nie traktujemy ich jako camera/site IDs.
- **224 pary kandydatów podobieństwa**, z czego **42 między splitami**,
  według 256-bit dHash, Hamming <=6. To heurystyka do przeglądu, nie dowód
  tożsamości. Pełna geometria i pochodzenie są ważniejsze niż sam hash.
- **1227 bboxów wychodzi poza obraz** (1180 train, 47 valid w parking lot).
  Oryginalne współrzędne zachowano; nie przycinano ich bez śladu.
- **768 nazw wskazuje na stock/iStock**. Całe test tego eksportu ma takie nazwy.
  Parking lines zawiera nazwy przypominające Flickr IDs. Sam eksport nie
  dokumentuje indywidualnych autorów, praw ani pierwotnych parking/camera IDs.
- README i COCO deklarują **CC BY 4.0** we wszystkich trzech eksportach.
  Jest to deklaracja publikującego. W szczególności nie rozstrzyga praw do
  wskazanych materiałów stockowych ani pojedynczych zdjęć z innych serwisów.
- Nie znaleziono identycznych Git blobów w przypiętym PKLot/CNRPark-EXT.
  Brak takiego dopasowania nie wyklucza przeskalowanych kopii PUCPR.

Przegląd reprezentatywnych przykładów (`roboflow-semantic-review.jpg`) potwierdza,
że `parking_space` obejmuje **także stanowiska z samochodami**. Nie oznacza free;
`blocked_space` również nie oznacza automatycznie occupied. `free` bbox z trzeciego
zbioru nie jest obserwacją tej samej fizycznej projekcji co `car` bbox.
Nie wolno z samych konturów wygenerować etykiet zajętości ani z niewykrycia auta
wywnioskować wolnego miejsca. Maska linii też nie daje automatycznie poprawnego
tła poza opisanymi fragmentami farby.

Dlatego import zapisuje jawny typ zadania, `occupancy_label=null`, nieznane
physical parking/camera/session IDs i `training_eligible=false`. Obrazy test
pozostają `heldout`, pozostałe mają rolę `quarantine`. Dane są realnie rozpakowane,
opisane i dostępne do dalszego importu geometrii/farby po ustaleniu pochodzenia
i grup. **Żaden obraz Roboflow nie został użyty w tym klasyfikatorze occupancy.**
Nie uruchamiamy drugiego modelu geometrii w tej iteracji tylko po to, aby zużyć
niezgodne anotacje; wybrana ścieżka poprawy pozostaje occupied/free per ROI.

## Format i kontrakt nowego klasyfikatora

`data/manifest.json` łączy każdy crop z oryginalnym obrazem, anotacją, jej paczką,
parkingiem, kamerą i sesją. `arrays/*.npy` to **uint8 224×224×4 RGB + maska ROI**:
prostokąt obejmujący poligon z 25% kontekstem i wspólny letterbox, bez wycinania
części miejsca ani prostowania perspektywy. Maski nie poddaje się augmentacji
fotometrycznej. RGB ma normalizację ImageNet, maska zakres 0/1. Flip jest wspólny.
Kanał maski w pierwszej warstwie startuje z zer, RGB z ImageNet; kanał jest uczony.

Loader odrzuca naruszenie hashy, etykiet źródłowych, geometrii, parent/split,
wspólne parkingi/kamery/sesje, dokładne duplikaty cropów i zabronione źródła.
Sprawdza ponownie powiązanie z oryginalnymi COCO/anotacjami ACPDS.
PUCPR jest wykluczone allowlistą; odczytano tylko metadane zamrożonego locka.
Nie otwierano testowych obrazów, masek ani anotacji PUCPR.

`SlotPrediction` ma typowany stan `free/occupied/unknown`, wynik occupied oraz
powód abstencji. Progi badawcze 0.10/0.90 nie były strojone. Domyślny wynik to
`unknown`, dopóki osobno nie potwierdzono legalnego slotu, widoku i widoczności.
`empty` z datasetu nie potwierdza widoczności. Nie uczymy ani nie udajemy detektora
widoczności/OOD. Softmax pozostaje nieskalibrowanym wynikiem modelu.
Różne projekcje nie dostają fikcyjnego wspólnego physical slot ID.

R1/R2/R3, legalne C i nieparkingowy T-01 Family Dollar, ClearPix, C/D oraz
temporal v1 nie uczestniczą w treningu/tuningu. T-01 nie jest legalnym slotem.
Kontrolę zachowania wcześniejszych plików zapisano w `preservation-*.json`.

## Wykonany trening i dalszy fine-tuning

`runs/convnext-roi4-first-epoch/` zawiera snapshot kodu, `run.json`,
`history.jsonl`, `checkpoints.json`, `best.pt`, `last.pt`.
Seed 42, AdamW, LR backbone 2e-5 / head 5e-4, weight decay 0.05,
ważona CE z wagami wyłącznie z train, label smoothing 0.02, batch 16.
**1 pełna epoka, 1822 przykłady, 114 kroków optymalizatora, cały backbone uczony.**
Epoka z walidacją trwała około 241 s. W ograniczonym procesie urządzenia GPU nie
były widoczne, więc pierwszy run wykonano na CPU. Później potwierdzono działającą
**RTX 4070 i CUDA poza tym ograniczeniem**. Nie był to uszkodzony sterownik ani
brak karty; dalszy fine-tuning korzysta z GPU, z tą samą architekturą.

Walidacja względem etykiet źródłowych ACPDS: accuracy 69.43%, free recall przy
0.5 = 46.13%, occupied recall 99.52%. To słaby transfer free po pierwszej epoce;
checkpoint nie jest modelem gotowym do wdrożenia. Nie wykonano porównania
z zamrożonym baseline’em i nie deklarujemy wzrostu jakości. Wynik nie opisuje
zweryfikowanej widoczności ani zachowania C/D i temporal.

### Ukończona kontynuacja na RTX 4070

Po pierwszej epoce wykonano **19 dalszych epok na RTX 4070**, do epoki 20
włącznie. Powstał osobny run `runs/convnext-roi4-resume-20ep-rtx4070/`.
Wykonano 2166 dodatkowych aktualizacji wag (2280 łącznie). Czas epok z walidacją
na GPU wyniósł 284 s, około 15 s/epokę. Oryginalna pierwsza epoka i jej wagi
zostały zachowane. Seed, dane, augmentacje, progi i kryterium wyboru nie zmieniły się.
Zmiana CPU→CUDA jest jawna; nie deklarujemy bitowej zgodności między urządzeniami.

**Najlepszy checkpoint pochodzi z epoki 4**, wybranej przez najmniejszą
walidacyjną cross entropy. `last.pt` pochodzi z epoki 20. Nie zastępujemy
najlepszego modelu ostatnim: późniejsze epoki przeuczały się do parkingu UFPR.

| Walidacja źródłowa ACPDS valid | Epoka 1 | Wybrana epoka 4 | Epoka 20 |
| --- | ---: | ---: | ---: |
| Cross entropy | 0.8351 | **0.6834** | 0.8970 |
| Accuracy, próg 0.5 | 69.43% | **77.63%** | 76.26% |
| Recall empty/free, próg 0.5 | 46.13% | **61.04%** | 58.53% |
| Recall occupied, próg 0.5 | 99.52% | 99.04% | 99.16% |

Odczyt `best.pt` i ponowna kontrola tej samej dozwolonej walidacji potwierdziły
zapisane wyniki. Przy niezmienionych progach score <=0.1 i >=0.9:

- 489 wyników z sygnałem free; 0 na źródłowo zajętym ROI (831 takich ROI).
- Recall źródłowego empty przy <=0.1 wynosi **45.57%** (489/1073).
- 1041 wyników z sygnałem occupied; **262 na źródłowo pustym ROI**.
- 374 wyniki w paśmie abstencji, czyli 19.64% wszystkich 1904 ROI.

To statystyki score względem etykiet źródłowych, nie zatwierdzone decyzje
parkingowe. Widoczność w tych etykietach pozostaje nieoceniona, a wyniki
nieskalibrowane. W szczególności problem fałszywego occupied nadal jest istotny.
Sprawdzono też typowany odczyt modelu: bez potwierdzenia legalnego slotu i widoku
domyślna decyzja nadal jest `unknown`.

Run zawiera `finetune-summary.json`, `validation-score-readback.jsonl`,
`domain-contract-readback.json`, `gpu-provenance.json` i `training-curves.png`.
Ścieżki i SHA-256 wiąże nowy `finetune-reference.json` w tym katalogu kodu;
`run-reference.json` nadal opisuje pierwszy, zachowany przebieg.
Kontrolę wznowienia pokrywa dodatkowe 6 testów: **56 testów przechodzi**, Black
i Flake8 również. Poprawka zachowuje wcześniejszy globalny best przy wznowieniu,
nawet gdy żadna dalsza epoka go nie poprawi.

Odtworzenie w katalogu repozytorium, z istniejącym środowiskiem ML
(bez instalowania PyTorch do silnika):

```bash
cd /home/karoljaron/Projekty/ParkVision
python_ml="$PWD/research/paint-segmentation/.venv/bin/python"
work='/home/karoljaron/Dokumenty/ChatGPT/System parkingowy/dataset-model-improvement-2026-11-08-v1'

# Idempotentna kontrola przygotowanych danych; bez pobierania i treningu.
"$python_ml" -m research.occupancy.training_v1.data --output "$work/data" --verify-only
"$python_ml" -m research.occupancy.training_v1.roboflow --output "$work/roboflow-prepared-v1" --verify-only

# Nowy przebieg 20 epok; wymaga działającego GPU. Nie nadpisuje pierwszego runu.
CUBLAS_WORKSPACE_CONFIG=:4096:8 "$python_ml" -m research.occupancy.training_v1.train \
  --config research/occupancy/training_v1/finetune.json \
  --data "$work/data" --output "$work/runs/convnext-roi4-20ep" --device cuda

# Kontynuacja od epoki 1 do 20, z zachowaniem optimizer/RNG i tego samego datasetu.
CUBLAS_WORKSPACE_CONFIG=:4096:8 "$python_ml" -m research.occupancy.training_v1.train \
  --config research/occupancy/training_v1/finetune.json \
  --data "$work/data" --output "$work/runs/convnext-roi4-resume-20ep" --device cuda \
  --resume "$work/runs/convnext-roi4-first-epoch/last.pt"

# Testy techniczne nowego kodu; nie uruchamiają starego pilotażu.
"$python_ml" -m pytest -q research/occupancy/training_v1
for file in research/occupancy/training_v1/*.py; do
  BLACK_CACHE_DIR=/tmp/parkvision-black "$python_ml" -m black --check "$file" || break
done
"$python_ml" -m flake8 -j 1 --config apps/engine/.flake8 research/occupancy/training_v1
```

Wznowienie kopiuje i sprawdza najlepszy checkpoint poprzedniego przebiegu.
Nawet gdy żadna nowa epoka nie poprawi val loss, nowy katalog nadal zawiera
poprawny globalny `best.pt`; stan optymalizatora kontynuuje się z `last.pt`.

`first-run.json` odtwarza wykonaną pojedynczą epokę z `--device cpu` i nową nazwą
katalogu wynikowego. W nowym środowisku użyj osobnego research venv i zależności
z `requirements.txt`, dobierając oficjalne wheel PyTorch/CUDA do GPU.
Jawne `--device cuda` kończy się błędem, jeśli CUDA nie działa. Checkpointy
wewnątrz repozytorium są odrzucane. Zmiana datasetu przerywa resume.

`predict.py --help` udostępnia niezależny od obserwatora odczyt checkpointu
z kontrolą SHA-256; wykonano readback jednego przykładu train, wynik w
`checkpoint-readback.json`. Domyślne zabezpieczenia poprawnie dały `unknown`.

Najważniejsza dalsza praca: rozszerzenie **treningu** o niezależne parkingi
z prawdziwymi, sparowanymi etykietami ROI–occupied/free. Pełny przebieg 20 epok
już wykonano; nie ma podstaw do dalszego wydłużania samego treningu na UFPR.
Nowe ZIP-y nie dostarczają wymaganego nadzoru occupancy. Następnie należy zweryfikować
widoczność, skalibrować model na dozwolonej walidacji i dopiero po zamrożeniu
wag/progów wykonać jedno uczciwe porównanie z zachowanym baseline’em.
