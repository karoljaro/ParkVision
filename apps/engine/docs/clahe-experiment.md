# LSD kontra CLAHE — mały eksperyment ekstrakcji

2026-10-30. Początkowy HEAD: `b297e73b4d0de157fed916cc21239f36335f306a`.
Drzewo robocze było czyste. Nie wykonano commita.

Wniosek: pozostawić CLAHE jako opcję diagnostyczną. W tych próbach lokalny kontrast
rośnie, ale dodatkowe odcinki często opisują nawierzchnię, refleksy i roślinność,
a słabe oznaczenia nadal mają przerwy. Brak podstaw do domyślnego włączenia CLAHE.

## Metoda i zgodność

- OpenCV **5.0.0**, oryginalna rozdzielczość i te same współrzędne pikselowe.
- Baseline: niezmienione BGR → istniejące `OpenCVSlotDetector.extract_segments()`.
- Wariant: BGR2GRAY → `cv2.createCLAHE(clipLimit=2.0 lub 4.0,
  tileGridSize=(8, 8)).apply(gray)` → GRAY2BGR → **ten sam ekstraktor i obiekt LSD**.
  Adapter BGR zachowuje dokładnie wartości szarości po ponownej konwersji ekstraktora
  (sprawdzone testem). Nie rozszerzano API detektora ani `Frame`.
- LSD pozostaje `LSD_REFINE_STD` z dotychczasowymi ustawieniami OpenCV. Nie ma
  skalowania, dodatkowego rozmycia ani filtrowania segmentów. Nakładki A/B pokazują
  wszystkie fragmenty na tym samym oryginalnym BGR, linią 1 px z subpikselowymi końcami.
  Próg `--min-length` nadal dotyczy wyłącznie starego `preview_lsd.png`.
- CLAHE działa na uint8 grayscale; siatka 8×8 oznacza **liczbę kafelków**, zgodnie
  z [dokumentacją OpenCV 5](https://docs.opencv.org/5.0/main_modules/classcv_1_1CLAHE.html).
- Dla każdej ścieżki: 1 pominięta rozgrzewka i 7 pomiarów, naprzemienna kolejność A/B,
  bez zapisu plików i renderowania w pomiarze. Czasy obejmują ekstrakcję wraz z konwersją
  BGR2GRAY i tworzeniem `LineSegment`; preprocessing wariantu obejmuje też adapter BGR.
  Użyto zastanego ustawienia OpenCV: 12 wątków. Nie zmieniano ustawień globalnych.
  Mediany etapów i całości liczone są osobno, więc nie muszą się sumować.
- Wyniki CLAHE **nie trafiają** do grupowania, dopasowania, parowania ani topologii.
  Te etapy nadal otrzymują wyłącznie bazowe segmenty. `detect()` pozostaje nietknięte.

## Wyniki trzech głównych zdjęć

`wet` = `04-cnrext-camera3-wet.jpg`, 1000×750; `parking` = `parking.jpg`, 1440×960;
`suchy/cienie` = `02-empty-parking-shadows.jpg`, 3000×2331 (słoneczna, sucha kontrola).
N, mediana i suma dotyczą **wszystkich** segmentów. ≥100 px to wyłącznie statystyka.
Większa liczba segmentów i większa suma długości nie oznaczają lepszej detekcji miejsc.

| Zdjęcie | Wariant | N | Mediana [px] | Suma [px] | ≥100 px | Pre / ekstrakcja / całość [ms] |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| wet | baseline | 722 | 13.96 | 14714.5 | 9 | 0.00 / 37.68 / 37.68 |
| wet | CLAHE 2, 8×8 | 1647 | 11.26 | 24210.4 | 7 | 0.79 / 62.37 / 63.23 |
| wet | CLAHE 4, 8×8 | 2182 | 10.07 | 28320.9 | 5 | 0.78 / 76.86 / 77.68 |
| parking | baseline | 1866 | 21.65 | 74271.2 | 164 | 0.00 / 90.07 / 90.07 |
| parking | CLAHE 2, 8×8 | 2712 | 16.42 | 82001.5 | 145 | 1.31 / 130.01 / 131.30 |
| parking | CLAHE 4, 8×8 | 3384 | 13.97 | 85799.6 | 133 | 1.38 / 169.29 / 170.59 |
| suchy/cienie | baseline | 5884 | 20.15 | 216175.2 | 830 | 0.00 / 470.31 / 470.31 |
| suchy/cienie | CLAHE 2, 8×8 | 14268 | 12.51 | 288549.2 | 456 | 8.87 / 832.82 / 852.91 |
| suchy/cienie | CLAHE 4, 8×8 | 20460 | 10.67 | 325205.2 | 261 | 8.27 / 913.63 / 931.14 |

Baseline w tabeli pochodzi z uruchomienia CLAHE 2. Ponowny baseline w uruchomieniu
CLAHE 4 dał identyczną geometrię; mediany ekstrakcji wyniosły odpowiednio
37.37 / 94.78 / 413.15 ms. Czasy zależą od obciążenia i planowania wątków;
to pomiar diagnostyczny, nie gwarancja wydajności. Surowe 7 próbek dla każdej
ścieżki znajdują się w osobnych JSON-ach.

Dodatkowo sprawdzono `01-pklot-ufpr05-empty.jpg` (1280×720); nie zakładano na podstawie
samej nazwy, że jest to jednoznacznie sucha kontrola, dlatego dołączono zdjęcie z cieniami:

| Zdjęcie | Wariant | N | Mediana [px] | Suma [px] | ≥100 px | Pre / ekstrakcja / całość [ms] |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| UFPR05 | baseline | 1403 | 11.79 | 23974.1 | 4 | 0.00 / 43.60 / 43.60 |
| UFPR05 | CLAHE 2, 8×8 | 3862 | 10.07 | 48639.1 | 3 | 0.92 / 79.65 / 80.57 |
| UFPR05 | CLAHE 4, 8×8 | 5525 | 9.52 | 63075.5 | 3 | 1.07 / 118.08 / 119.15 |

## Ogląd wizualny: obserwacje

Sprawdzono oryginały, nakładki A/B i powiększenia oznaczeń oraz otoczenia.
Na mokrym zdjęciu obejrzano również obie przetworzone jasności.

- **Wet:** obie konfiguracje poprawiają widoczność słabej farby w jasności; miejscami
  dochodzą fragmenty jej krawędzi. W obszarze `(220..500, 180..300)` CLAHE 2 dodaje
  krótkie fragmenty przy oznaczeniach i wyspie. Nie daje to ciągłego odtworzenia linii.
  Słaba linia w dolnej lewej części, około `(170..200, 480..550)`, nadal jest w dużej
  części pominięta. Przy `(800..950, 250..320)` przybywa fragmentów związanych z mokrymi
  plamami/refleksami obok samochodu; rośnie też liczba krawędzi liści i trawnika.
  CLAHE 4 nasila ten efekt. Krawężniki i słupy pozostają silnymi źródłami odcinków.
- **Parking:** wyraźne białe oznaczenia były już dobrze widoczne. CLAHE dodaje
  krawędzie spękań/ziarnistości w środku miejsc i otoczenia wyspy; w oglądanych
  powiększeniach nie widać przekonującego zysku ciągłości oznaczeń. Przy 4 wzrasta
  zagęszczenie drobnych fragmentów, a liczba długich maleje 164 → 145 → 133.
- **Suchy/cienie:** baseline już pokazuje większość długich białych linii. CLAHE
  wzmacnia ślady nawierzchni i okolice cieni, dodaje wiele krótkich fragmentów
  pomiędzy miejscami. Liczba długich spada 830 → 456 → 261; same te liczby nie
  rozstrzygają, które oznaczenia są lepiej pokryte.
- **UFPR05:** silnie przybywa tekstury asfaltu, roślinności i detali wysp, szczególnie
  przy 4. Namalowane linie i krawężniki nadal są mieszane w jednym zbiorze LSD.

To ocena wizualna, bez ręcznych etykiet. Nie wyznaczano precision/recall ani udziału
poprawnych detekcji. Przypuszczenie: zysk może dotyczyć części lokalnych słabych krawędzi,
ale wymaga porównania na oznaczonych fragmentach; obecne metryki tego nie dowodzą.

## Pliki i uruchomienie

Zmiany w repozytorium (ścieżki od `apps/engine/`):

- `src/parkvision/detection/clahe_preprocessing.py` — jedna funkcja przetwarzania
  i wspólna walidacja parametrów; bez nowych modeli i klas.
- `scripts/run_parking_batch.py` — opt-in `--clahe-compare`, parametry, pomiary i zapis.
- `tests/detection/test_clahe_preprocessing.py` — formaty, rozmiar, typ, walidacja,
  zgodność z natywnym CLAHE i brak mutacji, także dla obrazu tylko do odczytu.
- `tests/scripts/test_run_parking_batch.py` — CLI, rzeczywiste LSD, brak uruchamiania
  CLAHE bez flagi, identyczność 11 poprzednich plików, wszystkie współrzędne/ID,
  kompletne metryki mimo wysokiego progu starego podglądu, nowe pliki i źródło bez mutacji.
- `docs/clahe-experiment.md` — ten raport.

Z katalogu `/home/karoljaron/Projects/ParkVision/apps/engine`:

```bash
.venv/bin/python scripts/run_parking_batch.py \
  --samples samples/04-cnrext-camera3-wet.jpg \
  --output /tmp/parkvision-clahe-2 \
  --clahe-compare --clahe-clip-limit 2 --clahe-tile-grid 8 8 --clahe-repeats 7
```

Dla drugiej konfiguracji zmienić limit na 4 i katalog wyjściowy. Pozostałe główne
wejścia: `samples/parking.jpg` i `samples/02-empty-parking-shadows.jpg`.
Bez `--clahe-compare` nie ma dodatkowej ekstrakcji, preprocessingu ani plików A/B.

Każdy katalog `image-N/clahe_comparison/` zawiera osobne `baseline.json`, `clahe.json`
(statystyki, surowe czasy, parametry i **wszystkie** segmenty), `preview_baseline.png`,
`preview_clahe.png`, `grayscale_baseline.png`, `grayscale_clahe.png`.
Nie nadpisuje to `preview_lsd.png`. Kolejne ustawienia należy zapisywać do osobnych
katalogów `--output`, bo nazwy plików porównania w ramach jednego uruchomienia są stałe.

Rzeczywiste artefakty są poza repozytorium:
`/home/karoljaron/Dokumenty/ChatGPT/System parkingowy/clahe-experiment-2026-10-30/`.
Zawierają `before/`, `clip-2-grid-8/`, `clip-4-grid-8/`, `summary-metrics.json`
(zbiorcze czasy w **ms**), `byte-regression.json` (hashe), oraz `review-*.png`
z powiększeniami. W źródłowych JSON-ach A/B czasy są w sekundach.
Do wszystkich uruchomień kontrolnych dodano również `--topology
--topology-max-extension 6 --topology-support`, wyłącznie w celu sprawdzenia zachowania
poprzednich plików; nie są wymagane do eksperymentu CLAHE.

## Weryfikacja i rekomendacja

- Przed zmianą: **723 passed**. Po zmianie pełny `pytest -q`: **755 passed**.
- Pełny `flake8 --jobs 1 src tests scripts`: przechodzi.
- Black `--check` dla czterech zmienionych/dodanych plików Pythona: przechodzi.
  Nie poprawiano formatowania w innych plikach.
- Mypy 2.4.1: **brak błędów w 4 plikach**. Narzędzie pobrano tylko do
  `/tmp/parkvision-clahe-mypy`; zależności projektu pozostały bez zmian.
- **88/88** porównań plików identycznych bajtowo: 11 wcześniejszych plików ×
  4 zdjęcia × 2 konfiguracje. Dla pierwszych trzech zdjęć kopie kontrolne zapisano
  przed edycją; suchą kontrolę odtworzono skryptem z początkowego HEAD.
- Nie zmieniono ekstraktora, `Frame`, `detect()`, grupowania, dopasowania,
  parowania, topologii ani pomiaru wsparcia. Nie utworzono commita.

Rekomendacja: zachować CLAHE 2/8×8 jako dostępną opcję diagnostyczną, pozostawić
baseline domyślnie. Nie uzasadniono globalnego wdrożenia; 4/8×8 wnosi wyraźnie
więcej drobnych krawędzi. Kolejny mały eksperyment: ręcznie oznaczyć kilka tych
samych odcinków słabej farby oraz obszarów z refleksami/krawężnikami i porównać ich
pokrycie oraz fragmentację dla baseline i 2/8×8. Tę propozycję pozostawiono na kolejny
etap; nie dodano filtrów, multi-scale, treningu ani kalibracji wieloklatkowej.
