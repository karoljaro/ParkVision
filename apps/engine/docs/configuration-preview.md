# Podgląd zapisanej geometrii na referencji kamery

2026-11-01. Początkowy HEAD `4e7018f191cda607de8bee54dc07920993fb78cf`,
gałąź `main`, czyste drzewo i indeks. Bez commita.

## Narzędzie i reużycie

`src/parkvision/configuration_preview.py` udostępnia CLI oraz dwie funkcje:

- `render_configuration_preview(configuration, camera_id, image)` — obraz BGR
  uint8 → nowy obraz podglądu, bez zmiany tablicy wejściowej;
- `save_configuration_preview(configuration_path, image_path, camera_id,
  output_path)` — wczytanie istniejącego JSON i obrazu, kontrola, zapis nowego PNG.

Etap [LabelMe](labelme-slot-setup.md) dodaje `save_geometry_preview()` dla już
zwalidowanych obiektów domenowych, także DRAFT bez metadanych zatwierdzenia.
Renderer oznacza taki podgląd napisem DRAFT na obrazie i w legendzie. Standardowe
CLI opisane poniżej nadal przyjmuje wyłącznie zatwierdzony plik konfiguracji;
podgląd LabelMe uruchamia osobne polecenie `parkvision.labelme_configuration preview`.

Przejrzano `ParkingConfiguration`, `SlotProjection`, `Camera`, `ParkingSlot`,
`Point`, `Polygon` i istniejące rendery diagnostyczne. Używamy bez zmian
`load_configuration()` oraz `projections_for_camera()`; nie ma drugiego schematu
JSON, nowego modelu projekcji ani własnego walidatora geometrii. Istniejący batch
rysuje wyniki LSD/topologii, więc nie włączamy go do podglądu konfiguracji.
Wykorzystywane są już obecne zależności NumPy i OpenCV 5.

CLI z głównego katalogu repozytorium, dla własnej zatwierdzonej konfiguracji
i odpowiadającej jej referencji (ścieżki `/dane/` zastąp własnymi):

```bash
apps/engine/.venv/bin/python -m parkvision.configuration_preview \
  --configuration /dane/parking-r1.json \
  --camera-id camera-east \
  --image /dane/reference-east.png \
  --output /dane/preview-east-r1.png
```

Katalog wyjściowy musi istnieć; plik wyjściowy musi być nowy i mieć rozszerzenie
`.png`. Błędna konfiguracja, nieznana kamera, inne wymiary, brak/uszkodzenie obrazu
lub kolizja pliku dają kod wyjścia 2 i komunikat błędu. Oryginalne obrazy,
konfiguracja i istniejące podglądy nie są nadpisywane, również przez alias ścieżki,
symlink lub hardlink. PNG jest kodowany przed otwarciem nowego pliku do zapisu.

## Interpretacja obrazu

- Obszar referencji zachowuje oryginalną rozdzielczość i pozycję `(0,0)`.
  Legenda zajmuje dodatkowe 440 px po prawej. Gdy potrzebuje więcej miejsca,
  płótno rośnie także w dół; nie rozciąga to obrazu ani poligonów.
- Rysowane są wszystkie i tylko zapisane projekcje wybranej kamery. Konwersja
  schema v1 to `x_px=x*(width-1)`, `y_px=y*(height-1)`, bez zaokrąglania do
  całych pikseli przed rysowaniem (OpenCV fixed-point, 8 bitów ułamka).
- Obrysy i małe numerowane znaczniki prowadzą do legendy z **pełnym trwałym ID**
  oraz etykietą `ParkingSlot.label`. Brak etykiety jest jawnie oznaczony jako
  `(none)`. Bardzo małe obrazy mogą nie pomieścić znacznika; legenda pozostaje.
- Kolor jest wyliczany deterministycznie z `slot_id`, a numer ze wspólnej listy
  ID danej konfiguracji. To oznaczenia diagnostyczne, nie zajętość. Numer może
  zmienić się w kolejnej rewizji przy zmianie listy miejsc; trwałe pozostaje ID.
  To samo fizyczne miejsce ma ten sam kolor/numer/ID w obu kamerach, ale używa
  odpowiedniego dla kamery `SlotProjection.polygon`.
- Legenda pokazuje parking, jego nazwę, rewizję, kamerę, jej nazwę, wymiary
  i liczbę projekcji. Długie teksty są zawijane, bez usuwania końców ID.
  Wbudowana czcionka `sans` OpenCV 5 renderuje polskie znaki; znaki kontrolne
  w danych są pokazywane jako sekwencje JSON, np. `\n`.
- Kamera bez przypisanych projekcji dostaje komunikat o braku ROI i obraz bez
  wymyślonych obrysów. Brak ROI nie jest stanem „wolne” ani „zajęte”.

Odczyt obrazu to 8-bit BGR bez automatycznej rotacji EXIF, resize i cropa.
Operator wybiera plik właściwej kamery w układzie współrzędnych konfiguracji.
Rozdzielczość nie dowodzi zgodności kadru/pozycji kamery. Wczytanie metadanych
zatwierdzenia również nie zastępuje wizualnej weryfikacji operatora.

Rysujemy istniejącą geometrię także wtedy, gdy linie na referencji zasłania
samochód. Nie wykonujemy detekcji miejsc, LSD, ML ani oceny zajętości. Podgląd
nie zmienia konfiguracji i nie inicjuje ponownej kalibracji. Ewentualna poprawka
pozostaje świadomą operacją operatora z nową zatwierdzoną rewizją.

## Weryfikacja

Początkowo przechodziło 811 testów silnika. Dodano **26 testów** w
`tests/test_configuration_preview.py`; cały zestaw daje **837 passed**.
Sprawdzono wybór obu kamer, wspólny slot, pełne ID i etykiety, polskie znaki,
brak etykiety, długie teksty, marginesy legendy, skrajne współrzędne 0/1,
brak ROI, błędne obrazy/wymiary/ID/JSON, niezmienność wejść, format PNG,
odmowę nadpisania (w tym symlink/hardlink), błąd kodowania oraz CLI z innego cwd.

Przechodzą Flake8 całego `src/tests/scripts` silnika, Black obu nowych plików
Python i Mypy tych plików. Polecenia z głównego katalogu repo:

```bash
apps/engine/.venv/bin/python -m pytest -q apps/engine/tests
apps/engine/.venv/bin/python -m flake8 --jobs 1 --config apps/engine/.flake8 \
  apps/engine/src apps/engine/tests apps/engine/scripts
apps/engine/.venv/bin/python -m black --check apps/engine/src/parkvision/configuration_preview.py
apps/engine/.venv/bin/python -m black --check apps/engine/tests/test_configuration_preview.py
research/paint-segmentation/.venv/bin/python -m mypy \
  --python-executable apps/engine/.venv/bin/python --follow-imports=silent \
  apps/engine/src/parkvision/configuration_preview.py \
  apps/engine/tests/test_configuration_preview.py
```

Mypy pochodzi z istniejącego środowiska research, a typy zależności z silnika;
nie zmieniano zależności ani środowisk.

Przez rzeczywiste wywołania CLI zapisano i wizualnie sprawdzono podglądy obu
kamer dla **istniejącej syntetycznej konfiguracji**. Grid referencyjny jest
sztuczny i ma wyraźny podpis; nie udaje zdjęcia ani kalibracji parkingu.
Wyniki są poza repozytorium:

`/home/karoljaron/Dokumenty/ChatGPT/System parkingowy/configuration-preview-2026-11-01-v1/`

- `reference-camera-east.png`, `reference-camera-west.png` — syntetyczne wejścia;
- `preview-camera-east.png`, `preview-camera-west.png` — wynik CLI;
- `synthetic-configuration.json` — kopia przykładu z poprzedniego etapu;
- `smoke-check.json` — wymiary, liczby projekcji, kody wyjścia i SHA256 wejść.

Są rzeczywiste obrazy w `samples/`, ale nie znaleziono powiązanej zatwierdzonej
konfiguracji miejsc. Nie naniesiono na nie poligonów testowych. Na rzeczywistym
`04-cnrext-camera3-wet.jpg` (1000×750) wykonano wyłącznie test negatywny: wybrana
syntetyczna `camera-east` wymaga 1280×720, więc CLI odrzuciło wejście i nie
utworzyło PNG. Hashe obrazu rzeczywistego, obu referencji syntetycznych i źródła
JSON pozostały identyczne przed i po próbach.

Ten krok został następnie wykonany w [pilocie LabelMe](labelme-slot-setup.md):
użytkownik zweryfikował i zatwierdził trzy rzeczywiste ROI na `parking.jpg`.
Zapisano `approved/parking-r1.json` i wygenerowano `approved-preview-r1.png`
w prywatnym katalogu pilota. Nie ma tu automatycznego tworzenia ani zatwierdzania miejsc.
Kod CV/ML, checkpointy, dane i format konfiguracji pozostają bez zmian.
