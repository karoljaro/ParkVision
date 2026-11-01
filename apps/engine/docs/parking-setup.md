# Jednorazowy setup i zapisana geometria parkingu

Stan sprawdzony 2026-11-01, początkowy HEAD
`a106b9ce8a4f0f4381243cdf83933c243e3f648e`, gałąź `main`, czyste drzewo
robocze i indeks. Repo zawiera `apps/engine` oraz niezależne eksperymenty
`research/paint-segmentation`; nie znaleziono obowiązującego `AGENTS.md`.

## Decyzja architektoniczna

**Geometrię konfigurujemy i zatwierdzamy raz dla stabilnego układu i kamery.**
Samochód, zasłonięte linie, deszcz lub cień w następnej klatce nie zmieniają
identyfikatorów miejsc ani poligonów i nie uruchamiają inicjalizacji.

Są trzy osobne operacje:

1. **Setup:** operator wybiera obraz referencyjny, ręcznie rysuje/poprawia ROI,
   przypisuje fizyczne miejsca i zatwierdza geometrię. CV/ML mogą pomagać.
2. **Świadoma rekonfiguracja:** zmiana kamery, kadrowania, obiektywu lub układu
   parkingu wymaga przeglądu i nowej zatwierdzonej rewizji. Do tego czasu stara
   konfiguracja nie powinna być używana dla zmienionego widoku. Nie ma tu
   automatycznego wykrywania przesunięcia kamery.
3. **Przyszły runtime zajętości:** wczytuje wybraną konfigurację i korzysta z ROI
   właściwej kamery. Słaba widoczność może dać brak wiarygodnej obserwacji
   zajętości; nie oznacza zniknięcia miejsca. Ani klasyfikacja zajętości, ani
   łączenie obserwacji kilku kamer nie są jeszcze zaimplementowane.

## Audyt istniejących elementów

Moduły `domain/`, `capture/` i `detection/` poniżej znajdują się pod
`apps/engine/src/parkvision/`; skrypt jest pod `apps/engine/scripts/`, a research
w głównym katalogu repozytorium.

| Element | Faktyczne działanie i reużycie | Brak przed tym etapem |
| --- | --- | --- |
| `domain/parking.py`: `Parking` | Niemutowalne `id`, `name`; identyfikuje parking. | Kontrola pustych/unikalnych ID, zapis. |
| `domain/camera.py`: `Camera` | `id`, `parking_id`, `name`; kamera należy do parkingu. | Spójność referencji, rozmiar i wersja widoku. |
| `domain/parking_slot.py`: `ParkingSlot` | `id`, `parking_id`, opcjonalny `label`; fizyczne miejsce niezależne od kamery. | Spójność referencji i trwały zapis. |
| `domain/slot_projection.py`: `SlotProjection` | `camera_id`, `parking_slot_id`, `Polygon`. Już wyraża relację wiele kamer → jedno miejsce oraz wiele miejsc → kamera. Istniejący test sprawdza różne projekcje tego samego miejsca. | Rejestr powiązań, unikalność pary, import/eksport. |
| `domain/geometry/point.py`: `Point` | Skończone współrzędne w `[0,1]`. | Powiązanie z rozmiarem obrazu i konwencją pikseli. |
| `domain/geometry/polygon.py`: `Polygon` | Minimum 3 różne wierzchołki, niezerowe pole, bez powtórzeń i samoprzecięć. Nie ogranicza do prostokątów. | Ocena semantyczna: poprawny wielokąt nie dowodzi istnienia miejsca. |
| `capture/frame.py`: `Frame` | Klatka BGR, kamera, sekwencja, czas. | Nie przechowuje statycznej geometrii; brak pętli runtime. |
| `detection/slot_detector.py`, `slot_proposal.py` | Protokół `detect(Frame)` i propozycja bez trwałego `slot_id`. | Zatwierdzenie propozycji i przypisanie fizycznego miejsca. |
| `detection/opencv_slot_detector.py` | Działa `extract_segments()` w pikselach; `detect()` wykonuje LSD i zgłasza `NotImplementedError`. | Nie ma działającego składania `SlotProposal` ani automatycznego setupu. |
| `scripts/run_parking_batch.py`, `detection/*` | Działają diagnostyki LSD, grup, dopasowanych linii, par, opcjonalnej topologii/wsparcia i porównania CLAHE. | JSON-y diagnostyczne nie są zatwierdzonymi miejscami ani konfiguracją parkingu. ID odcinków/węzłów są lokalne i nie nadają się na `slot_id`. |
| `research/paint-segmentation` | Osobny pipeline danych, treningu i map farby; raporty eksperymentów. | Maska farby nie jest poligonem miejsca, mapą parkingu ani oceną zajętości. |

Wyszukiwanie całego silnika nie wykazało innego magazynu konfiguracji, bazy,
serwera, API HTTP, UI, homografii ani mapy parkingu. Istniejące zapisy JSON
dotyczyły diagnostyki. Wymienione klasy domenowe były małymi dataclassami;
`NewType` ID pomaga statycznemu typowaniu, lecz sam nie waliduje danych runtime.

## Działający pionowy wycinek

Dodano jeden moduł `src/parkvision/configuration.py`. `ParkingConfiguration` to
koperta jednej rewizji z istniejącymi `Parking`, `Camera`, `ParkingSlot` i
`SlotProjection`, a nie nowy model fizycznego parkingu. Nie zmieniono tych klas.
Nie dodano zależności ani abstrakcyjnego repozytorium.

Kontrakt JSON **schema_version=1**:

- `parking`: `id`, `name`;
- `cameras`: `id`, `parking_id`, `name`, `image_width`, `image_height`;
- `slots`: `id`, `parking_id`, `label` (tekst lub `null`);
- `projections`: `camera_id`, `parking_slot_id`, `polygon`: tablica `{x,y}`;
- `revision`: dodatnia liczba całkowita, osobna od wersji formatu;
- `approved_by`, `approved_at`: deklaracja operatora i czas ISO 8601 ze strefą.

Import i konstruktor sprawdzają unikalność ID kamer/miejsc wewnątrz parkingu,
zgodność `parking_id`, istnienie obu końców relacji i unikalność pary
`(camera_id, parking_slot_id)`. Rejestr zawiera przynajmniej jedną kamerę i jedno
miejsce; każde miejsce objęte zatwierdzonym setupem ma przynajmniej jedną projekcję.
ID są stabilnymi identyfikatorami, nie numerami kolejności w tablicy i nie etykietami.
Przy przyszłym łączeniu plików różnych parkingów należy zachować unikalne ID
parkingów i stosować przestrzeń nazw parkingu dla kamer/miejsc; ten lokalny plik
nie udaje globalnego rejestru ID.

**Widoczność statyczna** jest wyrażona obecnością zatwierdzonej projekcji.
Brak pary oznacza brak skonfigurowanego, użytecznego ROI dla tej kamery, a nie
„wolne”, „zajęte” ani chwilowe zasłonięcie. Ten sam `slot_id` może występować
w wielu kamerach z różnymi poligonami. Identyfikację tego samego fizycznego
miejsca potwierdza operator; kod nie dopasowuje go po podobieństwie geometrii.
Nie przechowujemy procentu zasłonięcia ani zmiennej widoczności z kolejnych klatek.
Nierozstrzygnięte miejsca zostają w roboczych notatkach poza zatwierdzonym zakresem;
nie dopisujemy im zgadywanych poligonów.

Geometria odnosi się do **pełnego obrazu kamery**, po ustalonych transformacjach,
nie do tymczasowego cropa/ROI używanego w eksperymencie ML. Początek jest w lewym
górnym rogu, x w prawo, y w dół. Konwencja v1 opisuje środki skrajnych pikseli:
`x_normalized = x_pixel / (width - 1)`, `y_normalized = y_pixel / (height - 1)`;
odwrotnie mnożymy przez `width - 1`, `height - 1`. Oba wymiary są całkowite i ≥2.
Eksport nie zaokrągla współrzędnych; ostatniego wierzchołka nie powtarzamy.
Współrzędne po cropie/resize trzeba najpierw odnieść do pełnej referencji.
Nie ma automatycznego przeliczania ani „naprawiania” projekcji.

Import odrzuca błędne typy (także bool zamiast liczby), NaN/inf, współrzędne poza
zakresem, wadliwe poligony, nieobsługiwaną wersję, brakujące/nieznane pola oraz
powtórzone klucze JSON. Metadane zatwierdzenia są wymagane. Są **zapisem deklaracji**,
nie uwierzytelnieniem operatora ani dowodem, że przejrzał obraz.
Walidacja formalna nie potwierdza prawidłowego oznaczenia miejsca w świecie.

`load_configuration(Path)` odtwarza modele domenowe. `save_configuration(config,
Path)` zapisuje wyłącznie do nowego pliku (`FileExistsError` przy kolizji).
Zmiana po zatwierdzeniu wymaga nowej rewizji i ponownego przeglądu operatora;
numerację, historię i wybór aktywnego pliku prowadzi na razie osoba obsługująca.
Nie ma automatycznej migracji, wymuszania monotoniczności między różnymi plikami,
wskaźnika „active” ani atomowej publikacji. Przerwany zapis może zostawić wadliwy
nowy plik, ale nie niszczy starej wersji; nowy plik należy ponownie wczytać przed
użyciem. Pliki trzymamy poza katalogami wyników przetwarzania klatek.

Metoda `projections_for_camera(camera_id, image_width=..., image_height=...)`
zwraca zapisane obiekty po kontroli kamery i wymiarów. Nie wywołuje detektora,
nie ogląda pikseli i nie nadaje stanu zajętości. Rozbieżność wymiarów zgłasza błąd;
identyczny rozmiar **nie wykryje** zmiany pozycji, cropa ani obiektywu.

Opcjonalna wizualizacja parkingu powinna w przyszłości przypisywać obiekty mapy do
tego samego `slot_id`, niezależnie od projekcji kamer. Obecnie brak uzgodnionego
układu mapy, skali i transformacji, dlatego schema v1 nie zawiera pozornej
homografii ani kolejnego poligonu udającego geometrię świata. Sama obsługa zapisu
projekcji nie wymaga takiej mapy.

## Pierwszy workflow operatora

1. Wybrać stabilne kamery i zakres miejsc. Zachować prywatnie po jednym dobrym
   obrazie referencyjnym każdej kamery, jego SHA256, wymiary i opis kadrowania
   w notatce przeglądu obok konfiguracji. Nie jest to część danych treningowych.
   Przy zasłonięciu wybrać wcześniejsze/lepsze ujęcie albo odłożyć setup danego
   miejsca; nie trzeba wnioskować jego granic z pojedynczej trudnej klatki.
2. Nadać stałe ID parkingu, kamer i fizycznych miejsc. Ręcznie przypisać wspólne
   miejsca między kamerami. CV/ML mogą dostarczyć opcjonalne podpowiedzi, lecz
   operator może narysować ROI w całości sam.
3. Narysować poligony w lokalnym narzędziu/roboczym JSON, przeliczyć współrzędne
   według konwencji powyżej. Sprawdzić na referencjach zakres każdego miejsca,
   kolejność wierzchołków, sąsiadów, granice kadru i wszystkie powiązania kamer.
   Nie wpisywać stanu zajętości. Jeśli granice są niepewne, wrócić do punktu 1.
4. Po wizualnym sprawdzeniu całego zakresu wpisać `revision=1`, operatora i czas
   zatwierdzenia. Wczytać/walidować plik. Poprawka geometrii po zatwierdzeniu
   wymaga ponownego przeglądu i nowych metadanych, nie tylko zmiany numeru pliku.
5. Wyeksportować do nowego `parking-r1.json`, ponownie wczytać i zachować razem
   z prywatną notatką przeglądu. Do przyszłego runtime wybrać ten plik jawnie.
   Późniejsze klatki używają zapisanych projekcji; nie uruchamiają `detect()`.
6. Przy istotnej zmianie przygotować roboczą kopię, zachować ID niezmienionych
   fizycznych miejsc, przejrzeć zmienione widoki i zatwierdzić `parking-r2.json`.
   Stara wersja zostaje do audytu/odtworzenia. Zmiana jednego widoku nie wymaga
   ponownego odkrywania wszystkich miejsc we wszystkich kamerach.

To workflow plikowy dla programisty/operatora projektu studenckiego.
Nie dostarczono edytora ani nakładki wizualnej w UI. Działający, **wyłącznie
syntetyczny** przykład jest w `examples/parking-setup-v1.json`:

```bash
# Z katalogu głównego repozytorium; tylko odczyt i walidacja.
apps/engine/.venv/bin/python -m parkvision.configuration \
  apps/engine/examples/parking-setup-v1.json
```

Przykład API, po rzeczywistym przeglądzie pliku przez operatora:

```python
from pathlib import Path
from parkvision.configuration import load_configuration, save_configuration
from parkvision.domain.types import CameraId

config = load_configuration(Path("reviewed-parking-r1.json"))
save_configuration(config, Path("approved/parking-r1.json"))  # katalog już istnieje
restored = load_configuration(Path("approved/parking-r1.json"))
assert restored == config
rois = restored.projections_for_camera(
    CameraId("camera-east"), image_width=1280, image_height=720
)
# rois to geometria; przyszły moduł occupancy dostanie ją osobno od klatki.
```

Nie wolno stosować poligonów z syntetycznego przykładu do rzeczywistych zdjęć.

## Weryfikacja i następny krok

Przed zmianami: **755 testów silnika przechodziło**. Nowe testy
`tests/test_configuration.py` obejmują round-trip rzeczywistych obiektów domenowych
na syntetycznej geometrii, 2 kamery/2 miejsca/3 projekcje, niezmienność źródeł,
nową rewizję bez nadpisania, walidację referencji/formatu/poligonów, dopasowanie
wymiarów i CLI z innego katalogu. Syntetyczne klatki o różnej treści korzystają
z identycznej zapisanej geometrii, bez uruchamiania CV/ML.

Końcowy wynik: **811 passed** (755 dotychczasowych + 56 nowych). Przechodzą:
Flake8 całego `src/tests/scripts` silnika, Black obu nowych plików Python oraz
Mypy tych dwóch plików z `--follow-imports=silent`. Użyto istniejącego Mypy
ze środowiska research, bez instalacji lub zmiany zależności.

Polecenia z głównego katalogu repozytorium:

```bash
apps/engine/.venv/bin/python -m pytest -q apps/engine/tests
apps/engine/.venv/bin/python -m flake8 --jobs 1 --config apps/engine/.flake8 \
  apps/engine/src apps/engine/tests apps/engine/scripts
apps/engine/.venv/bin/python -m black --check apps/engine/src/parkvision/configuration.py
apps/engine/.venv/bin/python -m black --check apps/engine/tests/test_configuration.py
research/paint-segmentation/.venv/bin/python -m mypy --follow-imports=silent \
  apps/engine/src/parkvision/configuration.py apps/engine/tests/test_configuration.py
```

Zmieniono wyłącznie nowy moduł, jego testy, syntetyczny przykład, ten review,
plan ewaluacji ML i odsyłacz w README research. Nie zmieniono istniejącego kodu
CV/ML, klas domenowych, danych ani checkpointów. Nie wykonano commita.

Aktualizacja kolejnego etapu: [pierwszy DRAFT z LabelMe](labelme-slot-setup.md)
korzysta z tego samego modelu z obiema metadanymi approval równymi `None`.
To jawny stan roboczy w pamięci, wizualizowany z ostrzeżeniem DRAFT. Zapis/odczyt
zatwierdzonego formatu JSON v1 nadal wymaga operatora i czasu; DRAFT jest
przechowywany wyłącznie w odrębnej anotacji LabelMe i nie przechodzi eksportu.

Kolejny etap dostarczył **[lokalny podgląd zapisanej konfiguracji na referencji
kamery](configuration-preview.md)**, z ID miejsc, etykietami i kontrolą wymiarów,
bez detekcji i zmiany geometrii. Narzędzie używa zatwierdzonego formatu JSON;
nie jest edytorem ani mechanizmem nadawania zatwierdzenia. [Etap LabelMe](labelme-slot-setup.md)
zakończył następnie pierwszy rzeczywisty pilot: użytkownik zatwierdził trzy miejsca
na `parking.jpg`, zapisano rewizję 1 i sprawdzono finalny podgląd.
Plan małego niezależnego badania ML:
`research/paint-segmentation/setup-evaluation.md` względem głównego repo.
