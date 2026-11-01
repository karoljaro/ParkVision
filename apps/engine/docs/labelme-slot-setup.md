# Pierwszy rzeczywisty setup miejsc przez LabelMe

2026-11-01, początkowy HEAD `369e1ba116f921311f7d78a27d2dc7e32ab19f95`,
`main`, czyste drzewo i indeks. Nie wykonano commita.

## Pilot i stan zatwierdzenia

Wybrano `samples/parking.jpg` (1440×960): w lewym rzędzie widać pełne obszary
sąsiednich miejsc oraz ich granice przy alejce i obrzeżu. Porównano 9 próbek
na prywatnym zestawieniu `verification/sample-selection.png`. Ujęcia CNR mają
mniej czytelne granice lub zasłonięcia, a UFPR dalszą perspektywę i wyspy/drzewa;
wybrany obraz spełnia preferencję użytkownika i nie wymaga odgadywania granic
pod pojazdami. Nie korzystano z
anotacji paint/curb, maski ML, LSD, topologii ani automatycznego składania miejsc.

Przygotowano **3 ręcznie wskazane propozycje DRAFT**, całe obszary miejsc:
`parking-sample-left-a`, `parking-sample-left-b`, `parking-sample-left-c`.
Etykiety „Pilot A/B/C — lewy rząd” nie udają odczytu numerów namalowanych na
asfalcie. Są do sprawdzenia i ewentualnej korekty przez użytkownika w LabelMe;
nie trzeba zaczynać rysowania od pustego obrazu.

Po uwadze użytkownika o przecinaniu białych pasów przygotowano poprawiony
`draft-aligned-edges/parking.json` w tym samym prywatnym katalogu pilota.
Indywidualnie poprawiono 8 współdzielonych narożników; użytkownik zaakceptował tę
wersję. Pierwotny DRAFT i wcześniejsze przesunięcie o 2 px zachowano jako
historię. Instrukcja pilota wskazuje nowy plik, a powiększenie
`draft-aligned-edges-detail.png` pozwala porównać cienkie obrysy z pasami farby.
Weryfikacja wspólnych krawędzi, braku nakładania wnętrz i lokalnych przekrojów
widocznych pasów jest w `verification/aligned-edges-check.json`. Anotacja LabelMe
pozostaje źródłem roboczym; zatwierdzona konfiguracja jest osobnym plikiem poniżej.

Parking `parking-sample` i kamera `parking-sample-camera-01` są logicznymi ID
tego pilota. Lokalizacja i sprzętowe ID kamery pozostają nieustalone. Napis
„Camera 01” na obrazie identyfikuje widok, nie dowodzi mapowania do urządzenia.
Nie jest to kompletna konfiguracja całego parkingu ani gotowa kalibracja runtime.

Wszystkie rzeczywiste pliki robocze są poza repo:

`/home/karoljaron/Dokumenty/ChatGPT/System parkingowy/first-parking-setup-2026-11-01-v1/`

- `draft/parking.jpg` — identyczna bajtowo kopia referencji;
- `draft/parking.json` — poligony LabelMe, bez metadanych zatwierdzenia;
- `draft-preview.png` — rzeczywisty obraz z obrysami i napisem DRAFT;
- `approved/parking-r1.json` — zatwierdzona rewizja 1 dla trzech miejsc;
- `approved/reviewed-labelme-r1.json` — identyczna bajtowo kopia przyjętej anotacji;
- `approved-preview-r1.png` — końcowa nakładka z wczytanej zatwierdzonej konfiguracji;
- `verification/` — odczyt/zapis rzeczywistym LabelMe, zrzut jego okna,
  SHA256, kontrola eksportu i wersje środowiska edytora;
- `README.md` — kompletne polecenia przeglądu, poprawiania i zatwierdzenia.

Źródłowy obraz i jego kopia mają SHA256
`e82f1877883eea1a79bc092e23b2b844221855291661fd50abf9e59165188bdf`.
Początkowy DRAFT ma SHA256
`ef2eb0bfb98a343e885ac7e1541e41acd7041966422f7ab13ec988141574a9f8`.
Po edycji JSON trzeba wygenerować i obejrzeć nowy podgląd; jego hash będzie inny.

**Pierwsza rewizja została zatwierdzona i zapisana.** Użytkownik potwierdził
poprawność skorygowanych linii oraz podpis `karoljaron` i użycie bieżącego czasu.
Zapisano `approved_at=2026-11-01T20:07:08+01:00` (Europe/Warsaw).
Przyjęta anotacja ma SHA256
`0201fba39975b2abf31b8be00fbbd2a23e2315ff5e738ce77865d7c0b8b311a6`.
Istniejącym CLI wykonano rzeczywisty eksport, ponowne wczytanie i finalny PNG.
Porównanie potwierdza identyczne ID i geometrię, z błędem round-trip 0 px
dla tych wierzchołków. Wyniki, hashe i metadane:
`verification/approval-r1-check.json` w pilocie. Wcześniejsze kontrole DRAFT
zachowano jako zapis etapów przed akceptacją.

## Minimalny kontrakt importera

Nowy moduł `src/parkvision/labelme_configuration.py` obsługuje jedną kamerę na
plik LabelMe. DRAFT jest przechowywany w tym pliku; nie ma drugiego formatu JSON
konfiguracji ani nowego modelu parkingu.

Standardowe pola LabelMe:

- `imagePath` wskazuje dokładnie podany obraz względem katalogu anotacji;
- `imageWidth`, `imageHeight` zgadzają się z odkodowanym obrazem;
- `imageData=null`, czyli zapis przez `--nodata`;
- `shapes[].shape_type="polygon"`, `group_id=null`, jeden poligon na miejsce;
- `shapes[].label` jest trwałym `slot_id`, niezależnym od kolejności kształtów;
- `shapes[].description` jest opcjonalną etykietą użytkową (pusty tekst → brak);
- `shapes[].points` zawierają oryginalne współrzędne pikseli, bez powtórzenia
  pierwszego wierzchołka na końcu.

Rozszerzenie głównego obiektu `parkvision` zawiera dokładnie `task`, `parking_id`,
`parking_name`, `camera_id`, `camera_name`, `revision`, `image_sha256`.
`task="parking-slot-setup"` odróżnia ten plik od anotacji segmentacji farby.
Metadane wiążą tożsamość parkingu/kamery z hashem rzeczywistej referencji.
LabelMe 5.9.1 zachowuje rozszerzenie jako `otherData`; potwierdzono to odczytem
i ponownym zapisem rzeczywistego pilota przy użyciu jego `LabelFile`.
Własne `parking_id`, `camera_id`, `slot_id` wewnątrz kształtu są odrzucane,
żeby nie tolerować sprzecznych źródeł powiązań.

`load_labelme_configuration(annotation_path, image_path)` zawsze zwraca
**niezatwierdzone** `ParkingConfiguration`, z obiema wartościami approval `None`.
Wykorzystuje istniejące `Parking`, `Camera`, `ParkingSlot`, `SlotProjection`,
`Point`, `Polygon` i walidację relacji konfiguracji. Przelicza współrzędne przez
`x/(width-1)`, `y/(height-1)`; nie przycina ani nie naprawia kształtów. Odrzuca
duplikaty ID/kluczy JSON, złe typy i wymiary, niewłaściwy obraz, osadzone imageData,
NaN/inf, punkty poza obrazem, powtarzane wierzchołki i samoprzecięcia.

Referencja ma być w docelowej orientacji, bez transformacji crop/resize.
W `parking.jpg` sprawdzono brak EXIF Orientation. LabelMe potrafi stosować EXIF
podczas otwierania, a podgląd ParkVision używa surowej orientacji; inne referencje
z rotacją EXIF wymagają najpierw świadomego ustalenia wspólnej referencji.

## DRAFT, wizualizacja i eksport

Istniejący model `ParkingConfiguration` dopuszcza teraz brak obu metadanych
zatwierdzenia jako jawny stan DRAFT w pamięci. Jedno pole bez drugiego jest błędem.
Właściwość `is_approved` rozróżnia oba stany. Nie używamy fikcyjnego operatora,
czasu, pominięcia walidacji ani obejścia konstruktora.

**Zapisany format konfiguracji schema v1 pozostaje zatwierdzony:**
`save_configuration()` odrzuca DRAFT przed utworzeniem pliku, a
`load_configuration()` nadal wymaga prawidłowego operatora i czasu ze strefą
(odrzuca także oba `null`). Dotychczasowe zatwierdzone konfiguracje nie wymagają
migracji. Rejestr kamer/miejsc i walidacja geometrii są wspólne dla obu stanów.

Istniejący renderer `configuration_preview.py` otrzymał wyraźne oznaczenie DRAFT
na zdjęciu i w legendzie. Wydzielono `save_geometry_preview()` dla już
zwalidowanych obiektów domenowych; standardowe CLI z `--configuration` nadal
wczytuje wyłącznie zatwierdzony format. Obraz i JSON nie są nadpisywane.

Importer ma dwa osobne polecenia:

```bash
apps/engine/.venv/bin/python -m parkvision.labelme_configuration preview \
  --annotation /dane/draft/parking.json --image /dane/draft/parking.jpg \
  --output /dane/draft-preview.png

# Dopiero PO faktycznym przeglądzie. Wartości podaje operator.
apps/engine/.venv/bin/python -m parkvision.labelme_configuration approve \
  --annotation /dane/draft/parking.json --image /dane/draft/parking.jpg \
  --reviewed-annotation-sha256 HASH_Z_OBEJRZANEGO_PODGLADU \
  --approved-by IDENTYFIKATOR_OPERATORA --approved-at CZAS_ISO_8601_ZE_STREFA \
  --output /dane/approved/parking-r1.json
```

`preview` wypisuje SHA256 dokładnie wczytanej anotacji. `approve` wymaga tego
hasza i obu jawnych metadanych; nie bierze czasu z zegara ani operatora z systemu.
Zmiana pliku po przeglądzie blokuje zatwierdzenie do ponownego podglądu.
To nadal deklaracja operatora, nie uwierzytelnienie ani techniczny dowód
obejrzenia obrazu. Kod nie podejmuje za człowieka decyzji o akceptacji ROI.
Eksport nie nadpisuje istniejących rewizji. Finalny podgląd powstaje istniejącym
CLI `parkvision.configuration_preview` po wczytaniu zatwierdzonego JSON.

## LabelMe i weryfikacja

W dostępnej sesji nie było wcześniej uruchamialnego LabelMe; stare README
zawierało instrukcję jego instalacji. Przygotowano osobne środowisko
`/tmp/parkvision-labelme-env` z **LabelMe 5.9.1** i zależnościami. Potwierdzono
`--help`, odczyt/zapis `LabelFile`, zachowanie metadanych/ID/etykiet/punktów,
równość modeli po ponownym imporcie oraz otwarcie 3 poligonów w rzeczywistym
oknie Qt w trybie offscreen. Nie zastępuje to przeglądu użytkownika.
Środowiska engine i research nie były zmieniane. Biblioteka ONNX Runtime jest
wymagana przez edytor do uruchomienia; nie załadowano modeli AI ani nie wykonano
inferencji. Dokładne wersje: `verification/labelme-environment.txt` w pilocie.

Po zgłoszeniu awarii przy ruchu myszy przypięto **NumPy 2.3.5 wyłącznie w LabelMe**.
NumPy 2.5.4 nie obsługuje używanego przez tę wersję edytora `np.cross` dla wektorów
2D. Zaktualizowano plik odtwarzania środowiska i sprawdzono `pip check`, obliczanie
odległości, 12 krawędzi oraz 36 zdarzeń ruchu myszy w rzeczywistym oknie LabelMe
w trybie offscreen, bez wyjątków. Wyniki: `verification/labelme-hover-check.json`.
Poprzednia kontrola otwarcia/zapisu nie pokrywała tej interakcji. Po naprawie
trzeba zamknąć starą instancję edytora i uruchomić ją ponownie; NumPy silnika
ParkVision i anotacje pozostają bez zmian.

Testy `tests/test_labelme_configuration.py` wykorzystują struktury LabelMe
na sztucznym obrazie. Sprawdzają dokładne współrzędne, stabilność ID przy zmianie
kolejności, walidację, odróżnianie danych paint, niezmienność wejść, oznaczenie
DRAFT, blokadę niezadeklarowanego/starego zatwierdzenia, brak nadpisania,
CLI oraz pełny zatwierdzony round-trip i finalny PNG na danych syntetycznych.
Przed zgodą użytkownika w rzeczywistym pilocie sprawdzono DRAFT i **blokadę**
końcowego eksportu. Po zgodzie sprawdzono także rzeczywistą zatwierdzoną rewizję,
ponowne wczytanie, identyczność geometrii oraz końcowy PNG z tego pliku.

Końcowo: **878 passed** (837 dotychczasowych + 41 nowych), Flake8 całego silnika,
Black 4 zmienionych/nowych plików Python i Mypy 6 plików źródłowych/testowych
konfiguracji, podglądu i importera. Polecenia z głównego katalogu repo:

```bash
apps/engine/.venv/bin/python -m pytest -q apps/engine/tests
apps/engine/.venv/bin/python -m flake8 --jobs 1 --config apps/engine/.flake8 \
  apps/engine/src apps/engine/tests apps/engine/scripts
apps/engine/.venv/bin/python -m black --check apps/engine/src/parkvision/configuration.py
apps/engine/.venv/bin/python -m black --check apps/engine/src/parkvision/configuration_preview.py
apps/engine/.venv/bin/python -m black --check apps/engine/src/parkvision/labelme_configuration.py
apps/engine/.venv/bin/python -m black --check apps/engine/tests/test_labelme_configuration.py
research/paint-segmentation/.venv/bin/python -m mypy \
  --python-executable apps/engine/.venv/bin/python --follow-imports=silent \
  apps/engine/src/parkvision/configuration.py \
  apps/engine/src/parkvision/configuration_preview.py \
  apps/engine/src/parkvision/labelme_configuration.py \
  apps/engine/tests/test_configuration.py \
  apps/engine/tests/test_configuration_preview.py \
  apps/engine/tests/test_labelme_configuration.py
```

Zachowano wiele kamer w istniejącym modelu. Ten pilot nie zgaduje relacji między
ujęciami; import wielu kamer i łączenie slotów nie są dodawane. Po zatwierdzeniu
geometria jest stała, niezależna od kolejnych klatek i zasłonięć. Przegląd,
zapis r1 i finalna nakładka dla trzech miejsc są wykonane. Przed occupancy
pozostają ustalenie rzeczywistego źródła kamery i osobny kontrakt oceny zajętości
na zapisanych ROI; rozszerzenie zakresu miejsc wymaga kolejnego świadomego setupu.
LSD, topologia, ML, dane, checkpointy i test PUCPR pozostają bez zmian.
