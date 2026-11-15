# ParkVision v4.3 Phase 4B — inicjalizacja wieloklatkowa

Implementacja i sprawdzenie lokalne z 2026-11-15. Punktem odniesienia jest czysty
commit `68cad8d4692b874263af1133b01c3193db72b252`, zawierający Phase 4A.
Bez commit/push/PR, treningu, zmian historycznych manifestów i konfiguracji.

## Mechanizm

`analyze_frames` przyjmuje referencję oraz obserwacje tej samej kamery i źródła.
Każdy obraz przechodzi osobne wykrywanie. Zaakceptowane obserwacje geometryczne
są transformowane do referencji, ostrożnie grupowane i przekazywane do istniejącej
topologii, pomiarów wsparcia, składania ROI i rekonstrukcji otwartych rzędów.
Nie powstaje kompozyt obrazów ani masek samochodów. Modele domenowe pozostają bez zmian.
`OpenCVSlotDetector.detect(frame)` i `geometry_setup propose` zachowują zgodność.

Każda klatka zachowuje źródło, kamerę, identyfikator, SHA-256 pikseli i opcjonalnie
pliku, transformację, pomiary rejestracji oraz powód przyjęcia lub odrzucenia.
Obserwacje zachowują oryginalne fragmenty LSD i współrzędne, osie/krawędzie źródłowe,
współrzędne referencyjne i nowe identyfikatory surowych odcinków.
`source_topology`, `markings`, `candidate_sources` i `hypothesis_updates` wiążą
wynik z dowodami. Niesparowany kontekst nie staje się automatycznie farbą ani krawężnikiem.

## Sprawdzenie widoku

- Niezgodne identyfikatory źródła/kamery oraz rozdzielczości są odrzucane przed fuzją.
- Identyczne piksele nie dostarczają drugiej obserwacji. Bliskie duplikaty są
  pomijane tylko przy małej różnicy globalnej i maksymalnie trzech poziomach
  zmiany intensywności w pełnej rozdzielczości. Mały nowy narożnik nie znika
  z powodu samej średniej różnicy całego obrazu.
- Różne obrazy wymagają jawnej maski nieruchomego tła uint8 o wymiarach referencji.
  Należy wskazać konstrukcje, zabudowę, słupy i znaki widoczne we wszystkich klatkach,
  z pominięciem samochodów, powierzchni parkowania i roślinności. Maska jest deklaracją
  operatora, nie automatycznym rozpoznaniem semantycznym tła.
- ORB, wzajemne dopasowania i RANSAC wyznaczają transformację podobieństwa,
  bez homografii. Maksymalny wymiar roboczy to 1200, błędy są w pikselach źródłowych.
- Wymagane są przynajmniej 25 inlierów, udział 0.60, p90 błędu do 1.5 px,
  otoczka punktów obejmująca co najmniej 15% obrazu i sześć komórek siatki 4×3.
- Dopuszczalna zmiana skali wynosi 0.5%, obrót 0.75°, przesunięcie
  `max(8 px, 0.012 × przekątna obrazu)`. Progów nie rozluźniano pod nagrania.

Same wymiary i deklaracja identyfikatora nie potwierdzają widoku.
Mały błąd na jednym wąskim pasie również nie wystarcza.

## Łączenie i niepewność

Osie muszą zgadzać się położeniem, kierunkiem, skończonym zakresem i szerokością
oznaczenia. Zgodność wszystkich oryginalnych obserwacji zapobiega przechodniemu
dryfowi. Tolerancja kąta to 3°, odległość zależy od skali i szerokości farby,
a stosunek szerokości musi mieścić się w 0.65–1.55. Podobny kierunek nie łączy
równoległych sąsiednich separatorów. Kontekst również wymaga zgodności wszystkich osi.

Nowy odcinek pasujący do kilku dawnych fragmentów może je połączyć tylko wtedy,
gdy jego obie oryginalne grupy krawędzi LSD wspierają przerwę. Dopuszczalny jest
lokalny ubytek wielkości szerokości farby na skrzyżowaniu, nie długa nieobserwowana
dziura. Dopasowana obwiednia nie zastępuje pomiaru pokrycia. Istniejące zabezpieczenia
przed kreskowaniem, wewnętrznym separatorem, duplikatami, błędnymi poligonami,
wyjściem poza obraz i nieuzasadnioną głębokością nadal działają we wspólnym składaniu.

`multiple_frames` oznacza pochodzenie z kilku klatek, nie prawdopodobieństwo ani
wzrost pewności. `not_detected_in_frames` oznacza nieznaną widoczność. Uzupełniające
się zakresy są rozpoznawane bez twierdzenia, że brak detekcji potwierdza zasłonięcie.
Nieznane granice pozostają w `missing_boundaries`, a częściowa hipoteza ma nadal
`status=unresolved` i `proposal=None`. Nie ma arbitralnej głębokości wjazdu.

Pełne szkice referencyjne zachowują poligon i identyfikator, dopóki nowy rzeczywiście
zaobserwowany separator wewnętrzny nie zakwestionuje ich zakresu. Wtedy pozostają
nierozstrzygnięte do oceny operatora. Nakładające się nowe propozycje nie duplikują
zachowanych szkiców. Niezrekonstruowane częściowe dowody referencji również są
zachowywane. Dla częściowych hipotez zapisywane są brakujące granice przed/po i
informacja, czy rzeczywiste nowe wsparcie pozwoliło zamknąć ROI.
Wcześniej zatwierdzone konfiguracje nigdy nie są modyfikowane.

## CLI i operator

Nowy podtryb to `python -m parkvision.geometry_setup propose-sequence`.
Argumenty to `--manifest`, `--output-dir`, `--parking-id`, `--parking-name`,
`--camera-id`, `--camera-name` oraz opcjonalny `--revision`.
Ścieżki w manifeście są względem jego pliku. Pierwsza klatka jest referencją.
Każdy wpis musi mieć własne identyfikatory. Limit wynosi 32 klatki.

```json
{
  "role": "diagnostic_only",
  "background_mask": "static-background.png",
  "frames": [
    {"image": "frame-a.png", "source_id": "source-1", "camera_id": "camera-1"},
    {"image": "frame-b.png", "source_id": "source-1", "camera_id": "camera-1"}
  ]
}
```

Wyniki to `draft.json`, `evidence.json`, `preview.png` i `sequence-preview.png`.
Podgląd sekwencji pokazuje osie/narożniki źródeł, powody odmowy, wspólne dowody
na zielono, jednoklatkowe na złoto i sprzeczne na czerwono. Standardowy podgląd
zachowuje rzędy, pełne/częściowe hipotezy i brakujące granice.
Tylko pełne poligony trafiają do LabelMe. Korekta i zatwierdzanie używają
dotychczasowego workflow i formatu `ParkingConfiguration`.
Automatyczny wynik nigdy nie dostaje `operator_verified`.

## Rzeczywiste materiały i wynik

Artefakty są poza Git w
`/home/karoljaron/Dokumenty/ChatGPT/System parkingowy/geometry-initialization-v4-3-phase4b-2026-11-15/`.
`inventory/inventory.json` zawiera rzeczywiste ścieżki filmów/klatek, hashe,
wymiary, kamery, pochodzenie i role. Oryginały pozostały bez zmian.
Pierwsze wyeksportowane klatki są pikselowo zgodne z wcześniejszymi referencjami.

| Źródło | Czasy [s] | Rozdzielczość | Osie w pojedynczych klatkach | Osie po fuzji | Pełne ROI single → multi | Przyjęte dodatkowe |
| --- | --- | --- | --- | --- | --- | --- |
| ClearPix | 18/160/240/300 | 1920×1080 | 22/28/35/48 | 22 | 0 → 0 | 0/3 |
| Family Dollar | 0/450/535 | 3840×1600 | 25/7/4 | 25 | 0 → 0 | 0/2 |
| Ion Alarm | 0/240/350 | 1920×1440 | 93/108/96 | 93 | 0 → 0 | 0/2 |

To osie oznaczeń, także granic poprzecznych i fragmentów, nie liczba niezależnie
potwierdzonych fizycznych separatorów. Każda klatka ma osobny przegląd w
`final/<source>/single-N`. Późniejsze klatki Ion osobno dają cztery i jedną częściową
hipotezę. Nie są przenoszone przez niespełnione sprawdzenie widoku. Każdy wynik
po próbie fuzji zachowuje jeden ogólny nierozstrzygnięty przypadek referencji.

ClearPix ma p90 1.524–1.544 px i tylko 4.4–5.7% pokrycia otoczką w ścisłych
maskach konstrukcji. Family Dollar ma zaledwie 23.3–29.3% inlierów mimo szerokiego
pokrycia. Ion ma mały błąd, lecz niewystarczające pokrycie lub rozłożenie komórek.
Wszystkie siedem dodatkowych klatek rzeczywistych zostało odrzuconych.
Nie wykonano rzeczywistych połączeń między nimi, więc nie ma podstaw do liczenia
ich jako poprawne/błędne. Nie wykazano poprawy pokrycia ani skuteczności CCTV.

Źródła są historycznymi diagnostykami, nie niezależnym benchmarkiem ROI.
UFPR04 i UFPR05 to osobne kamery jednego parkingu. Nie łączono ich.
Chronione zbiory, validation_only i nowe archiwa geometrii pozostały nietknięte.
Nie powtarzano audytu datasetów.

## Kontrolowany wynik i regresja Phase 4A

Sekwencja syntetyczna używa rzeczywistego LSD, ORB, rejestracji, fuzji i eksportera.
Różne nieprzezroczyste zasłonięcia chowają przeciwległe separatory. Każda klatka
daje dwa poprawne pełne szkice. Fuzja daje trzy `observed` dla trzech narysowanych
stanowisk. Wizualna kontrola deweloperska znajduje zero dodatkowych, brakujących
i zduplikowanych stanowisk po fuzji, wobec jednego brakującego w każdej klatce.
To kontrolowany test mechanizmu, nie skuteczność na rzeczywistym CCTV.

Wynik ma siedem osi, w tym krótki izolowany fragment granicy poprzecznej.
Cztery osie mają źródła wieloklatkowe. Fizyczne cztery separatory i dwie poprzeczne
granice są widoczne łącznie. Rejestracja ma 1930/1930 inlierów i p90 równy zero.
Liczba narożników spada z 12 do 11 po deduplikacji. Trzecie ROI odzyskuje wsparcie
ciągłości granicy, a nie dodatnią różnicę każdej statystyki narożników.
Nie ma konfliktów ani ROI poza obrazem.

Wszystkie pola geometrii/dowodów i semantyczne szkice są zgodne z Phase 4A na
sześciu rzeczywistych i dziesięciu syntetycznych przypadkach jednoklatkowych.
Sześć rzeczywistych szkiców jest identycznych bajtowo. Różnice syntetycznych
szkiców dotyczą wyłącznie względnego `imagePath`. Zamrożone źródła bazowe
porównano z bieżącym HEAD. IoU A/B/C pozostaje
0.959984 / 0.961105 / 0.964244 wobec zatwierdzonej kontroli deweloperskiej.
Pilot zachowuje siedem pełnych szkiców i dwanaście częściowych, UFPR05 lokalny
łańcuch dziewięciu separatorów i osiem częściowych, UFPR04 pięć częściowych.
Kontrole kreskowania, krawężnika i niejednoznacznej głębokości zachowują wyniki.

## Testy, artefakty i pliki

Pytest silnika ma **1021 passed**, w tym 36 nowych testów Phase 4B.
Black dla siedmiu zmienionych plików Python, Flake8 dla src/tests/scripts oraz
`git diff --check` przechodzą. Szczegóły są w `final-verification.json` i logach.
Regresja research ma 61 passed / 3 skipped.
Wykorzystano istniejące Pillow przez tymczasową ścieżkę importu, bez instalacji.
Testy obejmują rzeczywisty ORB, izolowaną fuzję geometrii, odmowy, duplikaty,
uzupełniające separatory/narożniki, otwarte T, perspektywę, oświetlenie, skończone
pokrycie, nieznane granice, konflikty i trwałość identyfikatorów. Testy jednostkowe
fuzji jawnie zastępują rejestrację. Test całościowy syntetyczny jej nie zastępuje.

LabelMe 5.9.1 rzeczywiście otworzył trzy syntetyczne propozycje, przesunął narożnik,
usunął jedną propozycję i zapisał dwie. Nieaktualny hash został odrzucony.
Zatwierdzenie i odczyt zachowały identyfikatory i współrzędne z tolerancją 1e-9 px.
To zatwierdzenie testowe nie akceptuje żadnej rzeczywistej geometrii parkingu.

`comparison.json`, `regression.json`, `operator-workflow.log`,
`run_final_diagnostics.py`, `baseline-68cad8d.tar` i `STATUS.json` pozwalają wznowić pracę.
Demonstracja jest w `final/synthetic/multi/sequence-preview.png`, odmowy rzeczywiste
w `final/<source>/multi/sequence-preview.png`.

Zmienione pliki produkcyjne to `detection/frame_registration.py`,
`detection/multi_frame.py`, `detection/opencv_slot_detector.py`,
`geometry_sequence_setup.py` i `geometry_setup.py`. Nowe testy to
`tests/detection/test_multi_frame.py` i `tests/test_geometry_sequence_setup.py`.
Dokumentacja obejmuje ten raport i przewodnik inicjalizacji.

## Pozostałe ograniczenia

Mechanizm działa na kontrolowanych dowodach wieloklatkowych. Rzeczywista fuzja
wymaga lepiej rozłożonych statycznych punktów odniesienia albo osobno uzasadnionej
metody rejestracji. Obecne słabe dopasowania są celowo odrzucane. Nie potwierdzamy
kamery wyłącznie wyglądem, nie odtwarzamy nigdy niewidocznej granicy wjazdu,
nie klasyfikujemy zasłonięcia automatycznie i nie ustalamy niezależnego FP/FN/recall
CCTV. Operator nadal ocenia maski i geometrię. Nie wykonano eksperymentu Phase 5.
