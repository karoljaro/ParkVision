# ParkVision v4.3 Phase 4A — otwarte rzędy i geometria częściowa

14–15 listopada 2026 · Baza lokalna `aeff2f1`

## Wynik

Na pustym UFPR05 inicjalizacja rozpoznaje **jeden lokalny łańcuch dziewięciu separatorów przy schodkowych wysepkach i osiem częściowych hipotez sąsiednich stanowisk**. Pełnych ROI pozostaje zero. W UFPR04 wykrywane są dwa krótsze fragmenty rzędu i pięć częściowych hipotez.

Współrzędne pełnych poligonów są identyczne przed i po zmianie, a pliki LabelMe `draft.json` są bajtowo identyczne we wszystkich 16 porównanych próbach. Pilot zachowuje siedem `estimated` oraz kontrolę A/B/C. Częściowe hipotezy mają `status=unresolved`, `geometry_kind=partial` i `proposal=None`.

![UFPR05 przed i po](</home/karoljaron/Dokumenty/ChatGPT/System parkingowy/geometry-initialization-v4-3-phase4a-2026-11-14/ufpr05-overview.jpg>)

Praca wykonana w bieżącym lokalnym repozytorium. Na początku katalog roboczy był czysty, a Phase 3 znajdowała się w commicie `aeff2f1`. Kod Phase 3 zamrożono w katalogu artefaktów. CNR, ACPDS i PUCPR nie należą do materiału użytego w tej fazie. Role danych odziedziczono z Phase 0–3 i Validation Gate.

## Dlaczego UFPR05 nie dawał ROI

LSD i parowanie farby znajdowały 52 osie oraz 33 węzły. Problem powstawał przy przypisaniu do rzędów. Phase 3 grupowała kotwice według jednego `rail_id` i strony tej linii. Przy schodkowych wysepkach kolejne narożniki mają osobne lokalne linie kontekstowe. Wszystkie 24 wcześniejsze rekordy `RowEvidence` zawierały po jednym separatorze, mimo że ich końce układają się w widoczną sekwencję.

Wykryty łańcuch odpowiada lewej stronie prawego pasa wysepek. Jego końce biegną od około (644, 25) do (1084, 348) px, a kierunek krótkich separatorów zmienia się stopniowo od około 125° do 93°. Każde sąsiedztwo jest oceniane lokalnie.

Rozpoznanie tego układu nie ustala fizycznej głębokości miejsc. Obraz pokazuje krótką farbę, krawężniki, wyspy i przestrzeń jezdni. Koniec farby może oznaczać koniec oznakowania w obrębie stanowiska. Sama krawędź jasnego obiektu lub trawnika nie określa automatycznie granicy powierzchni do parkowania.

Osobno sprawdzono zagrożenie połączenia przeciwległych stron wysepki. Fragment 1875 kończy się około (380, 176), a bliski fragment 1797 zaczyna się około (385, 171) i biegnie dalej przez układ farby i krawężnika. Nowa rekonstrukcja odrzuca ich relację sąsiedztwa jako współliniową. Nie łączy ich w bok pełnego miejsca.

[Kontrole wzrokowe](</home/karoljaron/Dokumenty/ChatGPT/System parkingowy/geometry-initialization-v4-3-phase4a-2026-11-14/independent-review/ufpr05-visual-controls.json>) zawierają cztery wskazane łańcuchy i współrzędne. Nowy wynik odtwarza dziewięć z dziewięciu separatorów oraz osiem z ośmiu kolejnych relacji jednego z tych łańcuchów. Pozostałe trzy sekwencje nie otrzymały wystarczającego potwierdzenia w tej implementacji. Są to kontrole deweloperskie, bez zweryfikowanej głębokości fizycznych miejsc.

## Implementacja

### Struktura otwartego rzędu

Nowy moduł `detection/row_reconstruction.py` rozszerza istniejący przebieg `OpenCVSlotDetector.analyze()`. Korzysta z tych samych `MarkingPair`, węzłów L/T/X, pomiarów wsparcia i grup źródłowych co składanie pełnych ROI.

- Analizuje oba skończone końce sparowanego oznakowania i zachowuje źródłowe kotwice.
- Łączy wzajemnie najbliższych, zgodnie ustawionych sąsiadów. Odrzuca współliniowe fragmenty, skrzyżowania i gwałtowną zmianę relacji obu końców.
- Pozwala na stopniową zmianę orientacji i szerokości. Rozdziela łańcuch przy ostrym skręcie lub ponad dwukrotnej zmianie kolejnego rozstawu.
- Wymaga co najmniej dwóch rozłącznych zestawów lokalnego kontekstu. Powtarzane przecięcia tej samej reprezentacji krawężnika lub karoserii liczą się jako jedno potwierdzenie.
- Dwa oddzielne T mogą utworzyć krótką częściową strukturę. Dwa równoległe odcinki bez dodatkowych kotwic nie wystarczają.
- Zachowuje obserwowane odcinki w ich źródłowym zasięgu. Nie dopisuje arbitralnej głębokości.

Pierwsza kontrola integracyjna wykryła pozorne łańcuchy na karoseriach w ClearPix i wózkach w Ion Alarm. Wymóg rozłącznego kontekstu i kontrola przebiegu lokalnej ścieżki usunęły te wyniki. Powstały osobne testy regresyjne odtwarzające te błędy.

### Rozszerzenie dowodów

Dotychczasowe wymagane pola i konstrukcje modeli zachowano. Nowe pola mają wartości domyślne.

| Obiekt | Nowe informacje |
| --- | --- |
| `RowEvidence` | `geometry_kind`, rodzaje kotwic, identyfikatory lokalnego kontekstu, przeciwległe widoczne końce separatorów, kierunek łańcucha, `row_id`. Otwarty rząd może mieć `rail_id=None`. |
| `SlotCandidate` | `geometry_kind=partial`, `missing_boundaries`, `row_id`. Granice zawierają źródłowe obserwacje, a poligon pozostaje pusty. |
| `evidence.json` | Osobne liczniki pełnych ROI, częściowych hipotez, pozostałych nierozstrzygnięć i nowych otwartych struktur. |
| Podgląd | Oryginał, LSD i osie, topologia i kotwice, struktura rzędów, pełne i częściowe propozycje oraz panel brakujących granic. |

Kotwice `paint_junction` i `marking_corner` pochodzą z par oznakowania. `boundary_context` jest wspartą lokalną linią o nieustalonym znaczeniu fizycznym. Może odpowiadać krawężnikowi, ale sama nazwa nie nadaje jej takiej klasy. `aligned_endpoint` opisuje koniec zgodny z lokalnym układem, bez własnego narożnika.

Częściowe wyniki wskazują m.in. nieznaną głębokość, granicę wjazdu i znaczenie granicy. Kierunek łańcucha nie rozstrzyga kierunku ruchu samochodu.

Hipotezy tej samej pary źródłowych separatorów są scalane przy zachowaniu węzłów i przyczyn alternatyw. Pełny ROI nie otrzymuje dodatkowej częściowej kopii dla tej samej pary oznakowania. Inne istniejące odmowy, np. kreskowanie i kontynuacja separatora, pozostają nierozstrzygnięciami.

## Operator i pełny ROI

Dotychczasowe składanie pełnych poligonów zachowuje reguły Phase 3. Rząd z jedną wspartą granicą i spójnymi skończonymi końcami nadal może otrzymać `estimated`. Nowa rekonstrukcja otwartych rzędów tworzy wyłącznie częściowe dowody. To ograniczenie wynika z braku potwierdzonej fizycznej głębokości w sprawdzonych kadrach.

CLI `geometry_setup propose` nadal zapisuje `draft.json`, `evidence.json` i `preview.png`. W szkicu LabelMe znajdują się pełne poligony. Dla hipotezy częściowej `slot_id` i `polygon` pozostają puste. Operator przegląda ślad i podgląd, a następnie dodaje lub poprawia fizyczne ROI, jeśli może ustalić brakujące granice.

Test zatwierdzenia szkicu zawierającego wyłącznie hipotezy częściowe potwierdza, że pusty zestaw poligonów nie tworzy `ParkingConfiguration`. W tym zadaniu żadna rzeczywista nowa hipoteza nie została automatycznie zatwierdzona.

## Porównanie rzeczywistych obrazów

Liczba nowych łańcuchów nie jest liczbą niezależnych parkingów ani kompletnych fizycznych rzędów. W UFPR04 dwa wyniki są lokalnymi fragmentami widocznego pasa.

| Obraz | Nowe otwarte łańcuchy Phase 3 → 4A | Częściowe hipotezy Phase 3 → 4A | Pełne O/E Phase 3 → 4A | Inne U Phase 3 → 4A |
| --- | --- | --- | --- | --- |
| [UFPR05 pusty](</home/karoljaron/Dokumenty/ChatGPT/System parkingowy/geometry-initialization-v4-3-phase4a-2026-11-14/comparisons/ufpr05-empty.jpg>) | 0 → 1 | 0 → 8 | 0/0 → 0/0 | 1 → 0 |
| [UFPR04 pusty](</home/karoljaron/Dokumenty/ChatGPT/System parkingowy/geometry-initialization-v4-3-phase4a-2026-11-14/comparisons/ufpr04-empty.jpg>) | 0 → 2 | 0 → 5 | 0/0 → 0/0 | 1 → 0 |
| [Pilot](</home/karoljaron/Dokumenty/ChatGPT/System parkingowy/geometry-initialization-v4-3-phase4a-2026-11-14/comparisons/cctv-pilot.jpg>) | 0 → 1 | 0 → 12 | 0/7 → 0/7 | 19 → 6 |
| [ClearPix](</home/karoljaron/Dokumenty/ChatGPT/System parkingowy/geometry-initialization-v4-3-phase4a-2026-11-14/comparisons/clearpix.jpg>) | 0 → 0 | 0 → 0 | 0/0 → 0/0 | 1 → 1 |
| [Family Dollar](</home/karoljaron/Dokumenty/ChatGPT/System parkingowy/geometry-initialization-v4-3-phase4a-2026-11-14/comparisons/family-dollar.jpg>) | 0 → 0 | 0 → 0 | 0/0 → 0/0 | 1 → 1 |
| [Ion Alarm](</home/karoljaron/Dokumenty/ChatGPT/System parkingowy/geometry-initialization-v4-3-phase4a-2026-11-14/comparisons/ion-alarm.jpg>) | 0 → 0 | 0 → 0 | 0/0 → 0/0 | 1 → 1 |

O/E oznacza `observed` / `estimated`. „Inne U” pomija jawnie wydzielone hipotezy częściowe, które również mają status `unresolved`. Część dawnych ogólnych odmów zastąpiono bogatszym dowodem częściowym, a powtarzające się pary źródłowe scalono. Spadek U nie oznacza automatycznego odzyskania pełnego stanowiska.

UFPR05 i UFPR04 są dwiema kamerami tego samego fizycznego parkingu. ClearPix i Family Dollar pozostają wcześniej ocenianymi sesjami diagnostycznymi. Ion to pojedynczy lokalny kadr wideo. Nie wyznaczano nowego niezależnego holdoutu.

### Poprawność i ograniczenia pomiaru

Na UFPR05 dziewięć separatorów i osiem kolejnych sąsiedztw odpowiada wskazanemu wizualnie łańcuchowi. W tym łańcuchu nie ma powtórzonych ID ani przeskoczonego separatora. W UFPR04 dwa lokalne odcinki przebiegają przy widocznych oznakowaniach wysepek. Przeznaczenie i dokładna fizyczna głębokość pozostają do kontroli operatora.

Dwanaście hipotez pilota jest materiałem do przeglądu, a nie dwunastoma dodatkowymi potwierdzonymi miejscami. Nie obliczano globalnego recall ani IoU dla częściowych danych bez poligonu. Rozłączne identyfikatory kontekstu ograniczają wielokrotne liczenie jednej reprezentacji krawędzi, a jej fizyczne znaczenie nadal wymaga weryfikacji.

Nie powstały nowe pełne ROI ani nowe fałszywe pełne poligony względem baseline. Pełne propozycje nie nakładają się na siebie. Źródłowe pary częściowych hipotez są unikalne po scaleniu. ClearPix i Ion nie zwracają wcześniej zaobserwowanych fałszywych łańcuchów karoserii. Pozostaje możliwość mylnej interpretacji lokalnej geometrii w innych scenach, co jest jednym z powodów utrzymania statusu częściowego.

A/B/C dopasowano 1:1 do wcześniejszej konfiguracji operatora. Wyniki IoU wynoszą **0,95998 / 0,96111 / 0,96424**. Geometria i identyfikatory szkiców są identyczne z Phase 3. Odczyt zatwierdzonej konfiguracji zachował fizyczne ID A/B/C.

## Dziesięć prób syntetycznych

| Przypadek | Otwarte łańcuchy 4A | Częściowe hipotezy 4A | Pełne O/E 4A | Sprawdzona własność |
| --- | ---: | ---: | --- | --- |
| Dwie poprzeczne granice | 0 | 0 | 3/0 | Zachowana obsługa zamkniętych stanowisk. |
| Jedna granica, otwarty wjazd | 0 | 0 | 0/3 | Istniejące końce i wspólna granica wystarczają do oszacowania domknięcia. |
| Schodkowy kontekst wysepki | 1 | 3 | 0/0 | Kotwice mają osobne konteksty, bez nadania klasy krawężnika. |
| Oddzielne T | 1 | 3 | 0/0 | Struktura rzędu przy braku wspólnej długiej linii. |
| Nierówny rozstaw w perspektywie | 1 | 3 | 0/0 | Lokalna zgodność bez jednakowej szerokości. |
| Zasłonięte końce | 1 | 3 | 0/0 | Obserwacje pozostają w źródłowym zasięgu. |
| Niejednoznaczna głębokość | 1 | 3 | 0/0 | Brak automatycznego poligonu. |
| Kreskowanie w większym obrysie | 0 | 0 | 2/0 | Dwa poprawne miejsca obok kreskowania, bez fałszywych częściowych miejsc. |
| Sam krawężnik | 0 | 0 | 0/0 | Brak hipotez miejsc. |
| Zmiana kierunku i koniec rzędu | 2 | 4 | 0/0 | Dwie lokalne struktury, bez połączenia przez ostry skręt. |

Przed zmianą wszystkie przypadki bez wspólnej poprzecznej linii pozostawiały pojedynczy ogólny rekord odmowy. Pełne poligony każdej próby syntetycznej pozostały identyczne. [Porównania syntetyczne](</home/karoljaron/Dokumenty/ChatGPT/System parkingowy/geometry-initialization-v4-3-phase4a-2026-11-14/comparisons>) pokazują te same kadry i skalę.

## Weryfikacja

| Sprawdzenie | Wynik |
| --- | --- |
| [Pytest silnika](</home/karoljaron/Dokumenty/ChatGPT/System parkingowy/geometry-initialization-v4-3-phase4a-2026-11-14/pytest-engine.log>) | **985 zaliczonych**, w tym 14 testów nowego modułu i regresje rzeczywistego UFPR05 oraz eksportu częściowego szkicu. |
| [Pytest research](</home/karoljaron/Dokumenty/ChatGPT/System parkingowy/geometry-initialization-v4-3-phase4a-2026-11-14/pytest-research.log>) | **61 zaliczonych, 3 pominięte** przez istniejące warunki dostępu do historycznych artefaktów. |
| [Black](</home/karoljaron/Dokumenty/ChatGPT/System parkingowy/geometry-initialization-v4-3-phase4a-2026-11-14/black.log>) | Osiem zmienionych lub nowych plików Python zgodnych. |
| [Flake8](</home/karoljaron/Dokumenty/ChatGPT/System parkingowy/geometry-initialization-v4-3-phase4a-2026-11-14/flake8.log>) | Całe `src`, `tests` i `scripts` bez zgłoszeń. |
| `git diff --check` | Bez zgłoszeń. |
| [LabelMe → approve → konfiguracja](</home/karoljaron/Dokumenty/ChatGPT/System parkingowy/geometry-initialization-v4-3-phase4a-2026-11-14/operator-workflow.log>) | LabelMe 5.9.1 otworzył trzy syntetyczne pełne propozycje, przesunął punkt o 1,125 px i zapisał dwie. Stary hash odrzucono. Po testowym zatwierdzeniu zachowano oba ID i współrzędne z tolerancją 10⁻⁹ px. |
| Częściowy szkic | Test CLI potwierdza widoczność dowodów, zero kształtów LabelMe i odmowę zatwierdzenia pustej konfiguracji. |
| Regresja eksportu | Wszystkie 16 `draft.json` oraz wszystkie pełne wielokąty są identyczne przed i po zmianie. |

## Zmienione pliki

- `apps/engine/src/parkvision/detection/row_reconstruction.py` — rekonstrukcja lokalnych łańcuchów i częściowych obserwacji.
- `apps/engine/src/parkvision/detection/initialization.py` — rozszerzenia dowodów z wartościami domyślnymi.
- `apps/engine/src/parkvision/detection/opencv_slot_detector.py` — integracja przez normalny przebieg analizy.
- `apps/engine/src/parkvision/geometry_setup.py` — eksport i osobne liczniki.
- `apps/engine/src/parkvision/geometry_setup_preview.py` — pięć warstw wizualnych i panel brakujących granic.
- `apps/engine/tests/detection/test_open_rows.py` — nowe regresje dodatnie i negatywne.
- `apps/engine/tests/detection/test_opencv_slot_detector.py` — kontrola rzeczywistego UFPR05.
- `apps/engine/tests/test_geometry_setup.py` — ochrona eksportu i zatwierdzania częściowej hipotezy.
- `apps/engine/docs/geometry-initialization.md`, `geometry-phase4a.md` — instrukcja i niniejszy raport.

## Wznowienie i dalsze ograniczenia

[Artefakty](</home/karoljaron/Dokumenty/ChatGPT/System parkingowy/geometry-initialization-v4-3-phase4a-2026-11-14>) zachowują `baseline-code`, manifesty wejść, `before`, `after`, ich odpowiedniki syntetyczne, `comparison.json`, `abc-regression.json`, skrypty i logi. `run_diagnostics.py` uruchamia normalne `propose` w nowym katalogu, a `compare.py` odtwarza wizualizacje z zapisanych dowodów.

~~~bash
cd /home/karoljaron/Projekty/ParkVision
OUT='/home/karoljaron/Dokumenty/ChatGPT/System parkingowy/geometry-initialization-v4-3-phase4a-2026-11-14'
PYTHONPATH=apps/engine/src apps/engine/.venv/bin/python \
  "$OUT/run_diagnostics.py" "$OUT/cases.json" "$OUT/repeated-review"
~~~

Rozwiązanie odzyskuje użyteczną lokalną strukturę UFPR05, ale nadal pomija inne widoczne sekwencje o niewystarczającym wsparciu kotwic. Mocna dystorsja, długa przerwa, rozbite konteksty i zasłonięte oznakowanie mogą zakończyć lokalny łańcuch. Funkcja przestrzeni przy krawężniku, głębokość, strona wjazdu i granice przy wyspach mogą wymagać decyzji operatora. Wyznaczenie pełnych ROI na podstawie tych częściowych danych pozostaje otwarte.

Wynik jest rozwojową kontrolą geometrii na znanych materiałach. Nie ustanawia generalizacji na nowym parkingu. Nie wykonywano treningu, ponownego audytu, łączenia klatek ani eksperymentu pusty → zapełniony. Moduły pojazdów i zajętości, anotacje historyczne, manifesty i splity nie należą do zmienionych plików. Zmiany pozostają w lokalnym katalogu roboczym, bez commita, pushu i PR.
