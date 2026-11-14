# ParkVision v4.3 Phase 3 — odporność i pokrycie geometrii ROI

14 listopada 2026 · Lokalna baza `fd42e00` · Wynik deweloperski

## Wynik

Poprawiono odporność inicjalizacji na słabą jasną farbę, cienie i przerwy farby przy węzłach. Zablokowano dwa fałszywe stanowiska, które Phase 2 tworzyła z ukośnego kreskowania w próbie syntetycznej. Składanie używa jednego limitu naprawy końca wspólnego separatora. Zagnieżdżony konkurencyjny obrys pozostaje konfliktem do przeglądu.

**Nie potwierdzono zwiększenia pokrycia ROI na innych rzeczywistych parkingach.** Pilot nadal ma siedem szkiców, a pozostałe siedem sprawdzonych widoków daje zero ROI. Z trzech wcześniejszych miejsc A/B/C odtworzono trzy, z pięciu miejsc ClearPix zero, z miejsca C w Family Dollar zero. Poprawę potwierdzono na kontrolowanych przypadkach syntetycznych i w regresjach kodu. Do uznania celu generalizacji za osiągnięty potrzebna jest dalsza walidacja na odrębnym parkingu.

[Artefakty porównania](</home/karoljaron/Dokumenty/ChatGPT/System parkingowy/geometry-initialization-v4-3-phase3-2026-11-14>) zawierają zamrożony kod Phase 2, obrazy, manifesty przypadków, kompletne dowody oraz skrypty odtwarzające pomiary. Duże pliki są poza repozytorium.

## Punkt odniesienia i dobór materiału

Przed pracą lokalny `main` był czysty i zawierał Phase 2 w commicie `fd42e00`. Punkt odniesienia odtworzono z kopii tego kodu na tym samym interpreterze Python 3.15.2, OpenCV 5.0.0 i NumPy 2.5.4. Pilot odtworzył 1 866 odcinków LSD, 69 osi, 106 węzłów, siedem `estimated` i 19 `unresolved`.

Przed wyborem przeczytano raport Phase 0 i kod Validation Gate v4.0. Nowe ZIP-y mają istniejące role `conditional` lub `diagnostic_only`. Materiał AVM/BEV oraz UAV pominięto w ocenie CCTV. Korzystano z istniejących lokalnych obrazów i pojedynczego kadru Ion Alarm wyodrębnionego z dostępnego MKV.

| Przypadek | Pochodzenie i rola | Geometria odniesienia |
| --- | --- | --- |
| Pilot `parking.jpg` | Już używany materiał deweloperski Phase 1/2. | Trzy miejsca A/B/C zatwierdzone w `parking-r1.json`. Siedem propozycji Phase 2 podlega osobnemu przeglądowi. |
| UFPR04 pusty | Gate `native:ff8567a6c30d3acaf04e1f14`, historyczna rola train. | Źródłowe etykiety pominięto jako GT fizycznego footprintu. |
| UFPR05 pusty | Gate `native:525de75ba2ae99fa07f88143`, historyczna rola train. Ten sam fizyczny parking co UFPR04, druga kamera. | Źródłowe etykiety pominięto jako GT fizycznego footprintu. |
| ClearPix, referencja 18 s | Gate `session:clearpix:18`, wcześniej oceniana sesja diagnostyczna. | Pięć zatwierdzonych miejsc z R2. Region `clearpix-front-06` ma niepotwierdzoną funkcję parkingową i jest wyłączony. |
| Family Dollar, kadr 0 s | Gate `session:family-dollar:0`, wcześniej oceniana sesja diagnostyczna. | Tylko zatwierdzone miejsce C z R3, dziewięć wierzchołków opisujących dystorsję. T-01 jest strefą nieparkingową. |
| Ion Alarm, kadr 0 s | Jedna klatka lokalnego, wcześniej wybranego wideo. | Brak zatwierdzonej geometrii. Wyłącznie przegląd diagnostyczny. |
| CNR kamera 8 i kamera 3 mokra | `validation_only`. Sprawdzone po zamrożeniu decyzji geometrycznych. Miejsce było oglądane we wcześniejszych etapach. | Brak niezależnie zweryfikowanych pełnych ROI w tej próbie. |

Osiem obrazów reprezentuje różne widoki, przy czym dwie kamery UFPR należą do jednej lokalizacji, a dwie CNR do drugiej. Żaden z tych materiałów nie stanowi nowego niezależnego benchmarku CCTV. [Manifest przypadków](</home/karoljaron/Dokumenty/ChatGPT/System parkingowy/geometry-initialization-v4-3-phase3-2026-11-14/cases.json>) i [manifest CNR](</home/karoljaron/Dokumenty/ChatGPT/System parkingowy/geometry-initialization-v4-3-phase3-2026-11-14/cases-validation.json>) zawierają dokładne ścieżki. Sumy SHA-256 są w podsumowaniach prób.

## Rzeczywiste przyczyny błędów

| Etap / rodzaj błędu | Dowód | Wniosek i działanie |
| --- | --- | --- |
| Wykrycie farby po LSD | Na kontrolowanym rzędzie o kontraście 23/255 LSD znajduje 21 odcinków. Phase 2 odrzuca wszystkie pary przez bezwzględny próg kontrastu 31. | Dodano ocenę słabego kontrastu względem szumu i tła po obu stronach. Dwie przeciwne krawędzie i ich wsparcie nadal są wymagane. |
| Narożnik T przy grubszej farbie | W próbie nierównych szerokości przerwy źródłowych odcinków przy przecięciu wynoszą około 4,6–5,4 px. Stałe okno 2,5 px i tolerancja 2 px wykluczają poprawne węzły, choć przecina je pas farby o szerokości około 7 px. | Wsparcie węzła uwzględnia połowę szerokości przecinającego pasa i sinus kąta. Limit do 10 px oraz końcowa kontrola pokrycia granic pozostają aktywne. |
| Fałszywe sąsiedztwo na kreskowaniu | Trzy ukośne kreski wewnątrz większego obrysu tworzą pozorny lokalny rząd, a ich końce pozwalają Phase 2 domknąć dwa małe ROI. | Domknięcie jest wstrzymane, jeśli tuż za końcami przebiega inna wsparta poprzeczna farba. Odległość dotyczy bliższej krawędzi pasa. |
| Wspólna granica przy nierównych szerokościach | Dla sąsiadów o szerokości 40 i 80 px ten sam skrócony separator może osobno uzyskać różne budżety naprawy. | Koniec separatora naprawiany jest raz dla lokalnego rzędu, według mniejszego z dwóch sąsiednich rozstawów. |
| Duplikat a konkurencyjny obrys | Dawna selekcja usuwała obrys zawierający mniejszy wielokąt przy wysokim przecięciu względem mniejszego pola, nawet gdy IoU wynosiło około 0,08. | Duplikat wymaga IoU co najmniej 0,80. Zagnieżdżone alternatywy zostają `unresolved` z powodem konfliktu. |
| Brak wspólnej granicy rzędu | W UFPR04 wszystkie 17 zapisanych stron lokalnych linii ma tylko jeden wsparty separator. UFPR05 ma 24 takie przypadki. Krawężniki i wyspy rozdzielają sąsiedztwo. | Więcej osi nie daje wspólnej granicy wymaganej przez obecny model składania. Pozostało zero ROI. |
| Zasłonięcia i nieobserwowalność | ClearPix i Ion mają wiele linii karoserii oraz zasłoniętej farby. Family Dollar ma wyblakłe oznakowanie i silną dystorsję. | Z pojedynczej klatki nie udało się potwierdzić rzędów. Hipotezy pozostają nierozstrzygnięte. |
| Prawidłowe odrzucenie | Cztery odrzucenia „continuation” pilota mają oryginalne odcinki za pozornym końcem separatora. Dłuższa luka syntetyczna przekracza dopuszczalny budżet. | Zachowano te odrzucenia. |

## Zmiany implementacyjne

- `marking_fragments.py` ocenia słabe jasne pasy przez lokalny stosunek sygnału do szumu, zgodność tła po obu stronach i potwierdzony spadek jasności. Pomiar szumu zaczyna się za zboczem farby, aby samo zbocze nie było liczone jako szum.
- `slot_assembly.py` uwzględnia szerokość farby przy pomiarze węzłów, ustala wspólną naprawę końca separatora, blokuje niejednoznaczne domknięcia przed poprzeczną farbą oraz rozróżnia duplikaty i konflikty.
- `initialization.py` dodaje `RowEvidence`. Dowody rzędu zawierają linię poprzeczną, stronę, separatory, węzły, pozycje, zgodność kolejnych par i uzasadnienie. Szkice zachowują identyfikatory separatorów oraz pozycje kotwic.
- `geometry_setup.py` eksportuje dodatkowe dowody przez dotychczasowe CLI. `geometry_setup_preview.py` pokazuje cztery panele w układzie 2×2 z osobnym widokiem topologii i sąsiedztwa.
- Istniejące modele domenowe, statusy szkiców i workflow konfiguracji są używane przez normalne punkty wejścia `detect`, `analyze`, `propose` oraz `verified`.

`unresolved` liczy hipotezy bez zaakceptowanego wielokąta. Pole `rows` zawiera także słabe lub zbyt krótkie sąsiedztwa, które wcześniej znikały przed składaniem. Liczba tych rekordów nie jest liczbą pominiętych fizycznych stanowisk. Identyfikatory LSD i osi są lokalne dla uruchomienia. Fizyczne ID pochodzą z zatwierdzonych konfiguracji.

## Wyniki na rzeczywistych obrazach

O/E/U oznacza `observed` / `estimated` / `unresolved`. LSD pozostaje identyczne dla obu wersji.

| Obraz | LSD | Osie Phase 2 → 3 | Węzły Phase 2 → 3 | O/E/U Phase 2 → 3 | Ocena |
| --- | ---: | ---: | ---: | --- | --- |
| [Pilot](</home/karoljaron/Dokumenty/ChatGPT/System parkingowy/geometry-initialization-v4-3-phase3-2026-11-14/comparisons/cctv-pilot.jpg>) | 1866 | 69 → 70 | 106 → 107 | 0/7/19 → 0/7/19 | Siedem lokalizacji odpowiada osobnym polom nawierzchni, zero oczywistych fałszywych miejsc w przeglądzie. Trzy szkice mają oszacowane domknięcie. |
| [UFPR04](</home/karoljaron/Dokumenty/ChatGPT/System parkingowy/geometry-initialization-v4-3-phase3-2026-11-14/comparisons/ufpr04-empty.jpg>) | 2215 | 40 → 39 | 24 → 23 | 0/0/1 → 0/0/1 | Zero propozycji. Brak wystarczającego wspólnego rzędu. |
| [UFPR05](</home/karoljaron/Dokumenty/ChatGPT/System parkingowy/geometry-initialization-v4-3-phase3-2026-11-14/comparisons/ufpr05-empty.jpg>) | 1403 | 46 → 52 | 28 → 33 | 0/0/1 → 0/0/1 | Dodatkowe linie nie tworzą potwierdzonych fizycznych granic. |
| [Family Dollar](</home/karoljaron/Dokumenty/ChatGPT/System parkingowy/geometry-initialization-v4-3-phase3-2026-11-14/comparisons/family-dollar.jpg>) | 6506 | 26 → 25 | 14 → 14 | 0/0/1 → 0/0/1 | Zatwierdzone C pozostaje pominięte. Zero propozycji na T-01. |
| [ClearPix](</home/karoljaron/Dokumenty/ChatGPT/System parkingowy/geometry-initialization-v4-3-phase3-2026-11-14/comparisons/clearpix.jpg>) | 4846 | 23 → 22 | 4 → 4 | 0/0/1 → 0/0/1 | Pięć zatwierdzonych miejsc pozostaje pominiętych. |
| [Ion Alarm](</home/karoljaron/Dokumenty/ChatGPT/System parkingowy/geometry-initialization-v4-3-phase3-2026-11-14/comparisons/ion-alarm.jpg>) | 4019 | 77 → 93 | 34 → 36 | 0/0/1 → 0/0/1 | Zero propozycji na polach parkingowych i dolnym kreskowaniu. |
| [CNR kamera 8](</home/karoljaron/Dokumenty/ChatGPT/System parkingowy/geometry-initialization-v4-3-phase3-2026-11-14/comparisons/cnr-camera8.jpg>) | 501 | 11 → 18 | 4 → 4 | 0/0/1 → 0/0/1 | Słabe i krótkie oznakowanie, brak wspartych sąsiadów. |
| [CNR kamera 3 mokra](</home/karoljaron/Dokumenty/ChatGPT/System parkingowy/geometry-initialization-v4-3-phase3-2026-11-14/comparisons/cnr-camera3-wet.jpg>) | 722 | 14 → 22 | 8 → 13 | 0/0/1 → 0/0/1 | Dodatkowe osie, nadal zero ROI. |

W obu wersjach żadna z propozycji w tych próbach nie wychodzi poza raster. Nie występują pary ROI z nakładaniem ponad 10%. Na pilocie nakładanie wynosi 0%. Dla kadrów bez propozycji zero fałszywych ROI współistnieje z małym pokryciem. Globalnego recall ani FP/FN całych rzeczywistych parkingów nie obliczano z powodu niepełnej geometrii odniesienia.

### Kontrola A/B/C

Dopasowanie maksymalizuje sumę IoU przy ograniczeniu 1:1 i możliwości pozostawienia obiektu bez pary. Próg zaliczenia wynosi 0,50. Wszystkie trzy dopasowania przekraczają 0,95. Błąd granicy obliczono symetrycznie przez próbki po obu obwodach w pikselach oryginalnego obrazu.

| Miejsce | IoU Phase 2 | IoU Phase 3 | Średni błąd granicy Phase 3 | Maksymalny błąd próbki |
| --- | ---: | ---: | ---: | ---: |
| A | 0,95998 | 0,95998 | 1,366 px | 3,055 px |
| B | 0,96111 | 0,96111 | 1,773 px | 5,640 px |
| C | 0,96424 | 0,96424 | 1,317 px | 7,173 px |

Żadna nowa propozycja rzeczywista nie otrzymała zatwierdzenia operatora w tym zadaniu.

## Kontrolowane przypadki syntetyczne

Każdy wariant ma jawnie zapisaną geometrię źródłową. Są to warianty kilku prostych układów stworzone do diagnostyki i regresji. Nie są niezależnymi widokami CCTV.

| Przypadek | Poprawne / FP / pominięte Phase 2 | Poprawne / FP / pominięte Phase 3 | O/E/U Phase 3 | IoU poprawnych Phase 3 |
| --- | --- | --- | --- | --- |
| [Nierówne szerokości](</home/karoljaron/Dokumenty/ChatGPT/System parkingowy/geometry-initialization-v4-3-phase3-2026-11-14/comparisons/synthetic-uneven.jpg>) | 3/0/0 | 3/0/0 | 3/0/0 | 0,996–0,997 |
| [Słaby kontrast 23/255](</home/karoljaron/Dokumenty/ChatGPT/System parkingowy/geometry-initialization-v4-3-phase3-2026-11-14/comparisons/synthetic-low-contrast.jpg>) | 0/0/3 | 3/0/0 | 2/1/0 | 0,958–0,997 |
| [Cień](</home/karoljaron/Dokumenty/ChatGPT/System parkingowy/geometry-initialization-v4-3-phase3-2026-11-14/comparisons/synthetic-shadow.jpg>) | 0/0/3 | 2/0/1 | 0/2/0 | 0,960–0,964 |
| [Szum bez oznakowania](</home/karoljaron/Dokumenty/ChatGPT/System parkingowy/geometry-initialization-v4-3-phase3-2026-11-14/comparisons/synthetic-noisy-negative.jpg>) | 0/0/0 | 0/0/0 | 0/0/1 | — |
| [Kreskowanie w większym obrysie](</home/karoljaron/Dokumenty/ChatGPT/System parkingowy/geometry-initialization-v4-3-phase3-2026-11-14/comparisons/synthetic-hatched.jpg>) | 2/2/0 | 2/0/0 | 2/0/3 | 0,997 |
| [Krótka luka](</home/karoljaron/Dokumenty/ChatGPT/System parkingowy/geometry-initialization-v4-3-phase3-2026-11-14/comparisons/synthetic-short-gap.jpg>) | 3/0/0 | 3/0/0 | 1/2/0 | 0,996–0,997 |
| [Długa luka](</home/karoljaron/Dokumenty/ChatGPT/System parkingowy/geometry-initialization-v4-3-phase3-2026-11-14/comparisons/synthetic-long-gap.jpg>) | 1/0/2 | 1/0/2 | 1/0/3 | 0,997 |
| [Obrócony rząd](</home/karoljaron/Dokumenty/ChatGPT/System parkingowy/geometry-initialization-v4-3-phase3-2026-11-14/comparisons/synthetic-rotated.jpg>) | 3/0/0 | 3/0/0 | 3/0/0 | 0,996–0,997 |

W wariancie cienia nadal brakuje jednego stanowiska. Linie wspólnej granicy rozdzielają się przy zmianie jasności, a w pełni wsparte sąsiedztwo nie obejmuje całego rzędu. Przerwa 45 px nadal pozostawia dwa brakujące ROI. Wartość U liczy hipotezy, co wyjaśnia jej różnicę względem liczby pominięć.

## Weryfikacja i narzędzie operatora

| Sprawdzenie | Wynik |
| --- | --- |
| [Pytest silnika](</home/karoljaron/Dokumenty/ChatGPT/System parkingowy/geometry-initialization-v4-3-phase3-2026-11-14/pytest-engine.log>) | 969 zaliczonych. Obejmuje A/B/C i nowe przypadki słabej farby, cienia, obrotu, lokalnego szumu, kreskowania, nierównego budżetu wspólnego końca i zagnieżdżonego konfliktu. |
| [Pytest research](</home/karoljaron/Dokumenty/ChatGPT/System parkingowy/geometry-initialization-v4-3-phase3-2026-11-14/pytest-research.log>) | 61 zaliczonych, 3 pominięte przez istniejące warunki dostępu do historycznych artefaktów. |
| [Black](</home/karoljaron/Dokumenty/ChatGPT/System parkingowy/geometry-initialization-v4-3-phase3-2026-11-14/black.log>) | Wszystkie 10 zmienionych plików Python zgodnych. |
| [Flake8](</home/karoljaron/Dokumenty/ChatGPT/System parkingowy/geometry-initialization-v4-3-phase3-2026-11-14/flake8.log>) | Całe `src`, `tests`, `scripts` bez zgłoszeń. |
| `git diff --check` | Bez zgłoszeń. |
| [LabelMe → approve → ParkingConfiguration](</home/karoljaron/Dokumenty/ChatGPT/System parkingowy/geometry-initialization-v4-3-phase3-2026-11-14/operator-workflow.log>) | LabelMe 5.9.1 otworzył trzy syntetyczne szkice, przesunął wierzchołek o 1,125 px i zapisał dwa. Stary SHA-256 został odrzucony. Testowe zatwierdzenie i odczyt zachowały oba ID i współrzędne z tolerancją 10⁻⁹ px. |
| Odczyt istniejącej konfiguracji CCTV | Odtworzył trzy wcześniejsze projekcje A/B/C bez nowych zatwierdzeń. |

## Odtworzenie i dalsze ograniczenia

Manifesty artefaktów wskazują dokładne obrazy i role. `run_diagnostics.py` wywołuje normalny workflow `propose`, a `evaluate_and_render.py` wykonuje dopasowanie i rysuje porównania w jednej skali. `evaluation.json` zawiera pomiary wszystkich przypadków. `baseline-code` przechowuje kod Phase 2, a `phase3-code-frozen.json` utrwala stan przed sprawdzeniem CNR. Późniejsza korekta długości nagłówka podglądu dotyczy wyłącznie prezentacji.

~~~bash
cd /home/karoljaron/Projekty/ParkVision
OUT='/home/karoljaron/Dokumenty/ChatGPT/System parkingowy/geometry-initialization-v4-3-phase3-2026-11-14'
PYTHONPATH=apps/engine/src apps/engine/.venv/bin/python \
  "$OUT/run_diagnostics.py" "$OUT/cases.json" "$OUT/repeated-review"
~~~

Do niezależnej oceny na nowym parkingu brakuje obrazu referencyjnego ze znanej stacjonarnej kamery, udokumentowanego pochodzenia oraz kompletu fizycznych ROI zweryfikowanych przed predykcją. Potrzebne są jawne strefy nieparkingowe, widoczność granic i konwencja obrysu przy krawężnikach. Zestaw powinien obejmować poprawne miejsca, trudne końce rzędów i przypadki, w których operator świadomie pozostawia granicę nierozstrzygniętą.

Największe ograniczenie pokrycia rzeczywistych parkingów stanowi zależność składania od wspólnej poprzecznej linii z dostateczną liczbą separatorów. Słabsze oznakowanie, schodkowe wyspy, silna dystorsja i obrysy zasłonięte pojazdami pozostają otwarte. Zwiększanie liczby osi poprawia lokalne dowody, ale próby nie wykazały poprawy pełnych ROI na tych kadrach.

Zmiany są w bieżącym lokalnym repozytorium jako pliki robocze. Nie wykonano commita, pushu ani PR. Kod modeli pojazdów i zajętości, anotacje historyczne, chronione manifesty i splity nie należą do zmienionych plików.
