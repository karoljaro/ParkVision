# Niezależny review topologii i mały etap wsparcia LSD

Weryfikacja: 2026-10-29. Początkowy HEAD:
`74cc45985e8cfd2e5f39aac8d32f311dfef4c143`, czyste drzewo robocze.
Sprawdzono faktyczny kod, dostarczony `marking-topology.md`, stare JSON/PNG,
odtworzono przetwarzanie zdjęć oraz uruchomiono testy. Nie wykonano commita.

## Wniosek review

Nie znaleziono blokującego błędu implementacji dla obecnego etapu **hipotez
geometrycznych**. Jakość nie wystarcza do potwierdzania węzłów namalowanych
oznaczeń ani miejsc parkingowych. Dodanie pomiarów źródłowego wsparcia jest
uzasadnione jako diagnostyka, bez selekcji kandydatów.

- Wzory przecięcia i znaki parametrów są poprawne. Niezależny solver Decimal
  (60 cyfr precyzji) sprawdził wszystkie 7140 par 120 deterministycznie
  wygenerowanych osi, z pominięciem indeksu przestrzennego. Zestawy 470
  dopuszczonych przecięć i etykiety były zgodne; największa różnica współrzędnej
  wyniosła `4.98e-14` px. To test kontrolny, nie dowód dla wszystkich liczb float.
- Domyślny kąt 15° ogranicza niestabilność bliskich równoległości. Dodatkowy
  wyznacznik `1e-12` i bufor zakresów `1e-9` służą arytmetyce; dla ekstremalnych
  współrzędnych i bardzo małych kątów nie gwarantujemy dokładności. Progi
  klasyfikacji są ostre, więc wartości na ich granicach są wrażliwe na float.
- Limit przedłużenia jest sprawdzany **osobno dla obu linii**, wzdłuż odcinka.
  Tolerancja końca go nie powiększa. L/T/X zależy od liczby ramion dłuższych od
  tolerancji: to etykieta pary, bez wymogu kąta prostego dla L i bez agregacji
  ramion wielu linii. Zerowe przedłużenie nie oznacza ciągłości źródeł LSD.
- Deduplikacja ma stałą kotwicę i zachowuje wszystkie dokładne trafienia;
  nie łączy przez łańcuch sąsiadów. Promień 3 px może pozostawić dwa węzły
  przy jednym fizycznym narożniku: na UFPR04 są trafienia przy `(744,570)`
  i `(749,576)`. To ograniczenie skali/semantyki, nie utrata danych.
- Wszystkie linie i oba końce pozostają w wyniku, także krótkie, izolowane
  i poza kadrem. ID są lokalne; kolejność ID, ról, długości przedłużeń oraz
  powiązania axis → fitted edge → group → raw LSD sprawdzono w JSON i testach.
  Odwrócenie kolejności wejść nie zmieniło wyników na trzech zdjęciach.
- Kontekst niesparowany jest celowo ograniczony do końca osi i minimum 30 px.
  To zmniejsza przypadkowe trafienia, ale pomija także wiarygodne dojścia do
  krótkich fragmentów krawężnika i sytuacje, w których koniec reprezentanta
  nie odpowiada fizycznemu końcowi oznaczenia. Same fragmenty nie są usuwane.

## Porównanie zdjęć

Obrazy z `samples/`, w oryginalnej rozdzielczości: UFPR05/UFPR04 1280×720,
CNR wet 1000×750. Pozostałe parametry API i wcześniejszych etapów są domyślne.
W komórkach podano **węzły / dokładne hipotezy parowe**, nie potwierdzone węzły.

| Zdjęcie | Osie / niesparowane linie | 0 px | 3 px | 6 px | 12 px |
| --- | ---: | ---: | ---: | ---: | ---: |
| `01-pklot-ufpr05-empty.jpg` | 44 / 1216 | 0/0 | 0/0 | 6/6 | 13/13 |
| `02-pklot-ufpr04-empty.jpg` | 38 / 2016 | 0/0 | 0/0 | 3/3 | 9/10 |
| `04-cnrext-camera3-wet.jpg` | 17 / 608 | 0/0 | 1/1 | 1/2 | 3/5 |

Przy 6 px: UFPR05 ma 4 L i 2 T (4 hipotezy z kontekstem), UFPR04 3 L
(wszystkie z kontekstem), CNR 2 L we wspólnym węźle (obie z kontekstem).
Wszystkie te trafienia wymagają przedłużenia. Przy 12 px etykiety to odpowiednio
10 L + 3 T, 7 L + 3 T oraz 4 L + 1 T; nadal wszystkie są ekstrapolowane.
Brak X na tych zdjęciach nie świadczy o błędzie klasyfikatora: istniejący test
rzeczywistego LSD na syntetycznym białym krzyżu generuje X.

Ogląd oryginałów, pełnych nakładek 6/12 px i powiększeń każdego węzła przy 6 px:

- UFPR05: wiele trafień pasuje wizualnie do dojścia linii do wyspy/krawężnika.
  Osie obu krawędzi krawężnika też bywają traktowane jako pary; L osi/osi nie
  jest automatycznie rogiem farby. Widać również osie słupów.
- UFPR04: trzy trafienia 6 px leżą przy krawędziach wysp. Dwa bliskie węzły
  przy dolnej prawej wyspie opisują różne krawędzie tej samej okolicy. Korony
  drzew i fragmentacja źródeł ograniczają kompletność. Większy limit nie daje
  podstaw do twierdzenia, że rośnie precyzja.
- CNR wet: jeden węzeł 6 px leży przy narożniku wyspy. Dwie hipotezy korzystają
  z różnych fragmentów krawężnika. Wytarta farba, mokra nawierzchnia i drzewa
  ograniczają dane, a lampy i elementy tła też tworzą osie.

Sprawdzono również wrażliwość przy limicie 6 px:

| Zmiana pojedynczego parametru | UFPR05 węzły/trafienia | UFPR04 | CNR wet |
| --- | ---: | ---: | ---: |
| minimum kontekstu 30 → 0 px | 26/28 | 23/24 | 4/5 |
| tolerancja końca 3 → 6 px | 6/6 | 3/3 | 2/3 |
| kąt minimalny 15 → 5° | 6/6 | 3/3 | 1/2 |

Na podglądzie krótkiego kontekstu UFPR05 pojawiają się dalsze wiarygodne dojścia
do krawężnika, np. okolica `(380,176)`. Natomiast dodatkowy T przy tolerancji
końca 6 px na CNR `(767,216)` pochodzi od lampy przy wyspie, nie linii miejsca.
Zatem nie należy globalnie obniżać progów na podstawie samej liczby trafień.
Wcześniejsze dopasowanie/parowanie dostarcza geometryczne osie bez semantyki
farby; to źródło części fałszywych hipotez, nie błąd obliczania przecięcia.
Bez ręcznych etykiet nie podajemy precision/recall ani procentu poprawnych L/T/X.

## Dodany pionowy etap

`measure_topology_support(topology, pairs, source_groups, *,
junction_radius_px=3, arm_length_px=15)` otrzymuje istniejące modele oraz mapę
ID dopasowanej **krawędzi** → oryginalny `SegmentGroup`.
Modele wsparcia są w `topology_support.py`, algorytm w
`topology_support_measurement.py`. Istniejący konstruktor `TopologyGeometry`
został udostępniony jako `prepare_topology_geometry()`, bez zmiany matematyki.

Każda dokładna hipoteza ma osobny rekord identyfikowany przez `candidate_id`
i `intersection_index`. Dla obu linii oraz osobno dla każdej krawędzi pary:

- rzutujemy oryginalne fragmenty na kierunek osi/kontekstu;
- mierzymy sumę przedziałów bez podwójnego liczenia nakładek i bez wypełniania
  przerw: okno przecięcia `[-3,3]`, ramiona `[-15,-3]` i `[3,15]` px;
- zapisujemy długość, udział pokrycia i ID źródeł o dodatnim pokryciu okna,
  a także odległość wzdłuż linii do najbliższego źródłowego przedziału;
- nie przycinamy do wspólnego zakresu osi. Dłuższa źródłowa krawędź może
  wspierać hipotezę poza tym zakresem. Oba podpisane ramiona są mierzone także
  dla roli `end`; ich kierunek wynika ze start→end zapisanej linii.

To **pokrycie wzdłuż linii**, bez pomiaru farby, kontrastu i poprzecznej
odległości od punktu. Wysokie pokrycie krawężnika/słupa nie potwierdza farby.
Nie dodano filtra, punktacji ani zmiany klasyfikacji i nie usunięto żadnej linii.
Nie ma wystarczającego uzasadnienia dla dodatkowego progu obrazu w tym etapie.

Opcjonalna diagnostyka dopisuje wyłącznie `topology_support.json` oraz
`preview_topology_support.png`. Podgląd pokazuje oryginalne fragmenty użyte
w lokalnych oknach (całe, nieprzedłużone odcinki), dokładne punkty i procenty
pokrycia okna przecięcia osobno dla krawędzi; ramiona pozostają w JSON.

```bash
.venv/bin/python scripts/run_parking_batch.py \
  --samples samples/01-pklot-ufpr05-empty.jpg \
  --output /tmp/parkvision-topology-support \
  --topology --topology-max-extension 6 --topology-support
```

Okna można ustawić przez `--support-junction-radius` i `--support-arm-length`.
Wymagane są skończone `0 < radius < length` oraz pełna, jednoznaczna proweniencja.

Przy domyślnych oknach i limicie 6 px pomiar na rzeczywistych zdjęciach dał:

| Zdjęcie | Pomiary krawędzi w hipotezach | Okno przecięcia: zero / częściowe / pełne |
| --- | ---: | ---: |
| UFPR05 | 20 | 10 / 7 / 3 |
| UFPR04 | 9 | 3 / 5 / 1 |
| CNR wet | 6 | 1 / 5 / 0 |

Każda hipoteza UFPR05/UFPR04 ma co najmniej jedną krawędź z zerowym pokryciem
okna przecięcia. To mierzalna luka LSD, nie automatyczna negatywna ocena węzła.
Liczby dotyczą krawędzi w hipotezach; powtarzająca się krawędź jest mierzona
ponownie przy innym dokładnym punkcie.

## Zmienione/dodane pliki i weryfikacja

- `src/parkvision/detection/topology_support.py` — osobne modele wsparcia.
- `src/parkvision/detection/topology_support_measurement.py` — pomiar źródeł.
- `src/parkvision/detection/marking_topology.py` — współdzielony konstruktor geometrii.
- `scripts/run_parking_batch.py` — opcjonalne dwa pliki i walidacja opcji.
- `tests/detection/test_topology_support.py` — przerwy, sumy przedziałów,
  osobne krawędzie, kontekst, przedłużenia, obroty, kierunek, dokładne punkty,
  zachowanie danych i błędy proweniencji.
- `tests/scripts/test_run_parking_batch.py` — zgodność dziewięciu plików,
  powiązania JSON, źródłowe przerwy, walidacja CLI i rzeczywiste LSD.
- Ten raport.

Początkowo: **696 passed**. Po zmianach: **723 passed**. Flake8 (`--jobs 1`)
przechodzi dla całego `src tests scripts`. Black przechodzi dla wszystkich
zmienionych/dodanych plików; pełna kontrola daje 40/41. Jedyną przeszkodą jest
istniejące formatowanie `src/parkvision/detection/segment_group.py`, potwierdzone
również na treści z początkowego HEAD. Ten plik pozostawiono bez zmian.

Wyniki UFPR05 0/6 px są identyczne bajtowo ze starymi zapisami workspace.
Zmiana limitu przedłużenia zachowała siedem wcześniejszych plików na wszystkich
trzech zdjęciach. Włączenie wsparcia zachowało **wszystkie dziewięć** plików
topologii i wcześniejszych etapów na trzech rzeczywistych zdjęciach, również
po współdzieleniu konstruktora geometrii.

Pełne diagnostyki i metryki są poza repozytorium, w workspace dokumentów:
`topology-review-2026-10-29/`. Zawiera `review-metrics.json`, `support-metrics.json`,
warianty 0/3/6/12 px, próby wrażliwości, podglądy źródeł oraz powiększenia węzłów.
LSD, grupowanie, dopasowanie, parowanie i `OpenCVSlotDetector.detect()` pozostają
bez zmian. Nie implementowano rekonstrukcji `Polygon`, `SlotProposal` ani detektora.
