# Kandydaci przecięć i zakończeń — MVP

Etap rozszerza diagnostykę po parowaniu z commita `27909f0`. Przed zmianami
drzewo robocze było czyste, a pełny pytest przechodził: 637 testów. Nie zmienia
LSD, `segment_merging.py`, dopasowania, parowania, domenowego `Point`/`Polygon`
ani `OpenCVSlotDetector.detect()` (nadal `NotImplementedError`).

## Modele i API

`detection/marking_topology.py` udostępnia:

```python
find_topology_candidates(
    pairs: tuple[MarkingPair, ...],
    unpaired_lines: tuple[LineSegment, ...] = (),
    *,
    endpoint_tolerance_px: float = 3.0,
    max_extension_px: float = 0.0,
    min_crossing_angle: float = 15.0,
    dedup_radius_px: float = 3.0,
    min_context_length_px: float = 30.0,
) -> TopologyCandidates
```

Współrzędne pozostają subpikselowymi pikselami obrazu. Wykorzystujemy istniejące
osie `MarkingPair.centerline`, `LineSegment`, typ współrzędnych `PixelPoint`
i `angle_difference()`; nie tworzymy kolejnego typu linii ani nie przenosimy
predykatów domeny do przestrzeni obrazu. Wyznaczenie unikalnego przecięcia
parametrycznych prostych jest nową operacją; nie kopiuje walidacji wielokątów.

Publiczne, niemutowalne dataclassy są w `detection/topology_candidate.py`:

- `LineIntersection`: dokładny punkt przecięcia, dwa ID, role, L/T/X,
  przedłużenie każdej linii i informacja o kontekście niesparowanej krawędzi;
- `JunctionCandidate`: pobliskie trafienia z własnymi etykietami i wspólną kotwicą;
- `EndpointCandidate`: oryginalny koniec osi/krawędzi, ID linii i odnośniki do węzłów;
- `TopologyCandidates`: wszystkie wejściowe osie i niesparowane linie,
  niesparowane ID, węzły i końce.

Pomocnicza dataclassa `TopologyGeometry` jest w osobnym
`detection/topology_geometry.py`. Przechowuje linię, kierunek jednostkowy,
długość i bbox wyszukiwania. `marking_topology.py` zawiera wyłącznie algorytm
i funkcje pomocnicze, bez definicji modeli danych.

Główna funkcja prowadzi przez walidację, przygotowanie geometrii, wyszukiwanie
przecięć, deduplikację i powiązanie końców. Każdy z tych kroków ma własną
funkcję; indeks przestrzenny nie miesza się z dokładną geometrią przecięcia.
W skrypcie batchowym osobne funkcje obsługują opcje, wyszukiwanie obrazów,
diagnostykę LSD, wykonanie etapów geometrii i zapis wyników. Budowa danych
topologii jest oddzielona od ich renderowania i zapisu na dysk.

Wynik zachowuje obiekty linii i ich ID; nie przepisuje źródeł. Kolejność linii
to rosnące ID, końców: ID i start/end, węzłów: współrzędne kotwicy. ID węzłów
są lokalne dla wywołania, nie opisują fizycznych miejsc ani śledzenia w czasie.
Fragmenty i `SegmentGroup` pozostają u wywołującego, dostępne przez diagnostykę.

## Dokładna semantyka geometryczna

1. Rozpatrujemy jedynie pary o różnicy orientacji co najmniej
   `min_crossing_angle`. Równoległe i współliniowe odcinki nie mają unikalnego
   węzła L/T/X. Dodatkowy próg wyznacznika `1e-12` zabezpiecza niestabilne
   przecięcie, gdy wywołujący ustawi kąt niemal równy zeru.
2. Liczymy punkt przecięcia prostych i odległość wzdłuż każdego odcinka.
   Punkt musi leżeć w skończonym zakresie lub wymagać przedłużenia nie większego
   niż `max_extension_px` **osobno dla każdej linii**. Domyślne zero zabrania
   ekstrapolacji. `endpoint_tolerance_px` nie powiększa zakresów!
3. Ramieniem jest część reprezentanta od punktu do jego końca dłuższa od
   `endpoint_tolerance_px`. Dwa ramiona oznaczają rolę `interior`, jedno — `end`.
   Jeśli któraś linia nie ma rozróżnialnego ramienia, trafienie pomijamy,
   zachowując samą linię i jej końce. Stąd **L = end/end, T = end/interior,
   X = interior/interior**. L nie wymaga kąta prostego. Tolerancja jest domknięta
   przy końcu: ramię równe tolerancji nie jest liczone. Odwrócenie końców nie
   zmienia geometrii klasyfikacji.
4. `extension_px` odpowiada kolejności `line_ids`. Dodatnia wartość oznacza
   `hypothetical=True`, również gdy mieści się w tolerancji klasyfikacji końców.
   Jedynie błąd arytmetyczny do `1e-9` px traktujemy jako zero. To jawny bufor
   numeryczny, nie próg pokrycia farbą. Zero oznacza przecięcie **zakresów
   reprezentantów**, nie potwierdzenie obserwacji farby w punkcie.
5. Deduplikacja używa odległości euklidesowej od stałej kotwicy. Najpierw bierze
   trafienia w zakresach, następnie wymagające ekstrapolacji; w ramach tych klas
   osie przed kontekstem, potem punkt i ID. Trafienie trafia do najbliższej
   istniejącej kotwicy w promieniu, remisy rozstrzyga kolejność utworzenia.
   Kotwica nie przesuwa się i nie powstaje łańcuch scalania 0→2→4 px. Wszystkie
   dokładne punkty, przedłużenia i etykiety pozostają w `intersections`.

Kilka etykiet w węźle nie jest głosowaniem ani klasyfikacją sumy ramion.
Trzy linie mogą dać kilka różnych hipotez parowych. Nie nazywamy ich
potwierdzonymi rogami. Reprezentanci i osie mogą obejmować przerwy pomiędzy
oryginalnymi fragmentami LSD; nawet `hypothetical=False` nie potwierdza farby.

## Niesparowane krawędzie i zakończenia

Nie traktujemy wszystkich krawędzi jako namalowanych pasów. Porównujemy osie
między sobą, a krawędź niesparowaną wyłącznie z osią i wyłącznie przy końcu osi.
Taki kontekst może wskazać dojście do krawężnika, ale pozostaje słabszą hipotezą
(`uses_unpaired_context=True`). Nigdy nie porównujemy kontekstu z kontekstem,
nie generujemy X z przypadkowego przecięcia krawędzi przez wnętrze osi.
Kontekst krótszy od 30 px domyślnie nie uczestniczy w wyszukiwaniu trafień.

**Brak przecięcia niczego nie usuwa.** Wszystkie osie, niesparowane linie,
krótkie fragmenty oraz oba końce są w wyniku, także poza kadrem. Koniec
reprezentanta może wynikać z zasłonięcia lub granicy obrazu, nie musi być
fizycznym końcem farby. Zakończenia z rolą `end` odsyłają do pobliskich węzłów;
przecięcie we wnętrzu linii nie zużywa jej końców. Przy niejednoznacznie krótkim
ramieniu odnośnik trafia do bliższego końca (dokładny remis: start).

## Diagnostyka i koszt

W istniejącym `scripts/run_parking_batch.py` etap jest opcjonalny:

```bash
.venv/bin/python scripts/run_parking_batch.py \
  --samples samples/01-pklot-ufpr05-empty.jpg \
  --output /tmp/parkvision-topology --topology
```

Przełączniki odpowiadają argumentom API: `--topology-endpoint-tolerance`,
`--topology-max-extension`, `--topology-min-crossing-angle`,
`--topology-dedup-radius`, `--topology-min-context-length`. Wszystkie odległości
są w pikselach, kąt w stopniach. Tolerancje muszą być skończone i nieujemne,
a kąt w `(0, 90]`. Geometria musi być skończona, niezerowa, ID osi, obu krawędzi
par i niesparowanych linii muszą być rozłączne. Niepoprawne opcje skrypt odrzuca
przed odczytem obrazów i zapisem wyników.

Dodajemy wyłącznie `topology.json` i `preview_topology_candidates.png`.
Dotychczasowe siedem plików nie zmienia formatu ani zawartości. JSON przechowuje
parametry, wszystkie linie i końce oraz węzły z dokładnymi trafieniami. Powiązanie:
ID linii → `pair_centerline_id`/`group_indices` → `markings.json` →
`source_segment_ids` → `segments.json`. Nie duplikujemy źródłowego pokrycia.

Na podglądzie osie są zielone, kontekst szary, końce osi turkusowe. Końce
kontekstu pozostają w JSON, żeby tysiące znaczników nie zasłaniały obrazu.
Węzły z trafieniem w zakresach osi są żółte, sam kontekst fioletowy, wyłącznie
ekstrapolowane pomarańczowe. Mieszany węzeł dostaje kolor silniejszego trafienia;
dokładny status każdej hipotezy jest w JSON. Poza kadrem zachowujemy dane,
ale nie rysujemy znaczników węzłów. `?` przypomina, że etykieta jest kandydatem.

Indeks przestrzenny bbox odrzuca odległe linie przed sprawdzeniem kierunku
i dokładną geometrią. Przeszukują go wyłącznie osie; długość kontekstu ogranicza
liczbę uczestników, nie liczbę zapisanych linii. Komórka ma co najmniej 64 px
i rośnie z najdłuższą linią/limitem przedłużenia, więc długie bbox nie zajmują
nieograniczonej liczby komórek. Deduplikacja ma osobny indeks punktów. W gęstym
układzie wzajemnie nachodzących linii koszt i liczba hipotez nadal mogą być
kwadratowe. Nie ma ukrytego limitu ucinającego wyniki.

## Weryfikacja i ograniczenia

Testy obejmują L/T/X po obrocie i odwróceniu końców, skończone zakresy,
niezależność tolerancji końca od ekstrapolacji, granice przedłużenia obu linii,
równoległość, brak ramion, deduplikację bez łańcuchów, kolejność ID/metryk,
przerwy LSD, zachowanie obiektów i niesparowanego kontekstu, walidację oraz
2000 odległych osi bez wejścia w dokładną geometrię. Test skryptu porównuje
poprzednie pliki bajtowo, przechodzi powiązania JSON i obejmuje rzeczywiste LSD
na obrazie pustym oraz białym krzyżu (kandydat X).

Końcowy pełny `pytest -q`: **696 passed**. Black (`--check`, każdy plik osobno)
i Flake8 (`--jobs 1`) przechodzą dla wszystkich sześciu dodanych/zmienionych
plików Python. Nie utworzono commita.

Po uporządkowaniu kodu ponownie przeszło 696 testów oraz Mypy dla trzech modułów
topologii i skryptu. Na zdjęciu PKLot wszystkie dziewięć plików diagnostycznych,
łącznie z JSON i podglądem topologii, jest identycznych bajtowo przed i po
refaktoringu.

Na `01-pklot-ufpr05-empty.jpg`: 44 osie i 1216 niesparowanych reprezentantów
dają 1260 zachowanych linii i 2520 końców. Bez ekstrapolacji: 0 węzłów;
z limitem 6 px: 6 węzłów (4 L, 2 T), wszystkie ekstrapolowane, cztery z kontekstem.
Etap geometryczny trwał około 0,006–0,010 s. Wyniki są w workspace dokumentów
w `topology-candidates` i `topology-candidates-no-extension`. Siedem poprzednich
plików porównano z `marking-axes-default/image-2`: są identyczne bajtowo.
Na podglądzie trafienia przypadają przy połączeniach linii i krawędziach wysp;
bez etykiet nie oceniamy ich precyzji. Brak trafień bez ekstrapolacji nie oznacza
braku oznaczeń — wiele osi nie dochodzi do fizycznego połączenia.

Następny krok to ocena pokrycia źródłowych fragmentów przy punkcie i ramionach,
osobno dla obu krawędzi pary, oraz ocena farby na obrazie/ROI. Potem można
rozważyć agregację ramion wielu linii. Nie implementujemy jeszcze rekonstrukcji
`Polygon`, `SlotProposal`, fizycznych miejsc ani kompletnego detektora.
