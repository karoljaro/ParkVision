# Parowanie krawędzi namalowanego pasa — MVP

## Ocena danych i zakres

Punktem wyjścia jest commit `2919cdf` z `fit_group_line()`. Przed implementacją
sprawdzono modele, grupowanie, testy, zdjęcia źródłowe i dotychczasowy podgląd
reprezentantów. Zdjęcie lotnicze ma rozdzielczość 4750×3164, PKLot UFPR05
1280×720, a `parking.jpg` 1440×960. Na wyraźnych, długich oznaczeniach widać
dwie bliskie krawędzie, więc geometryczne parowanie jest wykonalne. Na zdjęciach
ukośnych farba zwęża się z odległością; występują też krawężniki, cienie, numery,
ubytki i zasłonięcia. Nie ma etykiet par krawędzi, więc podglądy pozwalają ocenić
przydatność hipotez, ale nie wyznaczają precyzji ani kompletności detekcji.

MVP zwraca **hipotezy geometryczne**, a nie potwierdzenie obecności farby.
Nie dodaje przecięć L/T/X, rekonstrukcji miejsc, `Polygon` ani `SlotProposal`.
`OpenCVSlotDetector.detect()` nadal zgłasza `NotImplementedError`.

## Przepływ i powiązanie ze źródłami

`Frame → extract_segments → group_segments → fit_group_line → pair_marking_edges`

Nie zmieniono grupowania ani istniejących funkcji geometrycznych.
`pair_marking_edges()` przyjmuje krotkę istniejących `LineSegment` reprezentujących
grupy. Wynik `MarkingPair` przechowuje referencje do dwóch reprezentantów,
oś jako kolejny `LineSegment`, szerokości przy obu końcach osi i udział nakładania.
Grupy i oryginalne odcinki LSD pozostają osobnymi danymi. Nie powstaje kolejna
klasa reprezentanta grupy ani nowy model punktu.

W skrypcie kolejność reprezentantów odpowiada kolejności grup. `markings.json`
zapisuje dla każdej grupy jej indeks, identyfikatory oryginalnych odcinków
oraz reprezentanta. Każda para odsyła do dwóch indeksów grup i ID reprezentantów.
Oryginalne współrzędne fragmentów pozostają w `segments.json`. W ten sposób można
przejść od osi przez obie grupy do wszystkich fragmentów LSD i ich przerw.
`unpaired_edge_ids` obejmuje wszystkich pozostałych reprezentantów, również
krótkich. Brak partnera jest prawidłowym wynikiem, np. dla pojedynczej krawędzi.

ID osi są większe od wszystkich wejściowych ID reprezentantów. Przydzielane są
kandydatom przed rozstrzygnięciem konfliktów, dlatego wynikowe ID mogą mieć luki.
Są lokalne dla uruchomienia; nie identyfikują fizycznych oznaczeń między klatkami.

## Geometria i konflikty

1. Sprawdzamy najmniejszą różnicę kątów, korzystając z istniejącej
   `angle_difference()`. Kierunek od początku do końca odcinka nie ma znaczenia:
   odwrócone końce nie dyskwalifikują pary. Obie orientacje są wyrównywane.
2. Średni kierunek jednostkowy wyznacza wspólną oś projekcji. Bierzemy część
   wspólną dwóch zakresów, nigdy ich sumę. Wymagamy minimalnej długości i udziału
   `długość wspólna / dłuższy zakres projekcji`; krótki fragment przy długiej
   krawędzi nie otrzymuje automatycznie udziału 100%.
3. Na obu końcach wspólnego zakresu interpolujemy punkty na obu krawędziach.
   Szerokość mierzymy wzdłuż normalnej do średniego kierunku. Obie szerokości
   muszą mieścić się w zadanych granicach. Odrzucamy zmianę strony (przecięcie)
   oraz zbyt duży stosunek większej szerokości do mniejszej. Dopuszcza to łagodne
   zwężanie perspektywiczne. Poza częścią wspólną niczego nie ekstrapolujemy.
4. Końce osi to środki tych dwóch przekrojów. Nie uśredniamy oryginalnych końców
   o różnych pozycjach wzdłuż pasa.
5. Dopuszczalne pary sortujemy kolejno: większy udział nakładania, mniejsza
   średnia szerokość, mniejsza różnica kątów, dłuższa część wspólna, rosnące ID
   obu krawędzi. Przyjmujemy parę, jeśli żadna krawędź nie jest już zajęta.
   Każda krawędź należy do najwyżej jednej pary, bez łańcuchowego scalania.

To deterministyczny algorytm zachłanny, bez ukrytej sumy wag i bez gwarancji
globalnie najlepszego dopasowania. Dokładne remisy rozstrzygają ID, nie pewność
semantyczna. Krawędzie za krótkie na minimalne nakładanie nie uczestniczą
w porównaniach, ale zostają w danych. Liczba porównań rośnie kwadratowo względem
liczby dostatecznie długich reprezentantów.

## Jawne parametry

Szerokości i długości są w pikselach **obrazu wejściowego**, bez normalizacji
rozdzielczości i bez przeliczania na metry. Granice są domknięte.

| Opcja skryptu | Argument API | Domyślnie | Znaczenie |
| --- | --- | --- | --- |
| `--marking-min-width` | `min_width_px` | 2 px | Najmniejsza szerokość przy obu końcach |
| `--marking-max-width` | `max_width_px` | 12 px | Największa szerokość przy obu końcach |
| `--marking-angle-tolerance` | `angle_tolerance` | 3° | Największa różnica orientacji |
| `--marking-min-overlap` | `min_overlap_px` | 30 px | Najkrótszy wspólny zakres |
| `--marking-min-overlap-ratio` | `min_overlap_ratio` | 0,6 | Najmniejszy udział względem dłuższej krawędzi |
| `--marking-max-width-ratio` | `max_width_ratio` | 1,5 | Maksymalne zwężanie w części wspólnej |

API wymaga jawnego podania `min_width_px` i `max_width_px`; pozostałe argumenty
mają powyższe wartości domyślne. Wymagamy skończonych wartości,
`0 < min_width_px <= max_width_px`, `0 <= angle_tolerance < 90`,
`min_overlap_px > 0`, `0 < min_overlap_ratio <= 1`, `max_width_ratio >= 1`.
Reprezentanty muszą mieć unikalne ID oraz skończoną, niezerową geometrię.

To zakres startowy do diagnostyki, nie uniwersalna kalibracja kamer. Górny próg
szerokości ma opisywać farbę, a nie rozstaw sąsiednich miejsc. Poszerzenie go do
rozstawu miejsc umożliwi błędne pary granic dwóch stanowisk. Dla odległych miejsc
także ich rozstaw może wejść w ten zakres — sama geometria tego nie rozstrzygnie.

## Diagnostyka

Z katalogu `apps/engine`, bez zmiany wartości domyślnych:

```bash
.venv/bin/python scripts/run_parking_batch.py --output /tmp/parkvision-marking-axes
```

`--samples` nadal pozwala wybrać jedno zdjęcie. `--min-length` nadal dotyczy
wyłącznie podglądu LSD. Grupowanie zachowuje dotychczasowe 3°, 2 px i 12 px.
Każde zdjęcie otrzymuje wcześniejsze `preview_lsd.png`, `preview_groups.png`,
`preview_fitted_lines.png`, `segments.json`, `source.txt` oraz nowe:

- `preview_marking_axes.png`: osie zielone, sparowane reprezentanty turkusowe,
  niesparowane szare;
- `markings.json`: parametry, grupy, reprezentanty, pary, osie i niesparowane ID.

Podgląd grup nadal rysuje rzeczywiste fragmenty z przerwami. Reprezentant i oś
opisują zakresy i mogą obejmować przerwy; nie są dowodem ciągłości farby.

## Ograniczenia przyjęte w MVP

- Brak rektyfikacji perspektywy, lokalnej kalibracji szerokości, ROI i weryfikacji
  koloru/kontrastu wewnątrz pasa. Bliskie krawężniki, cienie i detale samochodów
  mogą przejść te same testy geometryczne. Przeciwne kierunki LSD nie są
  wykorzystywane jako dowód dwóch stron jasnego pasa.
- Obecne grupowanie z tolerancją odległości 2 px potrafi już scalić obie strony
  bardzo cienkiej farby. Parowanie nie odzyska utraconej separacji; celowo nie
  zmieniamy sprawdzonego grupowania w tym etapie.
- Silna perspektywa, łuk lub szerokość poza zakresem mogą pozostawić poprawną
  fizyczną parę niesparowaną. Progi dotyczą wspólnego widocznego zakresu.
- Oś korzysta z zakresów reprezentantów, nie z przecięcia sum przedziałów
  rzeczywistego pokrycia fragmentami. Nawet nakładanie obejmujące głównie przerwy
  może utworzyć hipotezę. Źródła są zachowane do późniejszej oceny pokrycia.
- Przy zasłonięciu, gdy jedna strona dzieli się na kilka grup, druga może dostać
  tylko jednego partnera. Minimalny udział względem dłuższej strony dodatkowo
  odrzuca krótkie fragmenty. Pozostałe grupy nie są usuwane ani uznawane za błędy.
- Dopasowanie TLS nie dostarcza miary jakości; reprezentant źle dopasowany do
  tekstury może również wejść do pary. Ten etap nie ocenia ponownie grupowania.

Testy obejmują geometrię, obrót i odwracanie końców, częściowe nakładanie,
granice szerokości, łagodną perspektywę, przecięcia, konflikty i niezależność od
kolejności, walidację oraz zachowanie źródeł. Testy skryptu sprawdzają mapowanie
JSON, przerwy w poprzednich podglądach, niesparowane krawędzie, przekazanie opcji
i pełną ścieżkę od syntetycznego obrazu przez rzeczywiste LSD do osi.

## Wynik weryfikacji na bieżącym repozytorium

Uruchomiono skrypt na wszystkich dziewięciu zdjęciach, podając wyłącznie
`--output`, zgodnie z poprzednim uruchomieniem w `apps/engine/tmp`. Nowe wyniki
są w katalogu `marking-axes-default` w workspace dokumentów. Wszystkie pięć
wcześniejszych plików każdego zdjęcia porównano skrótami SHA-256 z `tmp`:
są identyczne bajtowo (45 plików).

| Zdjęcie | Pary geometryczne | Niesparowane reprezentanty |
| --- | ---: | ---: |
| `01-empty-parking-aerial.jpg` | 110 | 3137 |
| `01-pklot-ufpr05-empty.jpg` | 44 | 1216 |
| `02-empty-parking-shadows.jpg` | 453 | 3865 |
| `02-pklot-ufpr04-empty.jpg` | 38 | 2016 |
| `03-cnrext-camera8.jpg` | 17 | 397 |
| `03-pklot-ufpr05-occupied.jpg` | 28 | 2328 |
| `04-cnrext-camera3-wet.jpg` | 17 | 608 |
| `05-cnrext-camera1-occupied.jpg` | 26 | 2032 |
| `parking.jpg` | 151 | 1242 |

Obejrzano podglądy lotniczy, pusty PKLot UFPR05 i `parking.jpg`. Osie pokrywają
liczne widoczne pasy, ale także niektóre krawężniki, detale samochodu i inne
bliskie krawędzie. Część farby pozostaje niesparowana. Wyniki potwierdzają
przydatność jako etap diagnostyczny; podane liczby nie są liczbami wykrytych
prawidłowych oznaczeń ani miejsc parkingowych. Geometryczny środek w obrazie
nie zastępuje osi w układzie metrycznym po kalibracji kamery.

`pytest -q`: **636 passed** (przed zmianą: 579). Black i Flake8 sprawdzono dla
wszystkich czterech zmienionych/dodanych plików Python. Black uruchamiano osobno
dla każdego pliku, aby ominąć ograniczenie uruchamiania procesów roboczych
w tym środowisku. Nie utworzono commita.
