# Partia 03: eksport i konkretna korekta jakości

## Aktualny eksport v2 — uwagi zamknięte

2026-11-01. Użytkownik zmienił duży poligon na asfalcie na background,
przywrócił mały fragment krawężnika i oznaczył dwie niepewne plamy jako ignore.
Wyeksportowano `export-reviewed-v2`, bez modyfikowania źródłowych anotacji/flag
przez asystenta i bez nadpisania jakiegokolwiek pliku v1.

Porównanie piksel po pikselu: **265 curb → background, 61 paint → ignore**.
Wszystkie **7 pikseli** małego krawężnika pozostało curb. Nie ma zmian poza
wskazanymi obszarami. Maski dwóch pozostałych zdjęć są identyczne bajtowo z v1.
Obejrzano zbliżenie oryginał/v1/v2, także granicę tła z krawężnikiem.
Wskazane uwagi są zamknięte; to przegląd o określonym zakresie, nie deklaracja
niezależnego rozstrzygnięcia każdego piksela.

| Zakres v2 | background | paint | curb | ignore | Nadzorowane |
| --- | ---: | ---: | ---: | ---: | ---: |
| Partia 03 | 34 723 | 1873 | 723 | 2 727 481 | 37 319 |
| Bieżące 11 obrazów | 355 417 | 23 185 | 7811 | 29 875 987 | 386 413 |

Liczba poligonów partii pozostaje **62**: paint 15, background 33, curb 12,
ignore 2. Łącznie w bieżącej puli: **11 masek i 198 poligonów**. Nadal wszystkie
pilot; tylko jeden parking ma potwierdzone oryginalne źródło i warunki danych.
Nie rozpoczęto treningu, nie przypisano train/val/test i nie utworzono commita.

Artefakty v2: `dataset.json`, `summary.json`, `changes-from-v1.json`,
`quality-review.json`, `source-annotations/`, `review/overview.jpg` oraz
`review/corrections-before-after.png`. Hashe 7 wejść v2, wszystkich plików
eksportu v1 i 26 chronionych plików wcześniejszych partii pozostały zgodne.
Wszystkie 11 aktualnych masek odtworzono z JSON-ów bez różnic.

## Historyczny zapis kontroli v1 — przed ręczną poprawką

2026-11-01. Użytkownik zapisał trzy anotacje z `reviewed=true`, łącznie
**62 poligony**. Istniejący `prepare.py inspect/convert` wyeksportował 3 maski
i nakładki. **Technicznie eksport jest poprawny; kontrola wizualna wskazuje
jedną pomyłkę klasy i dwa niejednoznaczne fragmenty wymagające ręcznego przeglądu.**
Nie zmieniano źródłowych JSON-ów, flag ani masek. Bez treningu i commita.

Katalog eksportu:
`/home/karoljaron/Dokumenty/ChatGPT/System parkingowy/paint-segmentation-batch-03-2026-11-01/export-reviewed-v1/`.

## Uwagi wykryte w v1 na pierwszym, słonecznym zdjęciu

Plik `01-pklot-pucpr-sunny-20121027-143100.json`:

1. **Poligon 16, curb** (numer 1-based w zapisanej liście shapes), okolice
   x=1110–1140, y=364–383: duży trójkąt pod wąskim krawężnikiem przy niebieskim
   aucie obejmuje asfalt. Ma 265 pikseli po rasteryzacji. Usuń ten poligon lub
   zmień go na ignore; pozostaw wąskie fragmenty rzeczywistego krawężnika powyżej.
   Jeśli chcesz oznaczyć tło, nowy background powinien obejmować tylko pewny
   asfalt z marginesem od fizycznego krawężnika.
2. **Poligony 22 i 23, paint**: małe jasne plamy przy autach na prawej drodze.
   Na dostępnej rozdzielczości nie potwierdzono czytelnego obrysu farby. To
   nie jest stwierdzenie, że plamy są background. Przejrzyj je; jeśli nie można
   odróżnić farby od rozjaśnienia lub odbicia, zmień na ignore albo usuń poligon.

Dokładne dowody i lokalizacja:
`review/needs-correction-sunny-curb16.png` oraz
`review/needs-review-sunny-paint22-23.png`.

Nie trzeba odtwarzać całej anotacji. W pozostałych obejrzanych fragmentach
nie stwierdzono kolejnej jednoznacznej pomyłki klasy; nadal obowiązuje niepewność
rozmytych, cienkich brzegów. Przegląd miał określony zakres, nie rozstrzygał
niezależnie każdego piksela całych obrazów.

Odznacz reviewed na czas poprawek. Po sprawdzeniu i zapisie zatwierdź ponownie.
Wykonaj eksport do **export-reviewed-v2**; zachowaj v1 jako zapis przed korektą.
Nie edytuj samych masek/hashów ani kopii w `source-annotations/`.

## Liczniki eksportu v1 — przed korektą

| Zakres | Obrazy | Poligony | background | paint | curb | ignore |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Partia 03 | 3 | 62 | 34 458 | 1934 | 988 | 2 727 420 |
| Wszystkie eksporty | 11 | 198 | 355 152 | 23 246 | 8076 | 29 875 926 |

Nowa partia: **37 380 nadzorowanych pikseli**, 17 poligonów paint,
13 curb i 32 background. Cała wybrana pula: **386 474 nadzorowane piksele**.
To liczniki zapisanych etykiet, w tym wskazanej pomyłki v1, a nie liczba
bezbłędnych pikseli czy wynik modelu. Wszystkie flagi reviewed pozostawiono
zgodnie z zapisem użytkownika; oddzielny `quality-review.json` wskazuje status
`requires_manual_correction`.

## Weryfikacja i artefakty

Przeszły kontrole źródeł, SHA256, ścieżek, wymiarów, klas, geometrii,
konfliktu paint/curb i bezstratnego PNG. Maski wszystkich 11 eksportowanych
obrazów są identyczne z ponowną rasteryzacją właściwych anotacji. Kopie JSON-ów
nowego eksportu zachowano w `source-annotations/`. Sprawdzono zachowanie
7 bieżących plików wejściowych i 26 chronionych plików wcześniejszych partii.

Obejrzano 3 pełne pary oryginał/nakładka, 9 proponowanych okien oraz wszystkie
17 poligonów paint i 13 curb w osobnych powiększeniach. `review/overview.jpg`
pokazuje całość; obrazy `*.foreground-*.png` zawierają powiększenia poligonów.
`summary.json`, `quality-review.json` i nadrzędne `readiness.json` zawierają
liczniki, lokalizacje uwag i rozróżnienie walidacji technicznej od jakości.

Nie zmieniano kodu ani zależności; sprawdzenia danych korzystały z istniejącego
importera. Nie powtarzano Black/Flake8/Mypy ani testów niezmienionego kodu.
Źródła CV i wcześniejsze obrazy, anotacje oraz maski pozostały niezmienione.
HEAD nadal `552808e84be3a4e1445d2841253169ae937763e7`.

Trzy PUCPR mają potwierdzone oryginalne źródło i CC BY 4.0, ale przedstawiają
jeden parking. Wcześniejsze osiem nadal czeka na potwierdzenie proweniencji.
Całość pozostaje pilot; train/val/test 0/0/0, AVM poza A. Po lokalnej poprawce
nadal potrzebne są niezależne zweryfikowane grupy i zamrożenie etykiet przed
uczciwym treningiem. Obecnego v1 nie należy traktować jako gotowego GT treningowego.
