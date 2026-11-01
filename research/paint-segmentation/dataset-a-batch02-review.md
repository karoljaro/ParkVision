# Druga partia: ręczne anotacje i eksport masek

2026-11-01. Użytkownik zapisał i oznaczył `reviewed=true` na wszystkich czterech
zdjęciach. Istniejący `prepare.py convert` wyeksportował **4 maski uint8 PNG
i 4 nakładki** ze **107 poligonów**. Nie zmieniano poligonów ani flag użytkownika.
HEAD pozostał `552808e84be3a4e1445d2841253169ae937763e7`; bez commita i treningu.

## Artefakty i odtworzenie

Katalog partii:
`/home/karoljaron/Dokumenty/ChatGPT/System parkingowy/paint-segmentation-batch-02-2026-10-30/`.

- `images/*.json`: ręczne anotacje użytkownika.
- `export-reviewed-v1/dataset.json`: eksport w istniejącym formacie v1,
  z hashami, ścieżkami, wymiarami i licznikami pikseli.
- `export-reviewed-v1/*.mask.png`: jednokanałowe maski 0/1/2/255.
- `export-reviewed-v1/*.overlay.png`: nakładki do kontroli, nie etykiety treningowe.
- `export-reviewed-v1/source-annotations/`: kopie dokładnych JSON-ów użytych do
  eksportu, identyczne bajtowo z zapisami użytkownika w chwili konwersji.
- `export-reviewed-v1/review/overview.jpg`: cztery oryginały obok nakładek.
- `export-reviewed-v1/review/*.roi-review.png`: oryginał, nakładka i obrysy dla
  16 proponowanych okien; dodatkowo wszystkie poligony paint/curb camera8 osobno.
- `export-reviewed-v1/quality-review.json`: zakres i ograniczenia przeglądu.
- `export-reviewed-v1/combined-manifest.json` i `combined-inspect.json`:
  wspólna inspekcja 8 rekordów, bez modyfikacji manifestu starego pilota.
- `readiness.json`: zaktualizowane liczniki; wcześniejszy stan zachowano
  w `audit/readiness-before-export-2026-11-01.json`.

Wykonano istniejące polecenia `inspect` i `convert` na manifeście partii.
W przyszłości po zmianie JSON-ów użyj **nowego** katalogu eksportu v2. Eksport v1
pozostaje zapisem tej wersji etykiet; nie podmieniaj w nim samych masek/hashów.
Instrukcja LabelMe i pełne polecenia są w README partii.

## Liczniki rzeczywistych etykiet

| Zdjęcie | Poligony | background | paint | curb | ignore |
| --- | ---: | ---: | ---: | ---: | ---: |
| aerial | 36 | 255 241 | 17 470 | 3023 | 14 753 266 |
| CNR camera8 | 31 | 7037 | 1011 | 1459 | 740 493 |
| CNR camera1 occupied | 19 | 7194 | 612 | 755 | 741 439 |
| PKLot UFPR04 | 21 | 15 453 | 1081 | 647 | 904 419 |
| **Partia 02** | **107** | **284 925** | **20 174** | **5884** | **17 139 617** |
| **Obie partie** | **136** | **320 694** | **21 312** | **7088** | **27 148 506** |

W samej partii 02 są 32 poligony paint, 19 curb, 50 background i 6 ignore.
Łącznie z wcześniejszym pilotem: **8 obrazów, 8 masek, 136 poligonów,
349 094 piksele nadzorowane**. To liczniki adnotacji, nie wyniki modelu.
Nieopisane piksele pozostają ignore. W repo nadal jest jedno nieoznaczone
zdjęcie rezerwowe UFPR05 occupied; nie jest uwzględnione w sumach pikseli.

## Kontrola techniczna i wizualna

Sprawdzono `reviewed`, ścieżki obrazów, imageData=null, wymiary, hashe i klasy,
rasteryzację poligonów, konflikt paint/curb oraz bezstratny zapis PNG.
Ponownie zbudowano maski wszystkich **8** obrazów z ich bieżących JSON-ów;
są identyczne z zapisanymi eksportami i licznikami. Wszystkie 9 plików wejściowych
nowej konwersji (manifest, 4 obrazy, 4 JSON-y) pozostały niezmienione w trakcie pracy.

Obejrzano 4 pełne kadry obok nakładek, 16 proponowanych okien oraz wszystkie
12 poligonów paint i 5 curb z camera8 w osobnych powiększeniach. W tym zakresie
nie znaleziono oczywistego pomylenia klas uzasadniającego odrzucenie partii.
Dodane przykłady obejmują farbę w cieniu i przy aucie, słabą farbę, krawężniki
i trudne negatywy. Nie wszystkie ramki guide zawierają etykiety; były propozycją
lokalizacji, a opisane ROI określają rzeczywisty zakres nadzoru.

To przegląd o określonym zakresie, nie niezależne rozstrzygnięcie każdego piksela.
Na camera8 pozostaje niepewność rozmytych/zużytych brzegów i małych końcówek.
Przed przyszłym zamrożeniem GT wskazany jest osobny przegląd tych brzegów.
Nie wypełniano automatycznie ubytków ani nie przesuwano obrysów. Wcześniejsze
uwagi do brzegów starego pilota pozostają zachowane i nierozstrzygnięte.

Wszystkie **248 plików** chronionych początkowym spisem starego pilota mają
niezmienione hashe; niczego nie usunięto. Od tego spisu pojawił się jedynie
dodatkowy plik ustawień GUI LabelMe. Nie zmieniał on obrazów, poligonów ani masek.

Test importera: **52 passed** (`pytest -q research/paint-segmentation/test_prepare.py`).
Nie zmieniono kodu, zależności silnika ani wyników CV. Black/Flake8/Mypy nie
uruchamiano dla zmian wyłącznie dokumentacji i danych. Nie uruchamiano overfitu.

## Gotowość do dalszych prac

Całość pozostaje **split=pilot**; train/val/test nadal **0/0/0**. Tożsamość
oryginalnych plików, licencje i parking/camera/session IDs pozostają niepotwierdzone.
UFPR04/05 oraz kamery CNR wymagają wspólnego traktowania w swoich grupach.
Sukces inspekcji przy unknown nie poświadcza niezależności parkingów/sesji.

AVM nadal poza A; audyt masek nie rozstrzygnął licencji danych ani grupowania.
Następny warunek uczciwego treningu: zweryfikowana proweniencja i prawa,
więcej niezależnych parkingów, przegląd i zamrożenie GT oraz podział całymi
powiązanymi grupami. Te 8 obrazów nie stanowi jeszcze niezależnej walidacji.
