# Rozbudowa danych A: druga partia i AVM SS

**Aktualizacja 2026-11-01:** użytkownik ręcznie zatwierdził drugą partię;
wyeksportowano 4 maski ze 107 poligonów. Aktualne liczniki i kontrola jakości
są w [raporcie eksportu](dataset-a-batch02-review.md). Poniższy opis dokumentuje
przygotowanie partii i audyt źródeł w dniu 2026-10-30.

2026-10-30. Przygotowano **4 zdjęcia do ręcznej anotacji**, 16 wskazanych okien
ROI, manifest v1 i podglądy. Nowych poligonów/masek: **0**. Jest to realna
rozbudowa kolejki danych; liczba zatwierdzonych etykiet jeszcze nie wzrosła.
Sprawdzono też wszystkie 6763 oryginalne maski z dostarczonego ZIP-a AVM SS,
lecz pozostają poza A z powodu niepotwierdzonej licencji i podziału obiektów.

## Rzeczywisty stan wejścia

Repo `/home/karoljaron/Projects/ParkVision`, HEAD
`552808e84be3a4e1445d2841253169ae937763e7`; `git status --short` na wejściu pusty.
Nie utworzono commita. Zmiany w repo ograniczają się do tego raportu i odnośnika
w README. Nie zmieniono importera, overfitu, silnika, zależności ani pipeline CV.

Sprawdzono istniejące `prepare.py init/inspect/convert`, manifest v1 i LabelMe
z globalnym reviewed. Nie powstał nowy importer, UI ani `train.py`.
Nie uruchamiano treningu. Dawny wynik overfitu po 100 krokach pozostaje testem
zapamiętania 25 wycinków 128×128, bez wartości walidacyjnej.

Pilot z
`/home/karoljaron/Dokumenty/ChatGPT/System parkingowy/paint-segmentation-pilot-2026-10-30/draft-roi-v1/export-reviewed-v1/dataset.json`
sprawdzono względem bieżącego manifestu, SHA256 zdjęć/anotacji/masek, reviewed,
imagePath oraz ponownej rasteryzacji JSON-ów. Wszystkie 4 maski są identyczne
z rekonstrukcją i licznikami eksportu. Zachowano 248 plików drzewa pilota,
w tym wcześniejsze uwagi do jakości brzegów; nie poprawiano ich automatycznie.

## Co dodano poza repo

Katalog:
`/home/karoljaron/Dokumenty/ChatGPT/System parkingowy/paint-segmentation-batch-02-2026-10-30/`.

Najpierw otwórz jego **README.md** i **annotation-guide.jpg**. README zawiera
sprawdzone polecenie LabelMe 5.9.1 i eksportu; `roi-plan.json` oraz
`guides/*.details.jpg` prowadzą do konkretnych miejsc w natywnej rozdzielczości.
Ramki to wskazówki lokalizacji, nie etykiety. Nie tworzono nieprecyzyjnych
poligonów DRAFT ani fikcyjnych masek. Każda kopia JPEG ma SHA256 źródła.

| Nowe zdjęcie | Cel | Niezależność |
| --- | --- | --- |
| `01-empty-parking-aerial` | Linie w cieniu, krawężnik, tekstura i auto. | Inny wygląd obiektu; pochodzenie i niezależność niepotwierdzone. |
| `03-cnrext-camera8` | Słaba farba, jasna plama, gałęzie/lampa, krawężnik. | Powiązane z pilotowym CNR camera3; dodatkowa kamera nie jest nowym parkingiem. |
| `05-cnrext-camera1-occupied` | Linie przy autach, jasne karoserie, plamy i wyspa. | Kolejny widok deklarowanego CNRPark. |
| `02-pklot-ufpr04-empty` | Mokre/jasne smugi, zużyta farba i wyspy. | Ten sam parking co UFPR05 z pilota, inny widok. |

Spośród 9 zdjęć `samples` cztery należą do reviewed pilota, cztery do nowej
kolejki, a zajęty UFPR05 pozostaje nieoznaczoną rezerwą. Nie znaleziono dokładnych
duplikatów zdekodowanych pikseli, co nie wyklucza powiązań widoków i sesji.

[Dokument autorów PKLot](https://www.inf.ufpr.br/lesoliveira/download/pklot-readme.pdf)
potwierdza, że UFPR04/05 to dwa widoki jednego parkingu. [Źródło CNRPark](https://cnrpark.it/)
opisuje 9 kamer jednego parkingu. Bezpośrednie otwarcie CNR zwracało 502;
tekst strony był dostępny w indeksie wyszukiwarki. Te relacje stosujemy
konserwatywnie do lokalnych plików nazwanych jako te zbiory, z potwierdzeniem
wizualnym scen; nie odzyskano ich oryginalnych ścieżek/dat ani hashów źródłowych.
Zidentyfikowanych i potwierdzonych **nowych niezależnych parkingów: 0**.

[Strona właścicieli PKLot](https://web.inf.ufpr.br/vri/databases/parking-lot-database/)
podaje CC BY 4.0 dla danych. Dla CNRPark-EXT zaindeksowany tekst strony deklaruje
ODbL 1.0, lecz nie sprawdzono warunków oryginalnego archiwum. Nie przypisano tych
licencji lokalnym JPEG-om na podstawie nazwy. Ich manifesty nadal mają
`license=unknown`; oryginalne parking/camera/session IDs także pozostają unknown.
Trzy pozostałe sceny pilota/partii (aerial, shadows, parking) mają nieustaloną
proweniencję. Napis Camera 01 na `parking.jpg` nie oznacza CNR camera1.

## Policzona gotowość

| Zakres | Obrazy reviewed | Poligony reviewed | Obrazy/poligony z anotacją DRAFT | Nieoznaczone zdjęcia |
| --- | ---: | ---: | ---: | ---: |
| Stary pilot | 4 | 29 | 0 / 0 | 0 |
| Partia 02 | 0 | 0 | 0 / 0 | 4 |
| Rezerwa samples | 0 | 0 | 0 / 0 | 1 |

29 poligonów: background 16, paint 9, curb 3, ignore 1.

| Piksele | Rzeczywiste maski reviewed pilota | Nowa partia |
| --- | ---: | --- |
| background = 0 | 35 769 | brak masek |
| paint = 1 | 1138 | brak masek |
| curb = 2 | 1204 | brak masek |
| ignore = 255 | 10 008 889 | 17 450 600 nieopisanych pikseli; nie zapisano PNG |

Łącznie jest nadal **38 111 pikseli nadzorowanych**. Dla ośmiu wybranych obrazów
27 459 489 pikseli jest ignore lub bez anotacji. Licznik nie obejmuje rezerwy
ani AVM. Wszystkie 8 rekordów wspólnego manifestu pozostaje w pilot;
train/val/test: **0/0/0**. `surface=unknown` zachowano w metadanych, zamiast
utożsamiać wygląd pojedynczej plamy z potwierdzonym stanem nawierzchni.

`readiness.json`, `audit/sample-inventory.json` i `audit/provenance-review.json`
zawierają dane do dalszego przeglądu. `audit/combined-manifest.json` i wynik
`inspect` łączą partie bez modyfikacji pilota. Przy nieznanych ID pomyślne
inspect nie dowodzi braku przecieku parkingów/kamer/sesji.

## AVM SS: oryginalne maski obejrzane, dopuszczenie NIEZWERYFIKOWANE

Publiczny punkt wejścia: [repo autorów](https://github.com/Engro39/avm_dataset)
i linkowany tam Drive. Po nieudanej próbie pobrania narzędziem przeglądarki
użytkownik dostarczył `/home/karoljaron/Pobrane/avm_seg_db.zip` (252 992 120 B).
SHA256: `b9a8f70fcb9f57bb1c8360034cbf09b06277b2361453edfa5039960cd33c52ef`.
Bez checksumu wydawcy pochodzenie opiera się na przekazanym ZIP-ie i zgodności
z opublikowanym opisem.

Sprawdzono wszystkie 6763 pary JPG/PNG i obecność odpowiadających YAML-ów.
Maski są RGB uint8 320×160, z rzeczywistą klasą marker. Oprócz palety autorów
znaleziono **7071 żółtych pikseli w 540 maskach**. W dwóch sprawdzonych próbkach
odpowiadają dokładnie nakładaniu poligonów curb/vehicle; nie utożsamiono ich
z nową klasą ani nie naprawiono. Zielone other nie jest osobnym GT curb A.

Listy mają 4057 train / 2706 test bez wspólnych nazw i bez identycznych
zdekodowanych obrazów. Jednak `split_db.py` tasuje pojedyncze klatki, bez grup.
`00003756` test i `00003757` train po obejrzeniu pokazują niemal identyczną scenę
garażu. Metadane nie dostarczają parking/camera/session IDs. Nie potwierdzono
podziału na niezależne parkingi, kamery lub sesje.

W sprawdzonym opisie autorów i dostarczonym archiwum nie znaleziono licencji
**plików danych**. Dostęp techniczny jest potwierdzony; warunki użycia pozostają
NIEZWERYFIKOWANE. Licencji kodu nie przenoszono na dane. Do A dodano **0 par**.
Pełny audyt, surowe próbki i podglądy są w `external-avm-audit/` nowej partii;
liczniki AVM są oddzielone od liczników A. Nie badano kolejnych kandydatów.

## Weryfikacja i następny warunek

- `pytest -q apps/engine/tests research/paint-segmentation` w środowisku engine:
  **807 passed, 1 skipped** (opcjonalne testy PyTorch).
- `pytest -q research/paint-segmentation` w istniejącym środowisku overfitu:
  **69 passed**; sam overfit nie był uruchamiany.
- `init`, `inspect` partii i wspólnego manifestu wykonane; konwersja partii bez
  anotacji prawidłowo odrzucona przed utworzeniem wyjścia.
- Ponowna rasteryzacja 4 zatwierdzonych JSON-ów jest identyczna z maskami,
  hashe i liczniki zgadzają się. Oryginały i stare artefakty pozostają zachowane.
- Black/Flake8/Mypy nie uruchamiano: brak zmian kodu. Źródła CV i jego zależności
  pozostają identyczne z HEAD; nie regenerowano ani nie nadpisywano wyników CV.

Następny krok człowieka: narysować poligony, przejrzeć 200–400%, ręcznie
zatwierdzić reviewed i wykonać istniejący convert do nowego katalogu; następnie
obejrzeć overlay i brzegi w skali pikselowej. Instrukcja i polecenia są w README
partii. To nadal nie wystarczy do uczciwego treningu: potrzebne są zweryfikowane
źródła/licencje, wiele niezależnych parkingów, wspólne grupowanie powiązanych
kamer/sesji oraz zamrożone train/val/test całymi grupami z nietkniętym testem.
Trzy parkingi pozwalają jedynie formalnie rozdzielić trzy splity; nie gwarantują
wiarygodnej oceny. Wybór U-Net/SegFormer/DeepLab i sprzętu pozostaje otwarty,
bez ograniczania modeli możliwościami laptopa. Nie wdrażano fuzji CV+ML.
