# Trzecia partia: oryginalne zdjęcia PKLot PUCPR

**Aktualizacja 2026-11-01:** użytkownik opisał trzy obrazy i poprawił uwagi
przeglądu. Aktualny eksport v2 zawiera 3 maski z 62 poligonów; asfalt jest
background, dwie niepewne plamy ignore, mały krawężnik zachowano. Szczegóły
porównania z v1 są w [raporcie kontroli](dataset-a-batch03-review.md). Poniższy opis zachowuje
historyczny stan przygotowania partii przed ręczną anotacją.

2026-11-01. Przygotowano **3 pełne zdjęcia do ręcznej anotacji**, pobrane
bezpośrednio z oryginalnego archiwum wskazanego przez właścicieli PKLot.
Nowych poligonów ani masek nie utworzono. Wcześniejsze 8 masek i 136 poligonów
zachowano; nie uruchamiano treningu. HEAD nadal `552808e`, bez commita.

## Źródło danych i wykonane pobranie

[Strona właścicieli PKLot](https://web.inf.ufpr.br/vri/databases/parking-lot-database/)
podaje licencję **danych CC BY 4.0** i wskazuje archiwum
`https://www.inf.ufpr.br/vri/databases/PKLot.tar.gz`.
[README autorów](https://www.inf.ufpr.br/lesoliveira/download/pklot-readme.pdf)
opisuje PUCPR jako inny parking niż wspólny obiekt UFPR04/05.

Strumieniowo odczytano 18 493 440 B skompresowanego prefiksu i przeskanowano
2749 elementów. Zachowano 9 pełnych JPEG-ów PUCPR, po 3 z folderów Sunny,
Cloudy i Rainy, z różnymi datami w obrębie kategorii i godzinami 08–17.
Wszystkie dekodują się do 1280×720. Nie pobrano całego archiwum 4,9 GB i nie
deklarujemy jego pełnego hasha ani kompletnego audytu integralności.

Z 9 kandydatów wybrano 3 po obejrzeniu, kierując się czytelnością ROI,
oświetleniem i zajętością. Nie używano predykcji modelu ani wyników testowych.
W przeskanowanym zakresie znaleziono dwa rekordy z różną datą katalogu i pliku;
pominięto je bez zgadywania dnia przechwycenia. Nie uogólnia to kontroli na cały PKLot.

Oryginalny przykład XML ma `parking id=pucpr` i 100 miejsc z zajętością.
Zachowano go jako dowód rodzaju formatu, nie jako maskę farby ani parę do
wybranych JPEG-ów. Żadne bboxy, pola miejsc lub punkty nie zostały zrasteryzowane
jako GT paint.

## Nowa partia i metadane

Katalog:
`/home/karoljaron/Dokumenty/ChatGPT/System parkingowy/paint-segmentation-batch-03-2026-11-01/`.

| Kolejność LabelMe | Kadr | Pogoda według katalogu źródła |
| --- | --- | --- |
| 01 | 2012-10-27 14:31:00 | Sunny |
| 02 | 2012-10-28 13:27:06 | Cloudy |
| 03 | 2012-10-26 16:24:55 | Rainy |

`prepare.py init` skopiował oryginalne JPEG-i bez zmiany pikseli. Zachowano ich
hashe, oryginalne ścieżki archiwum i daty. Nadano jawne identyfikatory kuratora:
parking `pklot:PUCPR`, widok kamery `pklot:view:PUCPR`, grupa
`pklot:PUCPR:day:DATA`. Sprzętowy ID kamery, oryginalny session_id i strefa
czasowa przechwycenia pozostają nieznane. Grupa dnia jest zachowawczą jednostką
kuratora, a nie odkrytym ID sesji. Cały PUCPR musi pozostać w jednym splicie
przy ocenie na nowych parkingach. Trzy dni nie stanowią trzech niezależnych obiektów.

`source` wskazuje oryginalny URL i członka archiwum; `license=CC-BY-4.0` ma
odniesienie do strony właścicieli. `surface=unknown`: źródłowe Rainy nie dowodzi
aktywnego deszczu lub mokrości wszystkich powierzchni w konkretnym kadrze.
Nie przypisywano tej licencji wcześniejszym lokalnym zdjęciom na podstawie ich nazw.

W katalogu partii są README z poleceniami, `ATTRIBUTION.md`, manifest v1,
`provenance.json`, `readiness.json`, podgląd oraz **9 proponowanych okien ROI**.
Trzy numery w nazwach zapewniają identyczną kolejność w GUI, manifeście i guide.
Przewodnik wskazuje miejsca, nie etykiety; nie tworzy GT ani wymogu opisania
każdej ramki. Używane są istniejące klasy background/paint/curb/ignore.
Przygotowano lokalny launcher LabelMe z odrębnymi ustawieniami tej partii.

Surowe źródła i dowody pobrania:
`/home/karoljaron/Dokumenty/ChatGPT/System parkingowy/paint-segmentation-sources-2026-11-01/pklot/`.
Są tam zachowane strony i PDF z hashami, nagłówki HTTP, oryginalne nazwy
kandydatów, dokumentacja częściowego pobrania i sześć rezerwowych klatek.
Pełne zasady atrybucji i opis zmian nazw/podglądów zapisano przy obrazach.

## Gotowość i sprawdzenia

- Wybrana pula: **11 obrazów** — 8 reviewed, 3 bez anotacji.
- Nadal **8 masek, 136 poligonów i 349 094 piksele nadzorowane**:
  background 320 694, paint 21 312, curb 7088, ignore w istniejących maskach 27 148 506.
- Nowa partia: 2 764 800 nieopisanych pikseli, bez wygenerowanych masek.
- Oryginalne źródło i licencję plików potwierdzono dla 3 nowych obrazów,
  czyli jednego parkingu. Wcześniejsze 8 obrazów nadal czeka na mapowanie źródeł.
- Wszystkie 11 rekordów pozostaje pilot; train/val/test **0/0/0**. AVM poza A.
- `init` i `inspect` partii oraz wspólnego manifestu wykonane. Kopie są zgodne
  z hashami oryginalnych członków archiwum; daty katalogów i plików są zgodne.
  Nie znaleziono identycznych zdekodowanych obrazów w wybranej puli 11 kadrów.
- Hashe 26 sprawdzonych plików wcześniejszych eksportów i ich wejść pozostały
  niezmienione. Nie zmieniano wcześniejszych obrazów, JSON-ów, masek ani flag.
- Nie zmieniano kodu, zależności silnika, CV ani jego wyników. Nie wymagano
  nowych testów kodu; kontrola nowych danych korzysta z istniejącego importera.

Użytkownik doprecyzował, że wcześniejsze zdjęcia dostarczył GPT/Codex w ZIP-ie.
W bieżącym workspace, repo i Downloads nie znaleziono ZIP-a źródłowych zdjęć;
związany z zadaniem `avm_seg_db.zip` jest osobnym zbiorem. Dostępne ostatnie
fragmenty wskazanej rozmowy ChatGPT również nie odtworzyły pierwotnych URL-i.
Nie wyciągano wniosku o licencji ze sposobu dostarczenia plików. Ich anotacje
pozostają zachowane; możliwość odzyskania pochodzenia pozostaje otwarta.

Następny krok ręczny: oznaczyć małe, pewne ROI trzech nowych kadrów i zatwierdzić
reviewed. Do uczciwego treningu nadal potrzebne są zweryfikowane niezależne
parkingi, zamrożone etykiety oraz podział całymi powiązanymi grupami. PUCPR
zwiększa różnorodność względem UFPR, lecz nie rozwiązuje samodzielnie podziału A.
