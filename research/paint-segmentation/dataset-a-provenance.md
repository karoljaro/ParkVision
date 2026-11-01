# Dataset A: odzyskane pochodzenie i pula z trzech parkingów

2026-11-01. Potwierdzono pochodzenie pięciu wcześniej oznaczonych obrazów
PKLot/CNRPark-EXT oraz jednego nieoznaczonego obrazu rezerwowego. Nie trzeba
powtarzać anotacji. Z trzema PUCPR daje to **8 obrazów, 150 ręcznie zatwierdzonych
poligonów i 3 fizyczne parkingi** z ustalonym źródłem i deklaracją licencji danych.
Nie zmieniono starych obrazów, poligonów, flag reviewed, masek ani eksportów.
Nie zmieniono kodu, zależności ani pipeline CV. Bez treningu i commita.

## Co rzeczywiście sprawdzono

Autorzy [MetaPKLot](https://github.com/DSBD-Research/MetaPKLot-Dataset) udostępniają
pełne zdjęcia PKLot i CNRPark-EXT. Odkrywanie przez Git blob SHA1 wskazało sześć
dopasowań. Następnie pobrano **całe JPEG-i** z przypiętego commita
`9218800c4eb2b7c1b17b51c8cdb07ad51686f8a2`: wszystkie są identyczne bajtowo
z lokalnymi plikami. Sprawdzono także SHA256 oraz ścieżki i identyfikatory blobów
w drzewie `4db0db75525b0c1e0a30213fb89aa06ed6dea7eb` tego commita.
Nie odnaleziono dawnego ZIP-a; jest to odtworzenie źródła przez zgodność plików,
nie deklaracja odzyskania historii wcześniejszego pobrania.

| Lokalny obraz | Ścieżka w przypiętym MetaPKLot | Stan |
| --- | --- | --- |
| 04-cnrext-camera3-wet | CNRPark-EXT/camera3/RAINY/2015-11-21/2015-11-21_10_15_00.jpg | reviewed, stary pilot |
| 03-cnrext-camera8 | CNRPark-EXT/camera8/SUNNY/2015-11-22/2015-11-22_08_06_00.jpg | reviewed, partia 02 |
| 05-cnrext-camera1-occupied | CNRPark-EXT/camera1/OVERCAST/2015-11-25/2015-11-25_10_40_00.jpg | reviewed, partia 02 |
| 01-pklot-ufpr05-empty | PKLot/UFPR05/Sunny/2013-02-24/2013-02-24_08_00_02.jpg | reviewed, stary pilot |
| 02-pklot-ufpr04-empty | PKLot/UFPR04/Sunny/2012-12-08/2012-12-08_08_00_03.jpg | reviewed, partia 02 |
| 03-pklot-ufpr05-occupied | PKLot/UFPR05/Sunny/2013-02-22/2013-02-22_08_00_02.jpg | rezerwa, bez masek |

Licencja **plików danych**, potwierdzona w deklaracjach właścicieli:

- PKLot: [strona UFPR](https://web.inf.ufpr.br/vri/databases/parking-lot-database/)
  podaje **CC BY 4.0**. Trzy PUCPR pozyskano wcześniej wprost z archiwum autorów.
- CNRPark-EXT: [strona projektu](http://cnrpark.it/) podaje **ODbL 1.0**.
  Zachowano pobraną stronę i jej hash; HTTPS nie działał poprawnie, odczyt
  oficjalnego adresu HTTP się powiódł. Nie wyłączano weryfikacji certyfikatów.
- README przypiętej wersji MetaPKLot potwierdza te osobne warunki. Nie przeniesiono
  licencji kodu na zdjęcia ani jednej licencji na całą mieszaną pulę.

To weryfikacja zdjęć i pochodzenia, nie import cudzych masek farby. Oryginalne
poligony miejsc, bboxy i etykiety zajętości nie zostały użyte jako paint/curb.
GT pozostaje ręczną anotacją użytkownika. Pochodzenie nie potwierdza automatycznie
idealnej jakości brzegów; wcześniejsze uwagi do lokalnych ROI nadal obowiązują.

## Grupy i liczniki

[README PKLot](https://www.inf.ufpr.br/lesoliveira/download/pklot-readme.pdf)
potwierdza, że UFPR04/05 to widoki tego samego parkingu. CNR opisuje dziewięć
kamer jednego obiektu w Pizie. PUCPR jest odrębnym parkingiem. Wszystkie kamery
i dni jednego fizycznego parkingu pozostają razem. Nazwy kamer oznaczają
identyfikatory widoków zbioru; numery sprzętowe, strefy czasowe i oryginalne
session IDs pozostają nieznane. Grupy dzienne są zachowawczymi grupami kuratora.

| Parking / proponowana rola | Obrazy | Poligony | background | paint | curb | ignore |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| UFPR / train | 2 | 27 | 45 372 | 1503 | 1701 | 1 794 624 |
| CNR Pisa / val | 3 | 61 | 15 967 | 1811 | 2364 | 2 229 858 |
| PUCPR / test | 3 | 62 | 34 723 | 1873 | 723 | 2 727 481 |
| Pula ze zweryfikowanym źródłem | **8** | **150** | **96 062** | **5187** | **4788** | **6 751 963** |
| Cała dotychczas oznaczona pula | 11 | 198 | 355 417 | 23 185 | 7811 | 29 875 987 |

W nowej puli jest **106 037 nadzorowanych pikseli**. Pozostałe są ignore, także
poza opisanymi ROI. Nie ma nowych draft poligonów. Trzy niezweryfikowane zdjęcia
`01-empty-parking-aerial`, `02-empty-parking-shadows` i `parking` zachowano
z ich 48 zatwierdzonymi poligonami, ale pominięto w puli ze zweryfikowanym źródłem.
AVM nadal poza A: maski rzeczywiście sprawdzono, licencja danych i niezależność
parkingów/sesji nadal nie są potwierdzone.

## Artefakty i odtworzenie

Katalog roboczy:
`/home/karoljaron/Dokumenty/ChatGPT/System parkingowy/paint-segmentation-provenance-2026-11-01/`.

- `curated-inputs/images/`: kopie 8 obrazów i ich JSON-ów bez zmiany bajtów.
- `curated-manifest.json`: uzupełnione pochodzenie, identyfikatory, licencje;
  wszystkie obrazy nadal **pilot**.
- `export-reviewed-v1/dataset.json`: nowy eksport przez istniejące `prepare.py`,
  8 masek i 8 nakładek. Maski identyczne bajtowo z wcześniejszymi eksportami.
- `proposed-group-split-manifest.json`: **propozycja**, train/val/test 2/3/3;
  nie zastępuje aktywnego pilota i nie była używana do treningu.
- `readiness.json`, `verification.json`: liczniki i wyniki kontroli.
- `verified-source-matches.json`, `immutable-tree-verification.json`, zapisane
  strony i README: dowody plików, przypiętych ścieżek i warunków danych.
- `excluded-unverified.json`, `mapped-unannotated-reserve.json`: jawne wyłączenia
  i jeden rozpoznany obraz UFPR bez GT.
- `ATTRIBUTION.md`: autorzy i odnośniki wymagane przy korzystaniu ze źródeł.

Uruchomione istniejące polecenia: `prepare.py inspect curated-manifest.json`,
`prepare.py inspect proposed-group-split-manifest.json` i
`prepare.py convert curated-manifest.json --output export-reviewed-v1`.
Ścieżki wejść rozwiązywano z powyższego katalogu; użyto Pythona z `.venv` silnika.
Eksportu nie należy nadpisywać; kolejny wymaga nowego katalogu.

Przeszły kontrole SHA256, wymiarów, klas, reviewed, konfliktów paint/curb,
PNG i rasteryzacji wszystkich ośmiu masek. Kopie obrazów/JSON-ów i nowe maski
porównano z oryginałami. **104 chronione pliki wcześniejszych partii pozostały
niezmienione**. Podział przechodzi kontrolę przecieku; próba przeniesienia
jednego UFPR do test przy drugim w train została prawidłowo odrzucona.
Nie zmieniano kodu, więc nie powtarzano Black/Flake8/Mypy ani testów pipeline CV.

## Następny krok

Istnieje już minimalna podstawa do eksperymentu z walidacją na innym parkingu.
To nadal tylko jeden parking na rolę i bardzo skąpe ROI: nie jest to wiarygodny
benchmark wdrożeniowy. Przed treningiem należy ustalić i zamrozić protokół,
wejścia oraz podział, a docelowo rozbudować zwłaszcza zbiór uczący. Obraz UFPR
z rezerwy musi pozostać w tej samej grupie co pozostałe UFPR.

Przyszły baseline musi startować niezależnie od checkpointa wcześniejszego
testu zapamiętania. UFPR i CNR występowały w tamtym teście; PUCPR nie. Walidacja
służy wyborom podczas eksperymentu; test PUCPR dopiero końcowej, zamrożonej
ocenie. Raportować wyniki tylko w reviewed ROI, wykluczając 255, osobno dla
paint/curb i poszczególnych obrazów/parkingów. Nie interpretować milionów pikseli
jako milionów niezależnych przykładów. Dobór architektury i miejsca obliczeń
pozostaje otwarty; laptop/CPU nie jest ograniczeniem wyboru modelu.
