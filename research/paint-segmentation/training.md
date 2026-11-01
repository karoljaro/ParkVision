# Pierwszy U-Net ResNet50 z niezależną walidacją

**Wykonany wynik:** [23 epoki, najlepsza epoka 8, CNR foreground mIoU 18,53%](training-results.md).
Predykcje i checkpointy zapisano; PUCPR nie oceniano. Poniżej kontrakt pipeline.

2026-11-01. Badawczy pipeline `train.py` jest oddzielony od silnika CV.
Używa zatwierdzonych ROI Dataset A i podziału całymi parkingami:
**UFPR → train (2 obrazy), CNR Pisa → val (3), PUCPR → test (3)**.
UFPR04/05 to widoki tego samego obiektu; kamery CNR również pozostają razem.
Nie wykorzystuje checkpointu wcześniejszego testu zapamiętania.

## Zamrożenie danych

Katalog eksperymentu:
`/home/karoljaron/Dokumenty/ChatGPT/System parkingowy/paint-segmentation-training-2026-11-01/`.

`frozen-v1/` zawiera osobne, identyczne bajtowo kopie 8 JPEG-ów, 8 JSON-ów
LabelMe i 8 masek. `dataset.json` przypisuje role, a `lock.json` zapisuje SHA256
każdego wejścia, manifestu i kopii źródłowych manifestów. Dotychczasowych
partii i ich oznaczenia pilot nie zmieniono. Mapa klas pozostaje
background=0, paint=1, curb=2, ignore=255.

| Rola | Obrazy | background | paint | curb | ignore |
| --- | ---: | ---: | ---: | ---: | ---: |
| UFPR / train | 2 | 45 372 | 1503 | 1701 | 1 794 624 |
| CNR / val | 3 | 15 967 | 1811 | 2364 | 2 229 858 |
| PUCPR / test | 3 | 34 723 | 1873 | 723 | 2 727 481 |

PUCPR objęto kontrolą integralności przy zamrożeniu. Loader modelu odmawia
wczytania split=test; PUCPR nie uczestniczy w uczeniu, augmentacji, wyznaczaniu
wag klas, wyborze checkpointu, predykcjach ani metrykach tego eksperymentu.
Liczby w tabeli opisują istniejące etykiety, nie wynik modelu na PUCPR.

SHA256 zamrożonego manifestu:
`0d714fa87bb521630c9301f0117e9069c269d8395a2d68be15e993f7fa8df109`.
SHA256 lock.json:
`4eab7cd3f3cfd010be4c316d16d27c6f34923477ff5f05b4eb2e2e0e90b6be46`.

Każdy eksport jest walidowany względem aktualnych ręcznych poligonów,
reviewed, wymiarów i SHA256. Nie wystarczy podmienić hash maski. Podział jest
sprawdzany pod kątem wspólnych parkingów, kamer, grup i identycznych obrazów.
Przy treningu weryfikowane są hashe całego snapshotu, a dekodowane wyłącznie
train/val. Zmiana frozen danych przerywa dalszą pracę; nie odświeżamy hashów
ani reviewed automatycznie.

## Protokół zapisany przed treningiem

- SMP U-Net, encoder ResNet50; trzy logity, argmax bez strojenia progów.
  Encoder startuje z oficjalnych wag
  [Torchvision ImageNet1K V2](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html),
  decoder i głowica z losowej inicjalizacji seed=42. Brak wag overfit.
- Plik wag `resnet50-11ad3fa6.pth` pobrano w całości z oficjalnego adresu
  Torchvision; hash SHA256:
  `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`.
  Metadane i URL w `pretrained/pretrained.json`; ładowanie sprawdza hash,
  URL oraz zgodność wag encodera. Nie pobiera zdjęć treningowych ImageNet.
- Natywne rozłączne kafle 256×256: **10 train, 17 val**. Każdy oznaczony piksel
  występuje dokładnie raz w siatce przed augmentacją. Kafle bez etykiet odpadają,
  padding maski=255. Bez skalowania i CLAHE.
- Tylko train: losowy obrót 0/90/180/270°, odbicie poziome, kontrast 0,85–1,15
  i jasność ±0,10. Geometria obrazu i maski jest wspólna, bez interpolacji
  etykiet. RGB/255, mean [.485,.456,.406], std [.229,.224,.225].
- AdamW: encoder LR=0,0001, decoder/głowica LR=0,001, weight_decay=0,0001;
  batch=4, gradient clipping=5. Cały model jest dostrajany, ale statystyki
  BatchNorm encodera zachowują wartości pretrained; affine pozostaje uczone.
- CE z wagami odwrotnie proporcjonalnymi do pierwiastka liczby pikseli klasy
  **wyłącznie UFPR**, plus Dice klas foreground obecnych w batchu.
  `ignore_index=255` wyklucza piksele z obu składników i ich gradientów.
- CE obliczane log-softmax/gather na oznaczonych pikselach jest matematycznie
  równoważne ważonemu CE z ignore. Omija niedeterministyczną redukcję
  `nll_loss2d` CUDA. Test sprawdza wartości, gradienty oraz rzeczywistą CUDA.
- Float32, bez AMP; włączone deterministyczne algorytmy, stały seed i ustawienia
  cuDNN. Wynik bitowy między innymi platformami/wersjami nie jest obietnicą.
- Maksymalnie 100 epok. Najlepszy checkpoint: największe średnie IoU paint/curb
  z łącznej macierzy CNR; przy remisie wcześniejsza epoka. Epoka 0 jest tylko
  pomiarem przed uczeniem. Early stopping: patience=15, min_delta=0,001.
  Checkpoint zachowuje także mniejsze ścisłe poprawy; min_delta steruje
  odliczaniem cierpliwości. Bez przeszukiwania hiperparametrów.

`protocol-before-run.json` zachowuje pierwotny protokół. Pierwsze wykonanie
`unet-resnet50-imagenet-v1/` zatrzymało się na redukcji CUDA przed pierwszym
krokiem optymalizatora. Zachowano konfigurację, epokę 0, log i `aborted.json`.
Po testach równoważności straty przygotowano `protocol-before-run-v2.json` i
`unet-resnet50-imagenet-v2/`, przy tych samych danych, seedzie i parametrach.
Nie wybierano parametrów na podstawie wyniku przerwanej próby.

## Uruchamianie

Z katalogu `/home/karoljaron/Projects/ParkVision` użyj
`research/paint-segmentation/.venv/bin/python`. Środowisko badawcze odtworzono
trwale w `.tools/parkvision-training` w katalogu artefaktów i przywrócono
prowadzący do niego alias `.venv`. Nie instalowano ML w środowisku silnika.
Zależności: `requirements-train.txt`; faktyczne wersje wszystkich pakietów
są zapisane w katalogu wyniku jako `environment.txt`.

Nowy snapshot, z nową nazwą katalogu wyjścia:

```bash
research/paint-segmentation/.venv/bin/python research/paint-segmentation/train.py freeze \
  "/home/karoljaron/Dokumenty/ChatGPT/System parkingowy/paint-segmentation-provenance-2026-11-01/export-reviewed-v1/dataset.json" \
  "/home/karoljaron/Dokumenty/ChatGPT/System parkingowy/paint-segmentation-provenance-2026-11-01/proposed-group-split-manifest.json" \
  --output "/home/karoljaron/Dokumenty/ChatGPT/System parkingowy/paint-segmentation-training-2026-11-01/frozen-NEW"
```

Powtórzenie protokołu na już zamrożonych danych, do nowego katalogu:

```bash
MPLCONFIGDIR=/tmp/parkvision-matplotlib \
research/paint-segmentation/.venv/bin/python research/paint-segmentation/train.py train \
  --frozen "/home/karoljaron/Dokumenty/ChatGPT/System parkingowy/paint-segmentation-training-2026-11-01/frozen-v1" \
  --pretrained "/home/karoljaron/Dokumenty/ChatGPT/System parkingowy/paint-segmentation-training-2026-11-01/pretrained/pretrained.json" \
  --output "/home/karoljaron/Dokumenty/ChatGPT/System parkingowy/paint-segmentation-training-2026-11-01/unet-NEW" \
  --device cuda --epochs 100 --patience 15 --min-delta 0.001 \
  --tile-size 256 --batch-size 4 --seed 42
```

CLI odmawia nadpisania istniejącego snapshotu lub wyniku. Nie ma argumentu
pozwalającego wskazać pilot/overfit jako checkpoint początkowy. `--pretrained`
przyjmuje metadane sprawdzanych oficjalnych wag ImageNet.

## Artefakty i interpretacja

`best.pt` i `last.pt` zawierają parametry modelu, optymalizatora, epokę,
konfigurację, hashe danych, stan early stopping i RNG. Zapis checkpointu
jest atomowy. Po treningu najlepszy checkpoint jest ponownie wczytywany;
drugi świeży model musi dać identyczne predykcje i metryki CNR.

`history.json`, `learning-curves.png`, `config.json` i `result.json` zawierają
przebieg oraz wyniki. Kopie użytego kodu i wersje pakietów pozwalają ustalić
rzeczywiste wykonanie. `validation/` zawiera każdy obraz CNR, jego maskę,
rzeczywistą predykcję, predykcję ograniczoną do ROI, błędy i porównanie.
`validation/tiles/` zawiera **wszystkie 17** kafli walidacyjnych, bez wybierania
najlepiej wyglądających wyników. Kolumny: oryginał / GT użytkownika / predykcja;
background niebieski, paint magenta, curb pomarańczowy. Błędy w osobnym obrazie
są czerwone. Wersje na pełnym kadrze służą orientacji; szczegóły w kaflach 2×.

Metryki wylicza macierz pomyłek (wiersze GT, kolumny predykcji) wyłącznie dla
target !=255. IoU=TP/(TP+FP+FN), Dice=2TP/(2TP+FP+FN), precision=TP/(TP+FP),
recall=TP/(TP+FN). Zerowy mianownik daje null, nie sztuczne 100%.
Raport podaje klasy osobno, łącznie i dla każdego obrazu CNR. Miejsca poza
oznaczonym ROI nie są ani poprawnymi negatywami, ani mierzalnymi błędami.

**To eksperyment na dwóch obrazach jednego parkingu.** Trzy widoki CNR oznaczają
jeden parking walidacyjny, nie trzy niezależne obiekty. Sparse ROI mogą nie
reprezentować całych obrazów, a wcześniejsze uwagi do brzegów GT pozostają
ograniczeniem. Wynik wybranego na CNR checkpointu nie zastępuje niezależnego
testu i nie świadczy o jakości produkcyjnej. PUCPR pozostaje na końcową ocenę
zamrożonego przyszłego modelu, poza tym etapem.
