# AVM SS → UFPR: wykonany eksperyment v1

2026-11-01: zaimportowano cały lokalny AVM SS (**6763** pary), wykonano binarny
pretraining U-Net ResNet50 i fine-tuning na zamrożonym UFPR. CNR pozostało
walidacją, PUCPR zamkniętym testem. Bez zmian silnika CV i bez commita.

**Wynik CNR: foreground mIoU 18,53% → 18,94% (+0,41 p.p.)** na tych samych
20 142 oznaczonych pikselach / 17 wycinkach. IoU paint spadło z 22,28% do 15,18%,
curb wzrosło z 14,77% do 22,70%. Precision paint wzrosło do 89,46%, lecz recall
spadło do 15,46%. To wynik mieszany, bez podstaw do promowania wariantu jako
lepszego modelu farby. Baseline v1 oraz wszystkie wcześniejsze checkpointy zachowano.

[Pełny raport, liczniki, metryki, hashe i protokół](</home/karoljaron/Dokumenty/ChatGPT/System parkingowy/avm-ss-experiment-2026-11-01-v1/README.md>).
[Porównanie wszystkich 17 ROI](</home/karoljaron/Dokumenty/ChatGPT/System parkingowy/avm-ss-experiment-2026-11-01-v1/comparison-v1/all-17-roi-tiles.jpg>).

## Dane i zakres użycia

Źródło: [AVM SS autora](https://github.com/Engro39/avm_dataset), Chulhoon Jang
i Myoungho Sunwoo. Formalna licencja plików danych nieodnaleziona. Dla tego
lokalnego eksperymentu przyjęto jawne udostępnienie autora do treningu i badań
oraz potwierdzony przez użytkownika niekomercyjny projekt studencki.
Nie przypisujemy licencji CC/MIT ani praw do redystrybucji. Obrazy i maski
pozostają lokalnie poza repo; nie zostały opublikowane.

Mapowanie wyłącznie po dokładnym RGB: marker→paint, free_space/vehicle→background;
other/ego/nieznane→ignore. Curb nie występuje w datasetcie AVM. 7071 żółtych pikseli
w 540 maskach jest ignorowanych. Wynikowe maski 0/1/255, wszystkie pliki
zweryfikowane, oryginały zachowane bajt w bajt wraz z hashami i źródłowym YAML.

Wszystkie obrazy AVM mają rolę external_train. Identyfikatory parkingów, kamer
i sesji są nieznane. 103 pary bardzo podobnych sąsiednich klatek przecinają
oryginalny losowy train/test; tego podziału nie użyto do niezależnej walidacji.
Nie można dowieść rozłączności scen AVM i CNR — nie jest to niezależny benchmark.
Zgrubne granice i pominięcia oznaczeń w maskach autora pozostają ograniczeniem.

## Pipeline

- [Importer](/home/karoljaron/Projects/ParkVision/research/paint-segmentation/avm.py): ZIP + JSON pochodzenia → osobne oryginały, binarne maski,
  manifest, statystyki i lock SHA256. Odmawia zapisu do istniejącego katalogu.
- [Pretraining](/home/karoljaron/Projects/ParkVision/research/paint-segmentation/pretrain_avm.py): 2 klasy, ignore=255, ImageNet1K V2,
  3 ustalone epoki, batch=32, 636 kroków. Bez CNR/PUCPR; wybrany last.pt.
- [Fine-tuning](/home/karoljaron/Projects/ParkVision/research/paint-segmentation/train.py): opcjonalne `--transfer-checkpoint` i
  `--transfer-sha256`; encoder i decoder są przenoszone, głowica 3-klasowa nowa.
  Ustawienia baseline zachowane. Early stopping po 31 epokach; najlepsza epoka 16.
- Każdy trening ma własny katalog z kodem, konfiguracją, historią, środowiskiem,
  checkpointami, hashami i rzeczywistymi predykcjami. Katalogi wcześniejsze bez zmian.
- Testy: 86 research z GPU, 755 engine; Black, Flake8, Mypy przechodzą.
  Testy [importera](/home/karoljaron/Projects/ParkVision/research/paint-segmentation/test_avm.py) i [pełnego transferu](/home/karoljaron/Projects/ParkVision/research/paint-segmentation/test_pretrain_avm.py)
  używają sztucznych danych; rzeczywiste AVM sprawdzono osobno w całości.

Szczegóły wznowienia/odtworzenia oraz propozycja kolejnego kontrolowanego wariantu
są w pełnym raporcie. Nie uruchamiano kolejnego treningu po poznaniu wyniku CNR.
