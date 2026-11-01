# Binarna farba — jeden eksperyment AVM → UFPR

2026-11-01: wykonano U-Net ResNet50 z dwoma logitami, pełną inicjalizacją
z binarnego pretrainingu AVM (encoder, decoder, głowica), nowym AdamW i fine-tuningiem
na UFPR. Nie ponawiano pretrainingu. Trening: 20 epok; wybrana epoka 5 według
IoU farby na CNR, bez dobierania progu (argmax dwóch logitów).

Wspólna ocena paint/non-paint na identycznych 20142 oznaczonych pikselach CNR:

| Model | IoU paint % | Dice % | Precision % | Recall % | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| ImageNet baseline | 22,28 | 36,45 | 25,65 | 62,95 | 3305 | 671 |
| AVM encoder+decoder | 15,18 | 26,37 | 89,46 | 15,46 | 33 | 1531 |
| AVM encoder-only | 29,16 | 45,16 | 45,26 | 45,06 | 987 | 995 |
| **Binary paint** | **39,81** | **56,95** | **64,26** | **51,13** | **515** | **885** |

Mokra kamera 3: IoU **62,77%**, precision 100%, recall 62,77%; 118 TP, 0 FP,
70 FN w opisanych ROI. Kamera 8: IoU 52,22%. Kamera 1 z zajętymi miejscami:
IoU 22,42%, nadal słabsze niż baseline. Wyniki nie potwierdzają jakości poza ROI.

Logiczny widok danych mapuje 0→0, 1→1, 2→0, 255→255. Nie zmienia oryginalnych
masek, LabelMe ani manifestów. Curb jest negatywem farby, bez unieważniania geometrii.
Stare predykcje porównano po mapowaniu ich argmax: paint→paint,
background/curb→non-paint. Nie porównywano różnych definicji foreground mIoU.

Przed treningiem sprawdzono SHA256 źródła i zgodność **380 tensorów**, w tym obu
tensorów binarnej głowicy. Wagi klas pochodzą tylko z UFPR: non-paint=47073,
paint=1503. Seed=42, te same augmentacje, normalizacja, kafle 256×256,
batch=4, AdamW, LR, weight decay, clipping i parametry early stopping co wcześniej.
Wyjątkiem jest jawnie ustalone kryterium wyboru: IoU paint.

Wynik pakietu binarnego zadania i pełnej inicjalizacji AVM jest obiecujący na CNR,
lecz jeden eksperyment nie rozdziela wpływu zadania od inicjalizacji i wyboru
checkpointu. CNR pozostaje ograniczoną walidacją, a pochodzenie scen AVM nie
pozwala dowieść jej niezależności. Model nie został automatycznie promowany.
Kolejny etap to zróżnicowane, niezależne dane, bez dalszego strojenia na tych 3 kadrach.

[Pełny raport, metryki per obraz i SHA256](</home/karoljaron/Dokumenty/ChatGPT/System parkingowy/binary-paint-2026-11-01-v1/README.md>) ·
[Wszystkie 17 kafli / 4 modele](</home/karoljaron/Dokumenty/ChatGPT/System parkingowy/binary-paint-2026-11-01-v1/comparison-v1/all-17-roi-tiles.jpg>) ·
[Mapy FP/FN](</home/karoljaron/Dokumenty/ChatGPT/System parkingowy/binary-paint-2026-11-01-v1/comparison-v1/all-17-fp-fn.jpg>).

Mapy P(paint) float32 750×1000 oraz podglądy z kreskowanym ignore są w osobnym
katalogu `ufpr-binary-paint-v1/validation/` pod raportem. Surowe wyjścia poza ROI
są niezweryfikowane; pliki ROI mają NaN/255 poza nadzorem.

Kod: [trening i widok danych](/home/karoljaron/Projects/ParkVision/research/paint-segmentation/binary_paint.py),
[wizualizacje](/home/karoljaron/Projects/ParkVision/research/paint-segmentation/binary_views.py),
[testy](/home/karoljaron/Projects/ParkVision/research/paint-segmentation/test_binary_paint.py).
Dotychczasowy train.py i silnik CV bez zmian. Testy: 97 research z GPU i 755 engine;
Black/Flake8/Mypy przechodzą. PUCPR zamknięte, wcześniejsze artefakty zachowane.
Bez fuzji CV+ML, dodatkowych treningów i commita.
