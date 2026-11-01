# AVM SS — kontrolowany transfer wyłącznie encodera

2026-11-01: wykonano jeden zaplanowany fine-tuning U-Net ResNet50 z encoderem
z zachowanego checkpointu AVM. Decoder i trójklasowa głowica są nowe.
Nie powtarzano pretrainingu AVM ani nie wykonywano dodatkowych prób strojenia.

| Model | Foreground mIoU % | IoU paint % | Precision paint % | Recall paint % | paint→curb piksele |
|---|---:|---:|---:|---:|---:|
| Baseline v1 | 18,53 | 22,28 | 25,65 | 62,95 | 516 |
| AVM v1 encoder+decoder | 18,94 | 15,18 | 89,46 | 15,46 | 1094 |
| AVM encoder only | 20,12 | 29,16 | 45,26 | 45,06 | 664 |

Ten sam zamrożony Dataset A, seed=42, UFPR train / CNR validation / PUCPR test.
Ta sama augmentacja, loss, wagi klas, optimizer, LR i early stopping.
Trening zatrzymał się po 37 epokach; wybrano checkpoint z epoki 22 według
dotychczasowej reguły foreground mIoU CNR. Nie wybierano checkpointu pod paint.

Przed treningiem sprawdzono 318 tensorów encodera względem źródłowego AVM.
60 tensorów decodera i 2 głowicy pozostały zgodne ze świeżym modelem seed=42.
Każdy tensor ma audit SHA256 wartości początkowych, źródłowych i wynikowych.
Rzeczywisty przebieg odmawia startu, jeżeli audyt różni się od preflight.
Nowa inicjalizacja obejmuje również stałe wartości startowe biasów i BatchNorm.

Wynik jest mieszany: IoU farby wzrosło, ale recall pozostało niższe niż baseline.
IoU curb spadło do 11,07%. Na mokrej kamerze 3 IoU paint wynosi **0,43%**
(baseline 35,37%). Nie ma podstaw do uogólniania poprawy na mokre warunki.
CNR to ograniczona walidacja używana do wyboru checkpointu; pochodzenie scen AVM
pozostaje nieznane. To nie jest niezależny benchmark.

[Pełny raport, konfiguracja i hashe](</home/karoljaron/Dokumenty/ChatGPT/System parkingowy/avm-ss-encoder-only-2026-11-01-v1/README.md>) ·
[Audyt inicjalizacji](</home/karoljaron/Dokumenty/ChatGPT/System parkingowy/avm-ss-encoder-only-2026-11-01-v1/initialization-preflight.json>) ·
[Porównanie wszystkich 17 ROI](</home/karoljaron/Dokumenty/ChatGPT/System parkingowy/avm-ss-encoder-only-2026-11-01-v1/comparison-v1/all-17-roi-tiles.jpg>).

Nowe checkpointy są w osobnym katalogu `ufpr-avm-encoder-only-v1` pod raportem.
SHA256 best.pt: `b26fb02e0dab2f44fda8daa5e1f630b12d020e11e7d907be6ef4cfacf71a0883`.
Wszystkie wcześniejsze artefakty, maski i checkpointy zachowano bez zmian.
PUCPR bez inferencji, oceny i strojenia. Bez zmian silnika CV i bez commita.

Kod [train.py](/home/karoljaron/Projects/ParkVision/research/paint-segmentation/train.py)
obsługuje `--transfer-scope encoder` i `--expected-transfer-audit`.
Domyślny `encoder_decoder` zachowuje dotychczasowy zakres transferu.
Testy: 90 research z GPU i 755 engine; Black/Flake8/Mypy przechodzą.
