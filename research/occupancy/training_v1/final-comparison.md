# Końcowe porównanie v1 — wykonane

2026-11-08. Porównano zamrożony ACPDS RCNN i wybrany wcześniej ConvNeXt
RGB+ROI z epoki 13. Wykonano jeden test, po którym nie było treningu,
kalibracji, zmiany progów ani ponownego wyboru checkpointu.

Wyniki i pełny protokół:

`/home/karoljaron/Dokumenty/ChatGPT/System parkingowy/dataset-model-improvement-2026-11-08-v1/final-comparison-v1/`

Najważniejsze pliki: `Raport-koncowego-porownania.md`, `comparison.json`,
`predictions.jsonl`, `summary.csv`, `comparison-chart.png`, `protocol.json`,
`execution-lock.json` i `frozen/`. Ścieżki oraz SHA-256 wiąże
`final-comparison-reference.json` w tym katalogu kodu.

## Wyniki tej samej reguły D, przy progach 0.1/0.9

Poprawne rozstrzygnięcia liczymy względem wszystkich ROI; unknown nie jest
poprawną klasyfikacją źródłowo znanego stanu.

| Grupa | Poprawne: baseline → nowy | Fałszywe free | Fałszywe occupied | Unknown |
| --- | ---: | ---: | ---: | ---: |
| ACPDS test, 1490 ROI / 27 zdjęć | 97.38% → 97.99% | 14 → 12 | 9 → 5 | 16 → 13 |
| CNR Pisa, 636 ROI / 24 zdjęcia | 87.11% → 95.44% | 10 → 6 | 5 → 8 | 67 → 15 |

Pomocniczo łącznie: 24→18 fałszywych free, 14→13 fałszywych occupied,
83→28 unknown. To wyniki warstwy dowodowej na etykietach źródłowych,
nie ogólna dokładność ParkVision ani potwierdzone decyzje na docelowej kamerze.

Na CNR sam klasyfikator ograniczył fałszywe free 52→15. Niezmieniona D zatrzymuje
9 z nich w nowym modelu, zostawiając 6; model_conflict spadł 42→9.
Recall occupied wzrósł 83.55%→95.61%, recall free spadł 96.11%→95.00%.
Pozostaje więc koszt 3 dodatkowych fałszywych occupied w CNR.

## Widoczność i zakres wniosku

W istniejącym, poprawionym przez operatora podzbiorze ACPDS: oba modele
mają 24/24 poprawnych clear i 12/12 poprawnych partial ze znanym stanem.
Na 4 unreadable baseline daje 3 nieuzasadnione rozstrzygnięcia, nowy model 4.
**Sam softmax nie zastępuje oceny widoczności.** Zewnętrzna bramka pozostaje
konieczna; jej zadeklarowana symulacja „tylko clear” wstrzymuje unreadable
i partial w obu modelach. Nie zmienia to głównych wyników źródłowych.

Obie grupy były historycznie używane w pilotażu starego baseline’u, ale były
wyłączone z train/val i wyboru tego kandydata. Nie nazywamy ich nigdy wcześniej
nieoglądanym testem całego projektu. Nie testowano temporal, bo te zdjęcia
nie tworzą ciągłej osi czasu. PUCPR i T-01 nie weszły do porównania.

## Kod i kontrola

- `final_compare.py`: walidacja protokołu, holdoutów, brak przecieków, wspólne
  RGB i cache detektora, jednorazowa inferencja obu zamrożonych modeli.
- `comparison_metrics.py`: macierze, recall, coverage, konflikty i pary zmian;
  unknown oraz nieuzasadnione decyzje przy ręcznym unknown są liczone osobno.
- `test_final_compare.py`: syntetyczne testy liczników, granic C/D, widoczności,
  niedopuszczania train do testu i ponownie zakodowanych duplikatów.
- Wraz z wcześniejszym kodem **72 testy przechodzą**, Black i Flake8 również.

Oryginalny obserwator, C/D, temporal v1, geometria R1/R2/R3, stare wagi,
dane train/val i ZIP-y pozostają zachowane. Nie wykonano commita ani pushu.

Wykonana komenda (gotowy wynik jest chroniony przed nadpisaniem):

```bash
cd /home/karoljaron/Projekty/ParkVision
CUBLAS_WORKSPACE_CONFIG=:4096:8 research/paint-segmentation/.venv/bin/python \
  -m research.occupancy.training_v1.final_compare \
  --protocol '/home/karoljaron/Dokumenty/ChatGPT/System parkingowy/dataset-model-improvement-2026-11-08-v1/final-comparison-v1/protocol.json' \
  --device cuda
```

`build-report.py` w katalogu artefaktów odtwarza raport z zapisanych predykcji
bez ponownej inferencji. Modelu nie podmieniono automatycznie w `observe.py`.
