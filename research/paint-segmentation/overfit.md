# Techniczny test U-Net na zatwierdzonym pilocie

2026-10-30: wykonany lokalnie, **bez commita i bez zmian silnika**.
To sprawdzenie działania uczenia na tych samych przykładach, nie benchmark.

## Wejście i zakres

Wejście to `draft-roi-v1/export-reviewed-v1/dataset.json` w zewnętrznym pilocie.
Wszystkie 4 bieżące JSON-y mają ręcznie zatwierdzone `flags.reviewed=true`.
Wcześniejsze uwagi do rozmytych brzegów i mokrego fragmentu pozostają w raporcie
eksportu; nie poprawiano etykiet automatycznie. Test nie dowodzi ich poprawności.

Loader reużywa walidację `prepare.py`: sprawdza klasy, split=pilot, źródłowe
SHA256 i ścieżki, reviewed oraz dokładną zgodność maski z bieżącym JSON-em.
Zmiana anotacji po eksporcie wymaga nowej konwersji, nie podmiany samego hasha.
Kontrola przecieku przechodzi; identyfikatory parkingów pozostają unknown,
a wszystkie próbki nadal są w pilot. Nie utworzono train/val/test.

Zachowano 25 rozłącznych wycinków 128×128 z siatki w natywnej skali.
Każdy zatwierdzony piksel występuje dokładnie raz. Puste wycinki odrzucono;
padding etykiet ma wartość 255. Bez skalowania, augmentacji i CLAHE.
Liczby pikseli: background 35 769, paint 1 138, curb 1 204; razem 38 111.
Pozostałe piksele nie uczestniczą w loss ani w kontroli odtworzenia.

## Model i kontrola

- Gotowa implementacja `segmentation_models_pytorch.Unet`, encoder ResNet50,
  3 kanały RGB, 3 klasy wyjściowe, bez dodatkowej aktywacji wyjścia.
- Losowa inicjalizacja (`encoder_weights=None`), bez pobierania checkpointów
  lub zewnętrznych obrazów. To test mechaniki uczenia. Docelowy baseline
  z wagami ImageNet pozostaje osobnym eksperymentem z wcześniejszego planu.
- RGB /255, następnie mean [.485,.456,.406], std [.229,.224,.225].
- Adam, LR=0.001, batch=8, seed=42, limit 600 kroków, kontrola co 25 kroków.
- CE z ignore_index=255, wagami odwrotnie proporcjonalnymi do pierwiastka liczby
  pikseli klasy, oraz Dice dla klas foreground obecnych w batchu. Oba składniki
  wykluczają ignore; test potwierdza także zerowy gradient na ignore.
- Kontrola w `model.eval()` na wszystkich tych samych wycinkach. Kryterium:
  loss ≤20% początkowego oraz precision i recall ≥90% **dla każdej klasy**.
  To kryterium techniczne zapamiętania, nie próg jakości wdrożeniowej.
- Kontrola false positives jest konieczna: pierwszy przebieg zakończony po
  50 krokach miał wysokie recall, ale nadmiernie przewidywał curb na asfalcie.
  Jego wynik zachowano w `unet-overfit-v1`. Docelowy przebieg `v2` kontroluje
  zarówno precision, jak i recall, przy tych samych danych/modelu/LR/seedzie.

## Zapisany wynik

Katalog docelowy poza repo:
`/home/karoljaron/Dokumenty/ChatGPT/System parkingowy/paint-segmentation-pilot-2026-10-30/unet-overfit-v2/`.

Kryterium przeszło po **100 krokach**: loss **3.149596 → 0.111515**.
Wynik nie jest monotoniczny; `history.json` i wykres zachowują również skok
przy kroku 75. Nie ukrywano nieudanych punktów kontroli.

| Klasa | Precision na uczonych pikselach | Recall na uczonych pikselach |
| --- | ---: | ---: |
| background | 0.996934 | 0.999888 |
| paint | 1.000000 | 0.953427 |
| curb | 0.995656 | 0.951827 |

**Tabela opisuje wyłącznie te same piksele, na których model się uczył.**
Nie świadczy o jakości na nowych obrazach, odporności na deszcz, przewadze nad
LSD ani wyborze najlepszego modelu. W checkpoint nie ma integracji z detektorem.

RTX 4070, PyTorch 2.14.1+cu126, torchvision 0.29.1+cu126, SMP 0.6.0,
NumPy 2.5.4, OpenCV 5.0.0. Czas pętli uczenia wraz z okresową kontrolą:
3.091 s; bez instalacji/importów, inicjalizacji, zapisu i podglądów.
Peak allocated CUDA: 775 425 536 B. To zapis pojedynczego przebiegu, nie benchmark.

Checkpoint `unet-resnet50.pt` wczytano przez weights_only=True do nowego modelu;
przewidywane klasy na wszystkich wycinkach są identyczne z modelem przed zapisem.
Hashe wejść i reviewed sprawdzono ponownie; źródeł nie zmieniano.

Artefakty: `config.json`, `tiles.json`, `history.json`, `result.json`, checkpoint,
`training-loss.png`, `training-examples.png`, `previews/` i zamrożone środowisko.
W porównaniach są trzy kolumny: oryginał, etykiety użytkownika, model **tylko
w oznaczonych pikselach**. Poza nimi obraz pozostaje bez kolorów. Nie sugeruje
to, że model umie ignorować nieopisany obszar: surowe `*.prediction.png` zawierają
przewidywanie dla całego wycinka, ale poza GT nie było ono oceniane.

## Odtworzenie w osobnym środowisku

### VS Code

Otwórz `ParkVision.code-workspace` z katalogu głównego repo. Zawiera dwa osobne
foldery: Engine oraz Paint segmentation, każdy z własnym interpreterem `.venv`.
Pylance używa środowiska przypisanego do folderu; Mypy w research jest jawnie
uruchamiane z tego samego środowiska badawczego. Główne ustawienia edytora
w `.vscode/settings.json` nadal dotyczą samego silnika.

W tej sesji `research/paint-segmentation/.venv` jest lokalnym, ignorowanym przez
Git dowiązaniem do `/tmp/parkvision-unet-env`. Jeśli środowisko w `/tmp` zostanie
usunięte, odtwórz je według poniższych poleceń i doinstaluj Mypy w nim przez
`/tmp/parkvision-unet-env/bin/python -m pip install mypy`.
Na nowym checkout można zamiast dowiązania utworzyć zwykłe środowisko `.venv`
bezpośrednio w katalogu research i zainstalować w nim requirements-overfit.txt
oraz Mypy. Nie dodawaj tych zależności do środowiska silnika.

Poniższe polecenia zakładają katalog główny ParkVision i komputer z dostępem
do NVIDIA GPU. Nie instaluj pakietów treningowych do środowiska silnika.

```bash
python3.14 -m venv /tmp/parkvision-unet-env
/tmp/parkvision-unet-env/bin/python -m pip install \
  -r research/paint-segmentation/requirements-overfit.txt
/tmp/parkvision-unet-env/bin/python research/paint-segmentation/overfit.py \
  '/home/karoljaron/Dokumenty/ChatGPT/System parkingowy/paint-segmentation-pilot-2026-10-30/draft-roi-v1/export-reviewed-v1/dataset.json' \
  --output /tmp/parkvision-unet-reproduction --device cuda
```

Wyjście musi być nowym katalogiem. CPU można jawnie wybrać przez `--device cpu`.
Kod wyjścia 0 oznacza przejście kryterium, 2 oznacza koniec limitu bez przejścia,
1 oznacza błąd. Seed i wersje są zapisane; nie gwarantujemy identyczności bitowej
między urządzeniami lub kolejnymi przebiegami CUDA.

```bash
/tmp/parkvision-unet-env/bin/python -m pytest -q research/paint-segmentation
apps/engine/.venv/bin/python -m pytest -q apps/engine/tests research/paint-segmentation
apps/engine/.venv/bin/python -m black --check research/paint-segmentation/overfit.py
apps/engine/.venv/bin/python -m black --check research/paint-segmentation/test_overfit.py
apps/engine/.venv/bin/python -m flake8 --jobs 1 --config apps/engine/.flake8 \
  research/paint-segmentation/overfit.py research/paint-segmentation/test_overfit.py
```

69 testów research przechodzi w środowisku treningowym (52 importera + 17 nowych).
W środowisku silnika: 807 passed, 1 skipped (opcjonalny moduł testów PyTorch).
Black, Flake8 i pip check środowiska treningowego przechodzą.

Źródła API: [SMP U-Net](https://smp.readthedocs.io/en/latest/models.html#unet),
[PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).

Następny etap jakościowy wymaga poprawienia niepewnych etykiet i zebrania większej
liczby niezależnych, opisanych grup obrazów. Ten test potwierdził działający
loader/loss/model/zapis, nie zastępuje właściwego zbioru treningowego i ewaluacji.
