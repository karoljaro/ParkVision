# Occupancy · pojedyncza obserwacja offline

Moduł badawczy wczytuje zatwierdzoną `ParkingConfiguration` i zdjęcie,
używa istniejących `SlotProjection` i `Frame`, a następnie zapisuje
wersjonowany wynik JSON. Jest to punkt wejścia zamrożonego baseline’u
`occupancy-offline-baseline-v1`.

Model Improvement v2 dodaje jawne `--model convnext-roi4` obok domyślnego
`--model acpds-rcnn`. [Opis, kontrakt i komendy](model-improvement-v2.md)
oraz [osobny workflow widoczności](visibility_v2/README.md).

## Uruchomienie

Korzystaj z istniejącego środowiska research, które ma Torch 2.14.1+cu126,
torchvision 0.29.1+cu126, NumPy i OpenCV. Skrypt odnajduje modele domenowe
w `apps/engine/src`. Model działa domyślnie na CPU, a `--device cuda`
wybiera dostępną kartę. Nie ma pobierania wag w trakcie obserwacji.

Przykład z katalogu repozytorium po ustawieniu zmiennych na rzeczywiste
dane. `capture_time` musi być czasem pozyskania ze strefą. `asset_file`
wskazuje opis lokalnych wag podany poniżej.

```bash
research/paint-segmentation/.venv/bin/python research/occupancy/observe.py \
  --configuration "$configuration_path" --camera-id "$camera_id" \
  --image "$image_path" --captured-at "$capture_time" \
  --session-id "$session_id" --max-age-seconds 60 --view-confirmed \
  --assets "$asset_file" --output "$new_result_path"
```

`--view-confirmed` jest deklaracją operatora, że źródło i ustawienie
widoku odpowiadają konfiguracji. Zgodne wymiary nie dowodzą zgodności
kamery. Przy znanym zasłonięciu użyj powtarzalnego parametru
`--unreadable-slot-id`, aby otrzymać `unknown/occluded` dla tego miejsca.

Domyślnie świeżość klatki jest oceniana względem bieżącego czasu UTC.
W odtworzeniu historycznym ustaw jawnie `--evaluated-at` na czas
odtwarzanej obserwacji. Wynik zapisuje, czy zegar pochodzi z argumentu,
czy z systemu. Granicę świeżości wybiera użytkownik. Czas starszy niż
`--max-age-seconds` daje `unknown/stale_frame`, a czas z przyszłości
daje `unknown/future_frame`. Brak czasu pozyskania daje
`unknown/missing_capture_time` i pozwala sprawdzić wejście plikowe
bez wymyślania timestampu obrazu.

Opis zasobów to lokalny JSON z trzema ścieżkami, bez checkpointów w Git:

```json
{
  "detector_weights": "/lokalna/sciezka/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth",
  "per_slot_weights": "/lokalna/sciezka/RCNN_128_square_gopro.pt",
  "acpds_pooler": "/lokalna/sciezka/models/utils/pooling.py"
}
```

Ścieżki względne są rozwiązywane względem tego pliku. Kod sprawdza pełne
SHA-256 obu checkpointów i oryginalnego poolera ACPDS przed ich użyciem.
Pooler jest ładowany z lokalnej kopii przypiętego kodu autora.
[ACPDS i kod autora, MIT](https://github.com/martin-marek/parking-space-occupancy/tree/17da2c8b33055be8019483260f0efbcbdfda0478)
· [Faster R-CNN COCO_V1](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html)

## Reguły wyniku

Per-slot emituje `occupied` przy score ≥0,90, `free` przy score ≤0,10,
a między tymi granicami `unknown/low_confidence`. Detektor bierze
car/truck/bus/motorcycle ze score ≥0,60. D wymaga, aby najlepszy ROI
pokrywał >15% bboxa z przewagą >8 punktów procentowych nad drugim.
Konflikt `per-slot=free` i `D=occupied` daje `unknown/model_conflict`.
Brak detekcji D sam nie nadaje stanu free.

Wejściowe poligony zachowują konwencję konfiguracji `x*(width-1)`,
`y*(height-1)`. Dla wypukłych wielokątów używane jest to samo obliczenie
pola co w pilotażu. Proste wklęsłe poligony są przycinane do bboxa.
Pooler ACPDS tworzy prostokątny crop według oryginalnego algorytmu, także
dla projekcji mających inną liczbę wierzchołków niż cztery.

Wynik wiąże obserwacje z fizycznymi `slot_id`, kamerą, sesją, sekwencją,
czasem pozyskania, czasem oceny, hashem obrazu i rewizją oraz hashem
konfiguracji. Rejestruje też wersje modeli, środowisko i powody unknown.
`p_occupied` jest surowym score softmax, a `confidence` pozostaje
`null`, ponieważ nie wykonano kalibracji wiarygodności. `visibility`
domyślnie jest `unassessed`. Automatyczna widoczność pozostaje
nierozwiązana, więc `free` ma charakter wyniku eksperymentalnego.
Zakres `observe.py` to nadal pojedyncza klatka. Identyfikatory wspólnych miejsc
z różnych kamer są zachowane. Odrębny etap Temporal Occupancy Detection v1
dodaje agregację kolejnych obserwacji jednego widoku bez zmiany tego kontraktu.

Nieznane ID kamery lub błędna konfiguracja kończy polecenie kodem 2.
Brak klatki, niezgodne wymiary, niepotwierdzony widok albo niepoprawna
świeżość zapisują unknown dla istniejących projekcji tej kamery.
Awaria inferencji unieważnia wyniki całej klatki i kończy polecenie
kodem 1 po zapisaniu JSON z `inference_error`. Poprawny wynik, w tym
jawny unknown, kończy polecenie kodem 0. Publikacja kompletnego JSON
jest atomowa i odmawia zastąpienia istniejącego pliku.

## Sprawdzenie

```bash
research/paint-segmentation/.venv/bin/python -m pytest -q research/occupancy
apps/engine/.venv/bin/python -m black --check research/occupancy
apps/engine/.venv/bin/python -m flake8 --jobs 1 --config apps/engine/.flake8 research/occupancy
```

## Temporal Occupancy Detection v1

`replay.py` czyta manifest kolejnych zapisanych observation JSON i zapisuje
osobny wynik temporalny. Typowana, czysta maszyna stanów znajduje się
w `apps/engine/src/parkvision/domain/temporal_occupancy.py`.
Nie wymaga inferencji, nowych wag ani Torch. Domyślnie oczekuje trzech
kolejnych potwierdzeń przez minimum 2 s; unknown może podtrzymać dawny
stan wyłącznie jako `held` z nieodnawialnym TTL 3 s. Heartbeat obsługuje ciszę.
Parametry i ograniczenia opisuje
[kontrakt temporalny](../../apps/engine/docs/temporal-occupancy.md).

Dla każdego backendu przygotuj osobny manifest. Replay odrzuca zmianę znanego
backendu lub SHA-256 checkpointu klasyfikatora w jednym stanie. Dla starszych
obserwacji zapisany SHA-256 przypiętej wagi pozwala ustalić backend. Brak
metadanych modelu w obserwacji nie kasuje ustalonej tożsamości.

```bash
apps/engine/.venv/bin/python research/occupancy/replay.py \
  --manifest research/occupancy/examples/temporal-v1/manifest.json \
  --slot-id synthetic-slot --output /tmp/parkvision-temporal-v1.json
```

Załączona sekwencja jest w całości syntetyczna: potwierdza zachowanie
kontraktu, nie poprawę jakości na nagraniu. Stan confirmed/occupied nie
oznacza wykrytego postoju ani zdarzenia przyjazdu. Widoczność unassessed
pozostaje ograniczeniem także po agregacji. Źródłowe JSON nie są zmieniane;
przy kolejnym uruchomieniu wybierz nową ścieżkę wyjściową.

## Single-Camera Live Occupancy

`python -m research.occupancy.live` uruchamia jeden plik wideo lub strumień
RTSP. Współdzieli klasyfikację klatki z `observe.py` i stan domenowy z
`replay.py`. W dzienniku zapisuje surową obserwację i aktualizację. Bieżący
snapshot zalicza do dostępności wyłącznie fizyczne `confirmed/free`.
[Instrukcja uruchomienia, format i diagnostyka](../../apps/engine/docs/live-occupancy.md).
