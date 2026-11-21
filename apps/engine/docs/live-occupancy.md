# Single-Camera Live Occupancy

Runner obsługuje jeden autoryzowany plik wideo lub strumień RTSP i jedną zatwierdzoną rewizję geometrii. Używa istniejącego obserwatora per-slot, Faster R-CNN ResNet-50 FPN v2, reguły D i Temporal Occupancy. Ładuje model i detektor raz na przebieg. Wyniki obserwacji można przekazać do istniejącego replay.

## Uruchomienie

Z głównego katalogu repozytorium użyj środowiska research z PyTorch, torchvision i OpenCV. Poniższy przykład działa na syntetycznej konfiguracji i materiale z pakietu Stage 2.

```bash
DOCS='/home/karoljaron/Dokumenty/ChatGPT/System parkingowy'
DEMO="$DOCS/live-occupancy-mvp-stage2-2026-11-21"

research/paint-segmentation/.venv/bin/python -m research.occupancy.live \
  --video "$DEMO/fixture/benchmark.mkv" \
  --video-anchor 2026-11-21T08:00:00+00:00 \
  --configuration apps/engine/examples/parking-setup-v1.json \
  --parking-id synthetic-parking --camera-id camera-east --view-confirmed \
  --view-reference "$DEMO/fixture/view-reference.json" \
  --diagnostic-slot-id slot-02 --session-id synthetic-demo \
  --model acpds-rcnn \
  --assets "$DOCS/model-improvement-v2-2026-11-09-v1/assets-rcnn.json" \
  --device auto --sample-hz 1 --output-dir "$(mktemp -d)"
```

- Dla `--model convnext-roi4` podaj `assets-convnext.json`. Loadery weryfikują SHA-256 lokalnych zasobów. `cpu` wymusza CPU i pozostaje wartością domyślną. `cuda` wymaga dostępu do GPU i działającego CUDA NMS w torchvision. `auto` sprawdza oba warunki i zapisuje wybrane urządzenie oraz powód decyzji. Błąd późniejszego ładowania lub inferencji pozostaje błędem tego przebiegu. Pochodzenie modelu zawiera urządzenia jego komponentów.
- Dla pliku `--video-anchor` podaje czas absolutny **PTS 0** z offsetem. Brakujące, powtarzające się albo cofające PTS przerywają przebieg. Czas ewaluacji pochodzi z osi nagrania. Opóźnienie wykonania na hoście jest mierzone osobno.
- Dla RTSP zastąp `--video` i `--video-anchor` przez `--rtsp 'rtsp://adres/ścieżka'`. Źródło używa FFmpeg w OpenCV z limitami otwarcia i odczytu około 3 s i ponawia połączenie. Czas klatki to UTC odbioru przez aplikację. Kamera i dekoder mogą ją wcześniej buforować.
- `--sample-hz` (domyślnie 1) określa maksymalne tempo wyboru klatek. Plik jest dekodowany kolejno i próbkowany według PTS. RTSP przechowuje najwyżej jedną oczekującą klatkę i przetwarza najwyżej jedną klatkę naraz. Nowsza klatka zastępuje starszą oczekującą. Wynik RTSP w wieku co najmniej 5 s traci rozstrzygnięcie. Dłuższa przerwa wymaga ponownego zebrania dowodów.
- Podaj wszystkie regiony diagnostyczne przez powtarzalne `--diagnostic-slot-id`. Schemat ParkingConfiguration v1 nie zawiera ich roli. Regiony uczestniczą w historycznej regule D i wypadają z liczników fizycznych. Opcjonalne `--configuration-revision` i `--configuration-sha256` sprawdzają oczekiwaną rewizję i bajty konfiguracji.

## Referencja i kontrola widoku

Operator wybiera obraz w natywnej rozdzielczości i potwierdza, że zatwierdzone ROI odpowiadają na nim fizycznym stanowiskom. Następnie tworzy dwie rozłączne binarne maski PNG o rozmiarze obrazu. Białe obszary zaznaczają nieruchome elementy widoczne po obu końcach i stronach ROI, np. fragmenty krawężników, ścian i trwałych znaków. Jedna maska służy do estymacji, druga do niezależnej kontroli. Samochody i miejsca, w których regularnie zasłaniają tło, należy wyłączyć. Punkty pochodzące z wnętrza ROI i jego marginesu są dodatkowo wykluczane w kodzie.

Zapisz manifest JSON obok plików. Uzupełnij SHA-256 każdego pliku i niezmienionej konfiguracji, np. poleceniem `sha256sum`. `approved_by` i `approved_at` dokumentują rzeczywistą decyzję operatora.

```json
{
  "schema_version": 1,
  "camera_id": "camera-east",
  "configuration_sha256": "<sha256 konfiguracji>",
  "configuration_revision": 1,
  "image_size": [1280, 720],
  "source": "<pochodzenie wybranego obrazu>",
  "approved_by": "<operator>",
  "approved_at": "<data i czas ISO 8601 z offsetem>",
  "reference": {"path": "reference.png", "sha256": "<sha256>"},
  "fit_mask": {"path": "fit.png", "sha256": "<sha256>"},
  "check_mask": {"path": "check.png", "sha256": "<sha256>"}
}
```

Flaga `--view-confirmed` pozostaje częścią CLI. Do potwierdzenia dostępności wymagany jest również manifest `--view-reference` i pozytywny wynik sprawdzenia bieżącego obrazu. Bez manifestu runner publikuje stan `unverifiable` bez wnioskowania occupancy. Pierwsza klatka sesji jest porównywana z referencją. Aktualnie działający strumień również podlega kontroli.

Rejestracja Phase 4C wykorzystuje korespondencje w obu maskach i zachowuje własne progi. Każdy zestaw musi mieć wystarczające pokrycie przestrzenne wszystkich ROI. Runner stosuje natywne współrzędne i akceptuje przesunięcie wierzchołków do 1,5 px, jeśli dopasowanie i niezależna kontrola je potwierdzają. Większa potwierdzona zmiana daje `mismatch`, a brak wiarygodnych danych daje `unverifiable`. Runner pozostawia zatwierdzoną geometrię bez zmian.

`--view-check-interval` (5 s) steruje harmonogramem. `--view-valid-seconds` (10 s) jest dłuższym terminem ważności dowodu. Pierwsza i odzyskiwana sesja czeka na wynik w stanie `checking`. Podczas kontroli okresowej ostatni dowód zachowuje ważność do wygaśnięcia lub uzyskania negatywnego wyniku. Zmiana wymiarów, utrata widoku, przeterminowanie lub utrata sygnału wycofują dostępność. Odzyskanie uruchamia nową sesję temporalną. Potwierdzenie miejsca wymaga ponownie trzech obserwacji w odstępach co najmniej 2 s dla pierwszego i ostatniego dowodu oraz przerw ściśle poniżej 5 s. `held` ma TTL 3 s.

## Snapshot i dziennik

`snapshot.json` zachowuje `schema_version=1` i `kind=parkvision_live_occupancy`. Rozszerzone pola obejmują `camera_integrity` ze statusem, przyczyną, terminem ważności i diagnostyką dopasowania, `model.device`, `source.health` i liczniki źródła. `frame_timing` zawiera czas odbioru RTSP, osobne czasy przetwarzania i inferencji oraz wiek w chwili publikacji. `exposure_at` pozostaje pusty, gdy źródło nie dostarcza czasu ekspozycji. `source_pts_seconds` odnosi się do pliku. `published_at` jest zegarem hosta, a `updated_at` dla pliku należy do osi nagrania.

Klient dostępności na żywo korzysta wyłącznie z `source.kind=rtsp`. Dla miejsca fizycznego wymaga `confirmed_available=true`, `source.health=streaming`, `camera_integrity.status=verified` i bieżącego UTC **przed** `confirmed_evidence_expires_at` oraz `camera_integrity.valid_until`. Po upływie terminu przelicza dostępność z wierszy lub wstrzymuje jej prezentację do odświeżenia. Dotyczy to także nagłego zatrzymania procesu, gdy stary plik pozostaje na dysku. `held/free`, `unknown` i `stale` nie zwiększają licznika wolnych miejsc. Identyfikatory fizyczne pozostają stabilne.

`events.jsonl` jest dziennikiem append-only ze snapshotem w każdym wpisie. Zdarzenia źródła i widoku zapisują stan bieżący. Zdarzenie `observation` dodaje dokument v1 z surowymi wynikami, stanami, przyczynami, PTS, pominięciami i czasami etapów. SHA obrazu w runnerze opisuje zdekodowane piksele BGR uint8. CLI pojedynczej obserwacji nadal raportuje SHA bajtów pliku. Dziennik i snapshot nie zawierają pikseli. Do replay eksportuj `observation` dla jednej efektywnej sesji i jednej tożsamości modelu, używając istniejącego manifestu replay.

## Diagnostyka i zatrzymanie

- `starting` zapisuje stan unknown przed ładowaniem modelu. EOF pliku wycofuje dostępność. Utrata sygnału RTSP unieważnia widok, a wynik ze starego połączenia jest odrzucany. Po odzyskaniu sygnału runner weryfikuje widok od nowa.
- Ctrl+C i SIGTERM zapisują stan zatrzymany. Restart nadaje nowy `run_id` i efektywną sesję. Opcja `--session-id` zachowuje nazwę rozpoznawalną przez operatora.
- Linuksowy katalog wyjścia ma jednego aktywnego pisarza. Atomowa podmiana zapewnia kompletny JSON snapshotu. Dziennik jest synchronizowany przed podmianą, więc może zawierać dodatkowe zdarzenie po przerwaniu. Przy wznowieniu runner nie wczytuje historycznych stanów. Dziennik wymaga rotacji operacyjnej w długim przebiegu.

## Benchmark

`--profile` zapisuje pomiary poszczególnych etapów dla skończonego wideo. Detektor, klasyfikacja i całkowita inferencja synchronizują CUDA na granicach pomiaru. Profil pliku mierzy wydajność wykonania wsadowego. Opóźnienie RTSP od odbioru do publikacji można odczytać z jego snapshotu w działającej instalacji.

```bash
nvidia-smi
bash "$DEMO/benchmark.sh" cpu "$DEMO/local-cpu"
bash "$DEMO/benchmark.sh" cuda "$DEMO/local-cuda"
research/paint-segmentation/.venv/bin/python "$DEMO/summarize_benchmark.py" \
  --cpu "$DEMO/local-cpu" --cuda "$DEMO/local-cuda" \
  --output "$DEMO/local-benchmark-summary.json"
```

Skrypt wykorzystuje dziesięć identycznych syntetycznych klatek na każdy backend i urządzenie. Pierwsze dwie stanowią rozgrzewkę. Zapisuje czasy ładowania, p50/p90/p95/średnią etapów na ośmiu pozostałych klatkach, zmienność, efektywny FPS, szczytowe RSS i pamięć CUDA alokatora PyTorch. Pomiar okresowej rejestracji występuje tylko na klatkach, na których kontrola jest zaplanowana.
