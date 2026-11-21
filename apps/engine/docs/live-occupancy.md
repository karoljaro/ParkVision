# Single-Camera Live Occupancy — Stage 1

Runner obsługuje jeden autoryzowany plik wideo lub strumień RTSP i jedną zatwierdzoną rewizję geometrii. Wykorzystuje istniejący obserwator per-slot, Faster R-CNN, regułę D i maszynę Temporal Occupancy. Wybrany backend i jego wagi ładuje raz na przebieg. Obserwacje z dziennika można podać do istniejącego replay.

## Uruchomienie

Uruchom z głównego katalogu repozytorium w istniejącym środowisku research z PyTorch, torchvision i OpenCV. Przykład używa syntetycznego pliku utworzonego w Stage 1 i gotowej syntetycznej konfiguracji z repozytorium.

```bash
DOCS='/home/karoljaron/Dokumenty/ChatGPT/System parkingowy'
DEMO="$DOCS/live-occupancy-mvp-stage1-2026-11-20"

research/paint-segmentation/.venv/bin/python -m research.occupancy.live \
  --video "$DEMO/synthetic-parking.avi" \
  --video-anchor 2026-11-20T12:00:00+00:00 \
  --configuration apps/engine/examples/parking-setup-v1.json \
  --parking-id synthetic-parking --camera-id camera-east --view-confirmed \
  --diagnostic-slot-id slot-02 --session-id synthetic-demo \
  --model acpds-rcnn \
  --assets "$DOCS/model-improvement-v2-2026-11-09-v1/assets-rcnn.json" \
  --device cpu --sample-hz 1 --output-dir "$(mktemp -d)"
```

Dla `--model convnext-roi4` podaj `assets-convnext.json`. Loadery sprawdzają SHA-256 lokalnych zasobów. `--device cuda` wymaga działającego urządzenia, sterownika i środowiska Torch. Brak CUDA kończy przebieg błędem.

- Dla pliku `--video-anchor` podaje czas absolutny **PTS 0** z offsetem. OpenCV dostarcza kolejne PTS. Przy brakujących, powtarzających się lub cofających PTS przebieg zgłasza błąd. Jego zegar ewaluacji jest zegarem zapisu, więc wolna inferencja nie zmienia historycznych odstępów. Rzeczywisty czas publikacji jest osobnym polem.
- Dla RTSP zastąp `--video` i `--video-anchor` przez `--rtsp 'rtsp://adres/ścieżka'`. Źródło używa backendu FFmpeg OpenCV z otwarciem i odczytem ograniczonymi do około 3 s, a potem ponawia połączenie. Czas klatki to UTC odbioru przez aplikację, z możliwym buforowaniem kamery i dekodera przed odbiorem. Obsługa lokalnego urządzenia kamery jest poza Stage 1.
- `--sample-hz` (domyślnie 1) określa maksymalne tempo wyboru klatek. Plik jest dekodowany kolejno i próbkowany według PTS. RTSP zachowuje najwyżej jedną oczekującą klatkę i ma co najwyżej jedną pracującą inferencję. Starsza oczekująca klatka ustępuje nowszej. Dziennik podaje pominięcia oraz osobno odrzucone wyniki. Opóźniony wynik RTSP w wieku co najmniej 5 s nie jest publikowany jako rozstrzygnięcie. Przy efektywnych przerwach ≥5 s polityka temporalna wymaga ponownego nagromadzenia dowodów.
- `--view-confirmed` to deklaracja operatora o zgodności kamery, kadru i konfiguracji. Każdy skonfigurowany slot musi mieć projekcję wybranej kamery. Przekaż wszystkie regiony diagnostyczne przez powtarzalne `--diagnostic-slot-id`, ponieważ schemat konfiguracji v1 nie zawiera ich roli. Regiony pozostają w historycznej regule przypisywania D, ale wypadają z liczników fizycznych. Opcjonalne `--configuration-revision` i `--configuration-sha256` weryfikują oczekiwaną rewizję i dokładne bajty.
- Konfiguracja jest przypięta na cały przebieg. Niezgodna rozdzielczość unieważnia widok do ponownego uruchomienia z ręcznie potwierdzonym kadrem. Integrator API może też podać kontrolę widoku `view_check`. CLI nie mierzy samoczynnie dryfu tej samej rozdzielczości. Po znanej zmianie kadru zatrzymaj runner i zweryfikuj geometrię przed ponowną deklaracją.

## Co czyta przyszła aplikacja

`snapshot.json` ma `schema_version=1` oraz `kind=parkvision_live_occupancy`. Zawiera parking, kamerę, sesję, zatwierdzoną rewizję i SHA, backend i SHA checkpointu per-slot, status źródła, czasy i listę miejsc. `captured_at` oznacza czas ostatniej klatki, `updated_at` służy do interpretacji stanu temporalnego, a `published_at` to zegar publikacji hosta. Przy pliku `updated_at` należy do osi nagrania. Przyszły klient dostępności na żywo powinien korzystać wyłącznie ze źródeł oznaczonych `kind=rtsp`.

Każdy wiersz miejsca ma `raw_state` i `unknown_reason` ostatniej obserwacji, `estimated_state`, `temporal_status`, `temporal_reason`, wiek ostatniego dowodu oraz znacznik `diagnostic_only`. Stan surowy może być pusty po starcie lub utracie sygnału. `confirmed_available` wymaga fizycznego miejsca z `confirmed/free`. W źródle na żywo czytelnik musi również wymagać, aby jego czas UTC był **wcześniejszy** od `confirmed_evidence_expires_at`. Po przekroczeniu tego czasu przelicza dostępność z wierszy albo wstrzymuje licznik potwierdzonych miejsc do odświeżenia. Ta reguła obowiązuje także po awarii procesu, gdy plik pozostaje na dysku.

`availability_counts` zawiera `physical_total`, `confirmed_free`, `confirmed_occupied`, `held`, `unknown` i `stale`. Pięć ostatnich kategorii jest rozłącznych i sumuje się do `physical_total` w momencie `updated_at`. `held/free` zachowuje informację dla operatora i nie podnosi `confirmed_free`. Unknown i stale zachowują brak rozstrzygnięcia.

`events.jsonl` jest dziennikiem append-only. Każdy wiersz ma wersję, rodzaj zdarzenia, licznik zdarzeń w przebiegu i odpowiadający mu snapshot. Zdarzenie `observation` dodaje dokument observation v1 z surowymi score, przypisaniami i przyczynami, PTS, pominięciami i czasami etapów. SHA obrazu w runnerze opisuje zdekodowane piksele BGR uint8, a istniejące CLI pojedynczej obserwacji nadal raportuje SHA bajtów pliku. Dziennik i snapshot nie zawierają pikseli. Do ponownego replay eksportuj `observation` z jednego efektywnego `source_session` i jednej tożsamości modelu do osobnych plików oraz wskaż je w istniejącym manifeście replay.

## Diagnostyka i zatrzymanie

- `starting` zapisuje unknown przed ładowaniem modelu. EOF pliku kończy przebieg z wycofaną dostępnością. Uszkodzony plik ma kod wyjścia błędu. Brak obrazu RTSP przechodzi w `waiting` lub `interrupted`. Znana utrata sygnału natychmiast unieważnia estimate. Wynik inferencji ze starego połączenia jest odrzucany, a odzyskanie obrazu publikuje stan po resecie i przydziela nową sesję.
- Ctrl+C i SIGTERM kończą przyjmowanie klatek i zapisują stan zatrzymany. Przy blokadzie w natywnej inferencji istnieje ograniczenie czasu zamknięcia procesu. Odbiorca musi przestrzegać terminu ważności potwierdzenia. Restart tworzy nowy `run_id` oraz efektywną sesję i rozpoczyna od unknown. Opcja `--session-id` zachowuje operatorowi rozpoznawalną nazwę przebiegu.
- Katalog wyjścia ma jednego aktywnego pisarza na Linuksie. Atomowa podmiana zapewnia kompletny JSON snapshotu. Dziennik jest synchronizowany przed podmianą, więc po przerwaniu może zawierać jedno zdarzenie więcej. Przy wznowieniu zapis nie używa historycznych stanów. Log należy rotować operacyjnie przy długiej pracy.
- Aktualny runner stosuje zamrożone parametry Temporal Occupancy v1. Jednoczesne strumienie, publikacja HTTP i magazyn długoterminowy są odrębnym zakresem. Skuteczność na nowym CCTV wymaga danych z potwierdzonym uprawnieniem, stabilnym widokiem i niezależnymi etykietami. W Stage 1 RTSP sprawdzono kontrolowanym backendem testowym, a prawdziwy model uruchomiono na pliku syntetycznym.
