# ParkVision · lokalny panel operatora

Runner zapisuje `snapshot.json` i `events.jsonl`. Adapter Python odczytuje te same pliki i serwuje read-only API oraz aplikację HTML/CSS/JavaScript. Przeglądarka odpytywana co sekundę wyświetla decyzje runnera i wycofuje je po utracie ważności. Nie trzeba instalować pakietów frontendowych ani uruchamiać bazy danych.

## Uruchomienie

W pierwszym terminalu uruchom autoryzowany runner. Przykład odtwarza syntetyczny materiał Stage 2. Plik przetwarzany jest bez czekania na upływ PTS, więc po zakończeniu panel pokaże stan EOF i historię.

```bash
cd /home/karoljaron/Projekty/ParkVision
DOCS='/home/karoljaron/Dokumenty/ChatGPT/System parkingowy'
STAGE2="$DOCS/live-occupancy-mvp-stage2-2026-11-21"
OUTPUT="$DOCS/parkvision-dashboard-local"

research/paint-segmentation/.venv/bin/python -m research.occupancy.live \
  --video "$STAGE2/fixture/view-recovery.mkv" \
  --video-anchor 2026-11-21T09:00:00+00:00 \
  --configuration apps/engine/examples/parking-setup-v1.json \
  --parking-id synthetic-parking --camera-id camera-east --view-confirmed \
  --view-reference "$STAGE2/fixture/view-reference.json" \
  --diagnostic-slot-id slot-02 --session-id local-synthetic-demo \
  --model acpds-rcnn --assets "$DOCS/model-improvement-v2-2026-11-09-v1/assets-rcnn.json" \
  --device auto --sample-hz 1 --view-check-interval 1 --view-valid-seconds 3 \
  --output-dir "$OUTPUT"
```

W drugim terminalu podaj **ten sam** katalog wyników. Uruchomienie adaptera przed runnerem jest bezpieczne. Panel pokaże, że oczekuje na plik.

```bash
cd /home/karoljaron/Projekty/ParkVision
OUTPUT='/home/karoljaron/Dokumenty/ChatGPT/System parkingowy/parkvision-dashboard-local'

apps/engine/.venv/bin/python -m apps.dashboard \
  --output-dir "$OUTPUT" \
  --configuration apps/engine/examples/parking-setup-v1.json \
  --port 8765
```

Otwórz [panel](http://127.0.0.1:8765/) lub [bieżący stan JSON](http://127.0.0.1:8765/api/v1/status). Aby przejrzeć zachowaną klatkę demonstracyjną z potwierdzonym wolnym miejscem, uruchom adapter z katalogiem `live-occupancy-mvp-stage3-2026-11-21/demo-history` w katalogu dokumentów. Ten snapshot pochodzi z rzeczywistego syntetycznego przebiegu Stage 2.

Dla RTSP użyj odpowiednich parametrów runnera, zatwierdzonej geometrii tej kamery i zatwierdzonej referencji widoku. Adapter pozostaje taki sam. Nie kopiuj konfiguracji syntetycznej jako geometrii rzeczywistej kamery. [Instrukcja runnera i Camera Integrity](../engine/docs/live-occupancy.md)

## Stan i ważność

- **Na żywo** oznacza zdrowe źródło RTSP i aktualną zgodność widoku. Miejsce zwiększa licznik wolnych wyłącznie wtedy, gdy runner oznaczył je `confirmed/free`, a publikacja, dowód widoku i dowód miejsca są ważne. Potwierdzone zajęte podlegają tym samym terminom. Przeterminowana obserwacja ma status nieaktualny.
- **Nagranie demo** pokazuje rezultat na osi zapisanego PTS. Liczniki dotyczą tego materiału w chwili wskazanej przez runner, a koniec pliku wycofuje dostępność. Panel nie używa czasu PTS jako daty wygaśnięcia względem czasu bieżącego.
- **Podtrzymane** pokazuje stan `held` z runnera. Nie zwiększa potwierdzonej dostępności. Region diagnostyczny wypada z projekcji i liczników.
- Adapter respektuje ścisłe terminy dowodu z runnera. Dodatkowo snapshot RTSP jest ważny maksymalnie 5 s od publikacji na hoście. Przeglądarka wygasza stan własnym zegarem, również gdy odpowiedź nie nadchodzi. Po błędzie API oczekuje na nowszą obserwację lub nową sesję z własnym aktualnym dowodem.
- Przy utracie sygnału, zmianie bądź braku potwierdzenia widoku panel pokazuje przyczynę i wycofuje dostępność. Po odzyskaniu status pochodzi z nowej sesji runnera. Liczba wymaganych obserwacji i odstępy pozostają w Temporal Occupancy.

Parametr `--configuration` jest opcjonalny. Schemat pokazuje zatwierdzone poligony wyłącznie przy zgodnym parkingu, CameraId, rewizji i SHA konfiguracji oraz dostępnych projekcjach fizycznych ID. W pozostałych przypadkach panel pokazuje karty miejsc. Schemat nie zawiera obrazu z kamery.

## API, diagnostyka i bezpieczeństwo

| Zasób | Odpowiedź |
|---|---|
| `GET /api/v1/status` | Projekcja v1 stanu i efektywnej dostępności, z czasem adaptera i terminami |
| `GET /api/v1/history?limit=24` | Ostatnie zdarzenia w kolejności dziennika, od 1 do 50, w tym zmiany przy sąsiadujących wpisach |
| `GET /api/v1/health` | Stan HTTP adaptera |

Brak pliku snapshot daje 404. Wadliwy JSON, timestamp lub nieobsługiwany schemat daje 422, a błąd odczytu 503. Snapshot Stage 1 bez danych Camera Integrity wymaga nowszego runnera. Dziennik jest czytany od końca, z limitem 4 MiB. Niezapisana końcówka bez znaku nowej linii jest pomijana. Pełny, błędny rekord w odczytywanym ogonie jest zgłaszany jako błąd historii. W razie mieszania starego i nowego schematu zacznij zapis Stage 2+ do nowego katalogu, aby zachować poprzedni dziennik.

Serwer nasłuchuje tylko na `127.0.0.1`, sprawdza nagłówek Host i serwuje zamkniętą listę zasobów. Odpowiedzi nie zawierają klatek, adresów RTSP, ścieżek plików, surowych obiektów modeli ani danych uwierzytelniających. Dostęp zdalny wymaga osobnych zabezpieczeń. Dane wejściowe do runnera muszą mieć potwierdzone uprawnienia do użycia.

## Testy lokalne

```bash
apps/engine/.venv/bin/python -m pytest -q apps/dashboard/tests
node apps/dashboard/tests/freshness.test.mjs
```

Test HTTP wiąże tymczasowy port loopback. W środowisku, które blokuje lokalne gniazda, raportuje pominięcie tego przypadku.
