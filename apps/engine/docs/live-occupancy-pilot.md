# Przygotowanie jednej kamery do pilota Live Occupancy

## Przyjęcie źródła

Operator zapisuje poniższe informacje w lokalnej karcie pilota. Przed
połączeniem potwierdza z właścicielem kamery zakres dostępu i zgody na analizę,
zapis referencji oraz ewentualne nagrania do walidacji.

| Pole | Co zapisać |
| --- | --- |
| ParkingId, CameraId | Trwałe nazwy logiczne i potwierdzone mapowanie na fizyczny widok, np. numer urządzenia i kanał NVR. |
| Źródło | RTSP lub autoryzowany plik testowy, identyfikator źródła bez adresu lub sekretów. |
| Obraz | Odczytane wymiary pikseli, obrót, odbicie, crop, orientacja i profil używanego strumienia. |
| Strumień | Kodek, FPS, bitrate, profil główny/pomocniczy i transport z ustawień kamery/NVR. Odróżnij ustawienia od pomiarów. |
| Czas | UTC odbioru na hoście dla RTSP, kotwica PTS 0 z offsetem dla pliku. Zapisz stan synchronizacji zegara hosta. |
| Stabilność | Ustawienie stałe, stan montażu, sposób zablokowania PTZ, zoomu i automatycznego przekadrowania, jeżeli występują. |
| Uprawnienia i retencja | Decyzja właściciela, data, zakres, osoby z dostępem oraz termin usunięcia klatek, nagrań i metryk. |
| Rewizja pilota | ParkingId, CameraId, rewizja i SHA konfiguracji, SHA manifestu i referencji, decyzja o zakresie fizycznych ROI. |

Dane i wyniki przechowuj lokalnie, poza repozytorium, w katalogu dostępnym
wyłącznie upoważnionym osobom. Kadry i podglądy dopasowań mogą zawierać twarze
lub tablice. Zachowuj tylko potrzebne i autoryzowane pliki, a do raportowania
wykorzystuj metryki lub osobne zanonimizowane kopie. Zachowaj natywne piksele
referencji roboczej. Gdy zgoda na przechowanie wygasa, usuń również kopie
podglądów. Dzienniki nie zawierają pikseli. Panel jest dostępny lokalnie.

## Pobranie referencji

Polecenia uruchamiaj z katalogu głównego repozytorium. Ustaw `PILOT` na nowy
lokalny katalog pilota, `PARKING_ID`, `PARKING_NAME`, `CAMERA_ID` i
`CAMERA_NAME` według karty oraz `REV` na nowy numer rewizji.

```bash
umask 077
mkdir -p "$PILOT"
ENGINE=apps/engine/.venv/bin/python
PY=research/paint-segmentation/.venv/bin/python
IMAGE="$PILOT/reference.png"

read -rs -p 'Adres autoryzowanego RTSP: ' RTSP_SECRET
printf '\n'
PARKVISION_RTSP="$RTSP_SECRET" "$PY" -m research.occupancy.pilot capture \
  --rtsp-env --authorized --camera-id "$CAMERA_ID" --image "$IMAGE"
```

`capture` zapisuje jeden natywny PNG pod nową nazwą, rozmiar i czas raportuje
w JSON. W przypadku pliku zamień `--rtsp-env` na `--video "$VIDEO"
--video-anchor "$ANCHOR"` i pomiń zmienną z adresem. `--authorized` oznacza
faktyczną decyzję operatora. Dla nieczytelnego kadru wybierz kolejny pod nową
nazwą. Jeśli porównujesz kilka ujęć, wybierz referencję świadomie po sprawdzeniu
tła, granic miejsc i orientacji. Wyczyść zmienną sesji poleceniem
`unset RTSP_SECRET` po ostatnim użyciu. Wprowadź adres bez echa terminala
i uruchamiaj te polecenia bez śledzenia powłoki (`set +x`).

## Zatwierdzenie fizycznych miejsc

```bash
REVIEW="$PILOT/roi-r$REV"
CONFIG="$PILOT/parking-r$REV.json"

"$ENGINE" -m parkvision.geometry_setup propose \
  --image "$IMAGE" --output-dir "$REVIEW" \
  --parking-id "$PARKING_ID" --parking-name "$PARKING_NAME" \
  --camera-id "$CAMERA_ID" --camera-name "$CAMERA_NAME" --revision "$REV"

xdg-open "$REVIEW/preview.png"
labelme --nodata "$REVIEW/draft.json"
```

W `preview.png` i `evidence.json` sprawdź pełne i częściowe hipotezy oraz
brakujące granice. W razie potrzeby porównaj wiele ujęć z tym samym widokiem
przez istniejące [propose-sequence](geometry-phase4b.md). W LabelMe skoryguj
krawędzie, ujawnioną fizyczną głębokość, granicę wjazdu i wspólne granice
sąsiadów. Odrzuć przejazdy, kreskowania i niepotwierdzone hipotezy. Dodaj
pominięte miejsca z widocznymi granicami. Ramki detekcji aut opisują chwilową
pozycję pojazdu. Każdy poligon opisuje całe fizyczne stanowisko i otrzymuje
trwały `ParkingSlotId` w polu `label`. Zachowaj ID istniejącego miejsca przy
korekcie i nie używaj ponownie ID usuniętego miejsca. Zacznij od dowolnej
liczby wiarygodnie widocznych miejsc, co najmniej jednego.

Zapisz poprawioną anotację jako `reviewed.json` obok szkicu, z `imageData: null`
i metadanymi `parkvision` szkicu. Skorzystaj z opisu
[formatu i weryfikacji LabelMe](labelme-slot-setup.md).

```bash
"$ENGINE" -m parkvision.labelme_configuration preview \
  --annotation "$REVIEW/reviewed.json" --image "$IMAGE" \
  --output "$REVIEW/reviewed-preview.png"
xdg-open "$REVIEW/reviewed-preview.png"

# Po przeglądzie ustaw REVIEWED_SHA na wypisany SHA anotacji.
# Ustaw OPERATOR i APPROVED_AT na rzeczywiste dane decyzji z offsetem.
"$ENGINE" -m parkvision.labelme_configuration approve \
  --annotation "$REVIEW/reviewed.json" --image "$IMAGE" \
  --reviewed-annotation-sha256 "$REVIEWED_SHA" \
  --approved-by "$OPERATOR" --approved-at "$APPROVED_AT" --output "$CONFIG"

"$ENGINE" -m parkvision.geometry_setup verified \
  --configuration "$CONFIG" --image "$IMAGE" \
  --camera-id "$CAMERA_ID" --output-dir "$REVIEW/verified"
```

Sprawdź finalną nakładkę względem referencji. Zachowaj SHA wybranej referencji
i zaakceptowanej anotacji obok geometrii. Ponowny przegląd i nowa rewizja są
potrzebne po zmianie pozycji, orientacji, obiektywu, cropu lub zatwierdzonych ROI.

## Manifest Camera Integrity i przegląd

Na tym samym obrazie przygotuj dwa rozłączne PNG w skali szarości (8 bit),
o wymiarach referencji. Zaznacz na biało (255) dwa odrębne zestawy stałych
elementów, a resztę ustaw na czarno (0). Rozłóż **każdy zestaw** wokół wszystkich
zatwierdzonych ROI, także na końcach rzędu. Wybieraj teksturowane stałe elementy
widoczne przy różnych warunkach. Wyklucz samochody, ich najbliższe otoczenie,
typowe zasłonięcia oraz ruchome cienie i roślinność. Runtime wyklucza wnętrze
i margines ROI oraz ogranicza pomiar na brzegu masek.

Sprawdź ręcznie podgląd masek względem referencji. Zapisz obok nich
`view-reference.json` według [istniejącego manifestu v1](live-occupancy.md#referencja-i-kontrola-widoku).
Podaj ten sam CameraId, rewizję i wymiary, SHA konfiguracji oraz SHA
referencji i obu masek. `source` opisuje źródło i czas kadru bez adresu RTSP.
`approved_by` i `approved_at` dokumentują decyzję operatora dotyczącą kadru
i masek. Obraz pozostaje tym samym plikiem użytym do przeglądu ROI. Dla
przyjętego zestawu zapisz hashe poza manifestem w karcie pilota.

Preflight porównuje referencję z samą sobą, aby wykryć brak użytecznych punktów
oraz brak przestrzennego pokrycia, po czym osobno sprawdza bieżący kadr
z wybranego źródła. Opcjonalny `--alignment-preview "$PILOT/alignment.png"`
zapisuje podgląd Stage 4C z korespondencjami dopasowania i niewykorzystanymi
w estymacji punktami kontroli. Przejrzyj ich rozkład wokół miejsc, p90 błędu
kontroli, odchylenie wierzchołków ROI i oba pola dotyczące pokrycia.
Podgląd transformacji służy do diagnostyki. Runner nadal pracuje na
natywnym, niewyrównanym obrazie. Jeżeli pokrycie lub dopasowanie się nie broni,
pozostaw wynik `unverifiable` i skoryguj referencję lub maski pod przeglądem.
Przy `mismatch` sprawdź właściwy kanał, rozdzielczość, pozycję i kadrowanie.
Start i reconnect weryfikują aktualny obraz względem zamrożonej referencji.

## Preflight i zamrożenie

Ustaw `VIEW` na plik manifestu, `ASSETS` na lokalny opis zasobów wybranego
backendu, `SESSION` na nową nazwę przebiegu, a `OUTPUT` na nową ścieżkę
wyników wewnątrz `PILOT`. Ustaw `CONFIG_SHA` i `VIEW_SHA` na wartości z
karty pilota, np. odczytane poleceniem `sha256sum "$CONFIG" "$VIEW"` podczas
zapisywania zaakceptowanego zestawu. Przy kolejnych uruchomieniach pobierz
wartości z tej karty. `--view-confirmed` oznacza potwierdzenie mapowania
źródła, ustawień widoku i dopasowania do zakresu ROI przez operatora.

```bash
INPUT=(--rtsp-env)
# Dla autoryzowanego pliku: INPUT=(--video "$VIDEO" --video-anchor "$ANCHOR")
COMMON=(--configuration "$CONFIG" --configuration-revision "$REV"
  --configuration-sha256 "$CONFIG_SHA" --parking-id "$PARKING_ID"
  --camera-id "$CAMERA_ID" --view-confirmed --view-reference "$VIEW"
  --view-reference-sha256 "$VIEW_SHA" --assets "$ASSETS" --model acpds-rcnn
  --device cuda --session-id "$SESSION" --output-dir "$OUTPUT")
# Historyczne regiony diagnostyczne wskaż tylko gdy występują w konfiguracji:
# COMMON+=(--diagnostic-slot-id "$DIAGNOSTIC_SLOT_ID")

PARKVISION_RTSP="$RTSP_SECRET" "$PY" -m research.occupancy.pilot preflight \
  "${INPUT[@]}" "${COMMON[@]}" --authorized --dashboard-port 8765 \
  > "$PILOT/preflight.json"
```

Zamknij poprzedni adapter przed sprawdzeniem planowanego portu lub wybierz
osobny port i użyj go też w instrukcji startu. Preflight używa tymczasowego
zapisu w planowanym katalogu, sprawdza świeżą sesję o stanie `unknown`,
odczyt statusu i historii, stronę oraz health. Pliki próby są sprzątane.
Wymaga pustego katalogu wyników. Zapisz raport preflight z datą w karcie
pilota i potwierdź decyzję po przeglądzie. Karta powinna podać również SHA
referencji i masek, identyfikator zasobów, backend, urządzenie, planowane
próbkowanie i wersję kodu użytego do testu.

| Wynik | Znaczenie i dalsze działanie |
| --- | --- |
| `ready` (0) | Pozytywna kontrola RTSP, zgodności widoku, CUDA, nowej sesji, zapisu i adaptera w chwili sprawdzenia. |
| `conditional` (3) | Sprawdzenie na pliku lub CPU przeszło. Powtórz na autoryzowanym RTSP i docelowym GPU przed próbą terenową. |
| `blocked` (2) | Odczytaj etap, kod przyczyny i działanie operatora. Powtórz sprawdzenie po usunięciu blokady. |

## Start pilota po akceptacji

Uruchom runner z tymi samymi ścieżkami, identyfikatorami i pinami, a adapter
w drugim terminalu z tym samym katalogiem wyników i zatwierdzoną konfiguracją.
Po przekazaniu źródła do runnera wyczyść zmienną z adresem w powłoce.

```bash
PARKVISION_RTSP="$RTSP_SECRET" "$PY" -m research.occupancy.live \
  "${INPUT[@]}" "${COMMON[@]}" --sample-hz 1
unset RTSP_SECRET

# Drugi terminal, z ENGINE, OUTPUT i CONFIG ustawionymi tak samo.
"$ENGINE" -m apps.dashboard --output-dir "$OUTPUT" --configuration "$CONFIG" --port 8765
```

Panel działa pod [127.0.0.1:8765](http://127.0.0.1:8765/). Sprawdź CameraId,
geometrię i nową sesję. Przy `checking`, `mismatch`, `unverifiable`, awarii
lub przeterminowaniu widoku oczekuj braku potwierdzonych wolnych miejsc.
`held/free` jest stanem informacyjnym. Przy Ctrl+C panel odczyta końcowy stan
runnera. Przy kolejnej próbie wybierz nowy katalog dziennika.
