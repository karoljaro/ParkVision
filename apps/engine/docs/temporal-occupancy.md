# Temporal Occupancy Detection v1

Etap odrębny od ukończonej obserwacji offline. Agregator stabilizuje kolejne
decyzje jednego widoku i miejsca. `confirmed/occupied` oznacza powtarzalny
dowód zajętości, **nie potwierdzony postój, przyjazd ani odjazd**.

## Architektura i zakres

`parkvision.domain.temporal_occupancy` to typowana, niemutowalna maszyna
stanów bez IO, zegara systemowego, modeli i dodatkowych zależności.
Funkcje przyjmują poprzedni stan, zdarzenie/czas i stałą dla przebiegu politykę,
a zwracają nowy stan. Kod nie otrzymuje poligonów ani obrazów.

`research/occupancy/replay.py` adaptuje istniejące observation JSON v1
z `observe.py`. Nie przelicza score, C/D ani konfliktu i nie ładuje wag.
Ponownie wykorzystuje atomowy, wyłączny zapis `observe.save_result`;
CLI wymaga istniejących zależności engine (NumPy/OpenCV), ale nie Torch.
`observe.py`, `models.py`, `rules.py` oraz kontrakt offline pozostają bez zmian.

Jeden stan jest związany z pełnym `StreamKey`:

```text
parking_id / camera_id / slot_id /
configuration_revision / configuration_sha256 / session_id
```

`session_id` pochodzi z offline `source_session`. Zmieszanie któregokolwiek
pola jest błędem. Zmianę sesji, rewizji lub hasha konfiguracji można obsłużyć
wyłącznie jawnym `reset_stream`. Reset usuwa całą historię dowodową i liczniki,
pozwalając zacząć sekwencję od zera. Parking/kamera/miejsce muszą pozostać te
same; inny widok potrzebuje odrębnego stanu. Nie ma fuzji kamer.

Geometria jest zatwierdzonym rzutem miejsca, a nie obrysem dachu samochodu.
Liście, cień, zasłonięcie linii i drobne przesunięcie ROI nie uruchamiają
rekonfiguracji. R1, PUCPR, checkpointy i reguły offline nie są zmieniane.

## Kontrakt stanu

| Pole | Znaczenie |
| --- | --- |
| `last_observation` | Ostatnia przyjęta surowa obserwacja; może być unknown lub nieważna. Nie obejmuje zignorowanych, spóźnionych wejść. |
| `estimated_state` | Aktualne oszacowanie occupied/free/unknown, niezależne od stanu surowego. |
| `status` | confirmed, held, unknown lub stale. Tylko confirmed/held mogą mieć rozstrzygnięte oszacowanie. |
| `reason` | Powód bieżącego stanu, podtrzymania lub braku rozstrzygnięcia. |
| `updated_at` | Czas ostatniego przetworzenia/heartbeat; także przy zignorowaniu duplikatu. Nie odświeża dowodu. |
| `state_changed_at` | Czas ostatniej zmiany `estimated_state`, nie każdej zmiany statusu. Null przed pierwszym rozstrzygnięciem. |
| `last_confirmation_at` | Czas ostatniego uzyskania potwierdzonego oszacowania po pełnym zbieraniu dowodów. Pozostaje historyczny po wygaśnięciu. |
| `last_evidence` | Ostatni świeży, jednoznaczny i dopuszczony dowód; może dotyczyć dopiero kandydata przeciwnego stanu. |
| `supporting_evidence` | Ostatni dowód popierający ostatnie potwierdzone oszacowanie. Przeciwne wejście nie odświeża go. Po unknown pozostaje wyłącznie historią. |
| `candidate` | Kierunek, liczba kolejnych potwierdzeń i początek ich czasu pozyskania. |
| `hold_since` | Początek podtrzymania. Powtórne unknown i przeciwne wejścia nie przesuwają tej granicy. |
| `last_capture_at` | Watermark czasu ostatniego przyjętego wejścia z poprawnym, świeżym czasem; może dotyczyć unknown. Nie jest dowodem zajętości. |

Surowa obserwacja zachowuje capture/evaluation time, sekwencję, widoczność,
status klatki, powód unknown i identyfikator źródła. W replay identyfikatorem
źródła jest SHA-256 **całego** oryginalnego JSON, więc score, modele,
hashe obrazu i konfiguracji można sprawdzić w niezmienionym źródle.
Wynik replay przechowuje również hash manifestu i pełną politykę.
Hash pozwala sprawdzać integralność; nie uwierzytelnia operatora lub kamery.

## Polityka początkowa

Wszystkie wartości są jawne i konfigurowalne przez `TemporalPolicy` lub
obiekt `policy` manifestu. To **heurystyki startowe dla około 1 obserwacji/s**,
bez dopasowania do 114 rozproszonych klatek ani deklaracji poprawy jakości.

| Parametr | Domyślnie | Uzasadnienie i skutek |
| --- | --- | --- |
| `occupied_confirmations` | 3 | Pojedynczy FP nie przełącza stanu; minimum dozwolone przez API to 2. |
| `free_confirmations` | 3 | Symetryczny punkt startowy, bez założenia o znanej częstości FN. Można badać większą wartość. |
| `occupied_min_seconds` | 2 | Potwierdzenia muszą obejmować także czas, nie tylko szybki burst klatek. |
| `free_min_seconds` | 2 | Analogiczna minimalna rozpiętość dowodów free. |
| `max_observation_age_seconds` | 10 | Limit wieku wejściowego dowodu względem jawnego zegara przetwarzania; również limit wieku dowodu wspierającego stan. |
| `max_gap_seconds` | 5 | Przerwa w danych kasuje ciągłość i kandydaturę. Wykrywana również przez heartbeat. |
| `unknown_hold_seconds` | 3 | Krótki, nieodnawialny TTL podtrzymania. Zero wyłącza podtrzymanie. |

Liczba **i** minimalna rozpiętość czasu muszą być spełnione jednocześnie.
To nie jest głosowanie w większości okna. Unknown przerywa serię kandydata;
przeciwna decyzja rozpoczyna nową serię. Po upływie TTL potrzebna jest
ponowna pełna seria, nawet jeśli kolejna klatka pasuje do dawnego stanu.
Jeśli trwa seria przeciwnego stanu, wygaśnięcie podtrzymania nie kasuje jej;
kasują ją unknown, przerwa, stary dowód, reset lub rozłączenie.

Wszystkie terminy są wyłączne: `wiek >= limit` oznacza wygaśnięcie.
Także równość `gap == max_gap` przerywa ciągłość. Ten temporalny kontrakt
jest odrębny od niezmienionej kontroli świeżości w komendzie offline.

## Unknown, cisza i błędy

- Unknown/low_confidence, model_conflict, occluded, brak klatki, błąd
  inferencji i brak wiersza znanego miejsca mogą krótkotrwale podtrzymać
  uprzednio potwierdzony stan, zawsze jako `held`. Nie mogą utworzyć free.
- Początek TTL to capture time pierwszej niepewności lub przeciwnej decyzji.
  Następne unknown nie przedłużają TTL. Dodatkowo ograniczają go wiek
  `supporting_evidence` i przerwa od ostatniego poprawnego capture time.
- `tick` oznacza brak nowej obserwacji. Pokazuje `held/no_observation`;
  TTL ciszy liczy od ostatniego capture time, a nie od spóźnionego ticka.
  Użytkownik API musi wywoływać heartbeat również podczas awarii kamery.
  Zapisany snapshot jest stanem na `updated_at`, nie obietnicą aktualności
  bez kolejnych wywołań zegara.
- Po wygaśnięciu podtrzymania wynik to `unknown/hold_expired`. Po
  przekroczeniu wieku/gap status to `stale`, a oszacowanie zawsze unknown.
  Świeże unknown nadal daje unknown; stary rozstrzygnięty dowód nie wraca.
- Znane rozłączenie (`disconnect`), niepotwierdzony widok, złe wymiary,
  brak capture time oraz przyszła/stara klatka unieważniają stan natychmiast,
  bez held. Następne poprawne obserwacje muszą zebrać pełną serię.
- `visibility=unassessed` jest dopuszczane zgodnie z baseline’em i **nie
  oznacza potwierdzonej widoczności**. Jawne partial/unreadable/
  operator_unreadable daje skuteczne unknown. Surowa decyzja pozostaje
  zapisana. Drzewa, liście, dalekie ROI i skraj kadru nie są automatycznie
  rozpoznawane; jeśli obserwator oznaczy je unknown, agregator nie zamieni
  ich w wolne miejsce. Powtarzalny błędny wynik modelu może się potwierdzić.
- Opcjonalne `motion=moving` jest zewnętrznym veto na dowód obu stanów
  i daje unknown/motion_detected. Brak sygnału ruchu ani `stationary` nie
  dowodzą postoju. Agregator nie oblicza optical flow ani zdarzeń parkingowych.

Sekwencje i czasy capture muszą rosnąć ściśle w obrębie sesji. Przy kolejnych
wywołaniach `observe.py` trzeba jawnie zwiększać `--sequence` (jego istniejąca
wartość domyślna to 0). Identyczne powtórzenie ostatniego numeru sekwencji
jest ignorowane jako duplicate; kopia może mieć
inny source hash/evaluation time. Zmieniona decyzja lub metadane pod tym
samym numerem to błąd, a nie drugi głos. Starsza sekwencja albo nierosnący
capture time to out_of_order: wejście jest zapisane w śladzie replay,
ale nie odświeża dowodu ani liczników. Zegar przetwarzania musi być
niemalejący i ze strefą; jego cofnięcie jest błędem. Nie ma cichego sortowania.

## Replay istniejących JSON

Manifest używa ścieżek względnych wobec własnego katalogu. Dla każdego
zdarzenia `observation` czas przetwarzania pochodzi domyślnie z zapisanego
`evaluated_at`. Opcjonalne `at` jawnie modeluje późne dostarczenie.
Nie ma odwołania do aktualnego czasu systemu. Nie należy nadawać
rozproszonym fotografiom fikcyjnych czasów po to, aby uzyskać serię.

```json
{
  "schema_version": 1,
  "policy": {"free_confirmations": 4, "unknown_hold_seconds": 3},
  "events": [
    {"observation": "observation-000.json"},
    {"observation": "observation-001.json"},
    {"observation": "observation-002.json", "motion": "unassessed"},
    {"tick_at": "2026-11-05T12:00:04+00:00"},
    {"disconnect_at": "2026-11-05T12:00:05+00:00"},
    {"observation": "new-session-000.json", "reset": true}
  ]
}
```

Pierwszy plik (także po resecie) musi zawierać wybrane slot_id. Późniejszy
brak tego wiersza daje unknown/missing_slot_observation. Inny slot wymaga
osobnego replay; nie jest tworzony przez zgadywanie ID. `reset: true` może
zmienić wyłącznie sesję/rewizję/hash dla tego samego parkingu/kamery/miejsca.

Uruchomienie demonstracji **syntetycznej**, z katalogu repozytorium:

```bash
apps/engine/.venv/bin/python research/occupancy/replay.py \
  --manifest research/occupancy/examples/temporal-v1/manifest.json \
  --slot-id synthetic-slot --output /tmp/parkvision-temporal-v1.json
```

Wybierz nową ścieżkę wyjściową przy ponownym uruchomieniu. Zapis atomowy
odmawia nadpisania istniejącego pliku. Błąd dowolnego zdarzenia kończy CLI
kodem 2 bez częściowego wyniku. Poprawny replay, również z unknown/stale,
kończy się kodem 0. Wynik `experimental_temporal_occupancy`, schema v1,
`aggregator_id=temporal-occupancy-v1` zawiera `snapshots` w kolejności wejść:
źródło/hash, osobny `raw_observation`, disposition i stan `temporal`.

Fixture ma deklaratywne, wymyślone decyzje; nie uruchamiano na nim inferencji,
nie reprezentuje prawdziwej konfiguracji ani oznakowanego klipu. Pokazuje
potwierdzenie occupied, pojedyncze przeciwne wejście, odzyskanie stanu,
konflikt/occlusion/liście/skraj kadru, koniec TTL, potwierdzenie free,
ciszę/stale i rozłączenie. To demonstracja kontraktu, a nie pomiar FP/FN.

## Testy i następny eksperyment

```bash
apps/engine/.venv/bin/python -m pytest -q apps/engine/tests research/occupancy
apps/engine/.venv/bin/python -m flake8 --jobs 1 --config apps/engine/.flake8 \
  apps/engine/src/parkvision/domain/temporal_occupancy.py \
  apps/engine/tests/domain/test_temporal_occupancy.py research/occupancy
```

Black sprawdza te same pliki (w środowisku bez obsługi jego równoległego
wykonania można wywołać `python -m black --check` osobno na każdym pliku).
Testy jednostkowe obejmują oba stany, serię, FP/FN, unknown, ograniczoną
widoczność, ruch, TTL, wiek, gap, reset, duplikaty, spóźnienia i zegary.
Testy integracyjne generują prawdziwy kontrakt `observe()` z modelem testowym,
sprawdzają replay, integralność źródeł i deterministyczny/wyłączny CLI.
Wagi ani realna inferencja nie są potrzebne w tym etapie.

Weryfikacja 5 listopada 2026: **994 testy przeszły** (878 dotychczasowych
engine, 33 dotychczasowe offline i 83 nowe temporal/replay). Black sprawdził
8 plików Pythona; Flake8 przeszedł dla nowej domeny, testów i całego
`research/occupancy`. Rzeczywisty CLI odtworzył 12 syntetycznych obserwacji
i 4 zdarzenia zegara/połączenia. Dwa uruchomienia tego samego manifestu są
porównywane bajtowo w teście integracyjnym.

| Czas syntetyczny [s] | Wejście | Oszacowanie / status |
| --- | --- | --- |
| 0–1 | occupied | unknown / oczekiwanie na potwierdzenie |
| 2 | trzecie occupied | occupied / confirmed |
| 3 | pojedyncze free | occupied / held |
| 4 | occupied | occupied / confirmed |
| 5–7 | konflikt, occlusion, liście/cień | occupied / held, stały początek TTL |
| 8 | unknown, skraj kadru | unknown / unknown, koniec podtrzymania |
| 9–10 | free | unknown / oczekiwanie na potwierdzenie |
| 11 | trzecie free | free / confirmed |
| 12 | heartbeat bez klatki | free / held |
| 14 | heartbeat | unknown / unknown, koniec TTL ciszy |
| 16 | heartbeat | unknown / stale, observation_gap |
| 17 | rozłączenie | unknown / unknown, camera_disconnected |

Hashe `observe.py`, `models.py`, `rules.py` i `test_observe.py` pozostały
identyczne jak w raporcie offline z 4 listopada. Nie wykonano treningu,
nowej inferencji ani oceny na PUCPR i nie pobierano datasetów.

Następny eksperyment: odrębny ciągły klip 15–30 minut ze stałej docelowej
kamery, zatwierdzona geometria i rzeczywiste capture time, replay np. 1 Hz.
Ręcznie oznaczyć przedziały occupied/free/unknown i widoczność kilku miejsc,
w tym za drzewem, daleko/skraj kadru i miejsce z częściowo widocznym autem.
Zaznaczyć także ruch, przerwy i niepewne granice czasowe. Zachować nieocenianą
część klipu do kontroli po wyborze parametrów. Porównać raw i temporal:
liczbę przełączeń, opóźnienie uzyskania oszacowania, czas błędnego free,
pokrycie unknown/stale i czas held; nie utożsamiać spadku przełączeń ze
wzrostem poprawności. Bez takiego klipu nie raportujemy ilościowej poprawy
ani czasów przyjazdu/odjazdu jako faktów.
