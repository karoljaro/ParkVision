# Osobny dataset widoczności ROI

Stan v2: przygotowanie danych i narzędzia operatora. **Bez treningu i bez bramki
widoczności w observe().** Etykieta occupied/free nie mówi, czy ROI jest czytelny.

## Słownik

| Wartość | Znaczenie |
| --- | --- |
| `clear` | Stan konkretnego miejsca da się ustalić bez istotnego zasłonięcia. |
| `partial-but-readable` | Część widoku jest zasłonięta lub zdegradowana, ale stan miejsca nadal da się ustalić. |
| `unreadable` | Nie da się wiarygodnie ustalić stanu miejsca z tej klatki. |
| `unassessed` | Brak oceny lub operator nie wie. Nie jest klasą treningową ani założeniem clear. |

Samochód stojący we własnym miejscu nie oznacza automatycznie unreadable.
Oceniany jest stan miejsca, a nie widoczność każdego piksela asfaltu. Nieczytelność
może wynikać z sąsiada, przeszkody, nocy, pogody, rozmycia lub utraty widoku.
Historyczne `partial` mapujemy na readable tylko, gdy operator podał znany stan
free/occupied. Nie wyprowadzamy etykiet widoczności z modelu, bboxa ani farby.

## Odtworzenie z istniejących plików

Z repozytorium:

```bash
python=research/paint-segmentation/.venv/bin/python
artifacts='/home/karoljaron/Dokumenty/ChatGPT/System parkingowy/model-improvement-v2-2026-11-09-v1'
"$python" -m research.occupancy.visibility_v2.prepare_local \
  --workspace '/home/karoljaron/Dokumenty/ChatGPT/System parkingowy' \
  --datasets '/home/karoljaron/Datasets/ParkVision' \
  --output "$artifacts/new-visibility-preparation"
```

Import wybiera jawnie poprawione wersje etykiet ACPDS i CNR6 oraz istniejące
kolejki UFPR05 i diagnostyki CNR6. Sprawdza łańcuch hashy manifest → kolejka →
etykiety i obrazy. Żadne źródło nie jest nadpisywane. Mała nowa kolejka obejmuje
24 ROI: po 12 z UFPR04 i CNR camera3, najwyżej jeden na dzień, według hasha
identyfikatora. Dobór nie używa wyników klasyfikatora.

Powstają `source-index.json`, `source-audit.json`, `dataset/manifest.json`,
RGB cropy PNG, tablice RGB+ROI NPY, pełne klatki do przeglądu i `review.html`.
Każdy rekord zachowuje źródło, SHA-256, geometrię, lokalizację, kamerę, dzień,
view-local slot ID, licencję i odnośniki do oryginalnych ocen. Nie wymyślamy
physical_slot_id, dnia ani lokalizacji ACPDS. Wszystkie ROI mają rolę fizycznego
miejsca parkingowego; T-01 i PUCPR są blokowane.

Identyczne zdekodowane cropy RGB+mask są scalane z pełnym pochodzeniem.
Sprzeczne etykiety trafiają do ponownej oceny. dHash 256-bit, Hamming ≤6,
wskazuje kandydatów bliskich powtórek, bez deklaracji pełnej deduplikacji.
Kandydaci przecinający podziały są kierowani do quarantine.

Propozycja podziału grupuje cały fizyczny parking UFPR (obie kamery, wszystkie
dni i miejsca) w train, a cały CNR Pisa w val. ACPDS pozostaje w quarantine:
brak pełnego mapowania lokalizacji. Nie ma niezależnego test. Pole
`training_eligible=false` i `training_ready=false` jest celowe. Proponowane
splity nie są upoważnieniem do uczenia. Historyczne kolejki diagnostyczne
nie stanowią losowej reprezentatywnej próby.

## Praca operatora

Otwórz `dataset/review.html` w przeglądarce. Domyślnie pokazuje 24 nieocenione
rekordy; można przełączyć na wszystkie historyczne oceny. Wpisz identyfikator
operatora. Widać crop z kontekstem i pełną klatkę z konturem celu.

- `1`: clear, `2`: partial-but-readable, `3`: unreadable, `U`: nie wiem/pomiń.
- Strzałki przechodzą między rekordami; ocena pozostaje przy danym rekordzie.
- Można dopisać uwagę. Postęp zachowuje się lokalnie w tej samej przeglądarce.
- „Pobierz etykiety JSON” zapisuje osobny plik związany hashem z manifestem.

Import ocen zapisuje nową rewizję obok oryginalnego manifestu:

```bash
"$python" -m research.occupancy.visibility_v2.dataset apply \
  --manifest "$artifacts/visibility/dataset/manifest.json" \
  --labels "$downloaded_labels_path" \
  --output "$artifacts/visibility/dataset/manifest-r2.json"
"$python" -m research.occupancy.visibility_v2.dataset verify \
  --manifest "$artifacts/visibility/dataset/manifest-r2.json"
```

Historię poprzednich etykiet, operatora i hash pliku zmian zachowuje manifest.
Aby obejrzeć nową rewizję, polecenie `dataset review --manifest ... --output ...`
generuje nowy HTML w tym samym folderze. Brak oceny nadal znaczy unassessed.

## Rzeczywiste braki

172 oceny historyczne: 87 clear, 65 partial-but-readable i 20 unreadable.
Do tego 24 nieocenione nowe rekordy, bez dokładnych duplikatów w tym eksporcie.
W UFPR są tylko 3 unreadable, w CNR 13, a 4 ACPDS pozostają w quarantine.
Nie ma trzeciej niezależnej lokalizacji testowej ani wystarczającej różnorodności
nieczytelności w train. Liczba cropów nie jest liczbą niezależnych miejsc/dni.

ClearPix ma 14 skorelowanych przedziałów (7 clear, 7 partial), bez unreadable.
Nie mnożymy ich przez liczbę klatek. Potrzebne byłyby sprawdzenie praw, powiązanie
czasów z cropami i pominięcie niepotwierdzonego regionu 6. Nie importowano ich.

Trzy lokalne ZIP-y Roboflow mają geometrię miejsc/farby lub bboxy car/free,
**zero etykiet widoczności** i niepełne site/camera/day. Kategorie blocked_space
i parking_space nie odpowiadają unreadable/clear. Zachowane są deklaracje
licencji eksportera; pochodzenie poszczególnych zdjęć pozostaje nierozstrzygnięte.
Nie przyjmujemy splitów eksportera i nie kopiujemy tych obrazów do datasetu.

Prawa: ACPDS MIT; PKLot CC BY 4.0, Almeida i in. Dla CNR lokalny materiał
powołuje ODbL 1.0 z metadanych MetaPKLot; upstream nie był niezależnie
potwierdzony. Dane zostają lokalne. MetaPKLot: przypięty commit
`9218800c4eb2b7c1b17b51c8cdb07ad51686f8a2`, de Almeida i in., DOI
`10.1007/s00521-026-12398-0`. Szczegóły w źródłowych README i source-audit.

Następne pojedyncze zadanie: ocenić przygotowane 24 ROI i zaimportować JSON.
Potem można określić konkretny brak unreadable i lokalizacji; nie uruchamiać
treningu ani kolejnego dużego pilotażu occupancy na podstawie samej liczby cropów.
