# ParkVision — rejestr pochodzenia i walidator v4.0

Moduł badawczy, niezależny od silnika. Rejestr opisuje źródłowy zbiór i jego
wersję, plik lub człon ZIP z SHA-256, historię przekształceń, lokalizację fizyczną,
kamerę, sesję, dzień oraz znaczenie i kompletność adnotacji. Każde ustalenie
może odsyłać do konkretnego miejsca w dowodzie. Brak informacji i potwierdzony
brak powiązania to dwa różne stany.

- `contracts.py` i `storage.py`: typy, ścisły JSON, hashe i sprawdzanie plików.
- `import_*.py` i `compiler.py`: odtwarzalny import lokalnych materiałów. Rola
  historyczna i oryginalny split pozostają w rejestrze niezależnie od decyzji v4.
- `graph.py` i `split_rules.py`: osobne zależności zdjęcia, treści i miejsca.
  Wspólny rodzic przestrzenny łączy podział danych bez kasowania kafli.
- `review_queue.py`: niezatwierdzone dopasowania; blokują dopuszczenie
  powiązanej próbki, ale nie tworzą relacji zdjęcia ani miejsca.
- `gate.py` i `supervision.py`: granice historyczne, dopuszczenie do konkretnego
  zadania oraz udokumentowane wyjątki dla pokrycia klas lub zmiany semantyki.

Walidator przyjmuje trzy oddzielne wejścia: rejestr, zaufaną politykę historyczną
i manifest intencji. Manifest podaje hashe rejestru i polityki. Właściciel danych
utrzymuje zaufane wejścia i dowody osobno od propozycji splitu. Zgodność hashy
potwierdza spójność wejść; autentyczność ustaleń o miejscu, prawach i pokryciu
klas nadal wymaga udokumentowanej weryfikacji. Zatwierdzony protokół musi wskazać
recenzenta, konkretne rekordy, klasy, dane i wersję implementacji. W v4.0
nie ma zatwierdzonych protokołów rozszerzających etykiety Roboflow.

Uruchomienie z głównego katalogu właściwego checkoutu, Python ≥ 3.11:

```bash
cd /home/karoljaron/Projekty/ParkVision
BASE='/home/karoljaron/Dokumenty/ChatGPT/System parkingowy'
python3 -m research.datasets_v4 build \
  --base "$BASE" \
  --audit "$BASE/deep-dataset-review-2026-11-10-v1" \
  --v3 "$BASE/roboflow-expansion-2026-11-09-v1" \
  --output "$BASE/model-improvement-v4-0-rebuild"
```

Katalog wyjściowy musi być nowy. Import korzysta z audytu, jego lokalnych
dowodów, wcześniejszych manifestów, przygotowanych plików i oryginalnych ZIP-ów;
nie przenosi ani nie modyfikuje wejść. Kopia uruchomionego kodu i jego SHA-256
trafiają do katalogu wyjściowego. Artefakty mają bezwzględne ścieżki do
sprawdzonych plików, więc odtworzenie na innej maszynie wymaga odpowiedniej
lokalnej struktury danych i ponownego importu.

```bash
V4="$BASE/model-improvement-v4-0-2026-11-10-v1/final"
python3 -m research.datasets_v4 validate \
  --registry "$V4/registry.json" \
  --policy "$V4/historical-policy.json" \
  --manifest "$V4/manifest-v4.json" \
  --output "$BASE/walidacja-v4-lokalna.json"
```

Kod wyjścia `0` oznacza prawidłowy manifest, `2` odmowę. Wynik zawiera
reguły, rekordy, uzasadnienia i powody kwarantanny. `accepted=true` przy
`quarantine` oznacza zgodę na zachowanie rekordu w kwarantannie. Wykorzystanie
do nauki wymaga poprawnego manifestu, zaakceptowanego rekordu z rolą
`train` lub `validation` oraz wskazanego zadania. `protected_test` służy
zachowaniu historycznej roli. Propozycja drogowa jest kontrfaktyczna i podlega
tym samym blokadom.

Testy jednostkowe uruchamia `python -m pytest research/datasets_v4`.
Test na lokalnym zbiorze wymaga ustawienia `PARKVISION_V4_ARTIFACTS`
na wynik `build`. Opcjonalny `PARKVISION_V4_ACCEPTANCE_OUTPUT` zapisuje
szczegóły testu i oba manifesty próbujące naruszyć granice w nowych plikach.
