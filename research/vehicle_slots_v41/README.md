# UFPR04 — ręczna kolejka relacji pojazd–stanowisko

Moduł działa na oryginalnych poligonach `cars` i `spots` z przypiętej wersji
MetaPKLot. Wymaga Pythona ≥ 3.11 i Pillow. Korzysta z typów oraz walidatora
`research/datasets_v4`. Używa historycznego rejestru, polityki i manifestu jako
oddzielnych wejść. Weryfikuje JPEG zarówno względem SHA-256 w v4, jak i blobu
w przypiętym drzewie MetaPKLot. Do kolejki dopuszcza wyłącznie użycie
diagnostyczne i zachowuje pierwotną oraz efektywną rolę źródła.

```bash
cd /home/karoljaron/Projekty/ParkVision
BASE='/home/karoljaron/Dokumenty/ChatGPT/System parkingowy'
OUT="$BASE/model-improvement-v4-1-2026-11-10-v1"
PY=research/paint-segmentation/.venv/bin/python
"$PY" -m research.vehicle_slots_v41 prepare \
  --v4 "$BASE/model-improvement-v4-0-2026-11-10-v1/final" \
  --cars "$OUT/source/ufpr04_cars.tar.xz" \
  --spots "$BASE/dataset-model-improvement-2026-11-08-v1/data/source/ufpr04_spots.tar.xz" \
  --tree "$BASE/paint-segmentation-provenance-2026-11-01/metapklot-immutable-tree.json" \
  --selection "$OUT/selection.json" \
  --output "$OUT/rebuild"
```

Wynik zawiera `queue.json`, `pairs.csv`, nakładki, kopie niezmienionych
adnotacji wybranych obrazów oraz wynik Validation Gate. Niezweryfikowane źródło
lub sprzeczność identyfikacji zdjęcia trafiają do `manual-holds.json`.
Sprzeczność danych `cars` i `spots`, brak maski oraz niejednoznaczne ID są
widoczne w rekordach klatki. Kod wyjścia `2` oznacza odmowę lub niepełny eksport.

Klucz `camera_slot_key` wiąże powtarzalny poligon z kamerą. Porównanie kluczy
obowiązuje w obrębie tej samej geometrii widoku. Tożsamość fizyczna przy zmianie
kadru wymaga osobnej weryfikacji. Identyfikator pojazdu źródłowego to `car_id`,
a jednostkę maski w danej klatce identyfikuje `car_annotation_id`. Krotka
`(image_id, car_annotation_id, spot_annotation_id)` identyfikuje parę do oceny.
Jeden pojazd może mieć oddzielną relację do dowolnej liczby stanowisk.

Do przeglądu skopiuj `pairs.csv`, obejrzyj nakładki i uzupełnij kolumny
`relation`, `reviewer` i `note` w ocenionych wierszach. Dostępne wartości
to `owning_slot`, `perspective_overlap_only`, `physically_blocks_neighbor`
i `uncertain_or_occluded`. Wiersze pozostawione jako `not_assessed` nie tworzą
decyzji. Można dodać parę z tej samej klatki, wskazując istniejące ID maski
i poligonu. Propozycje bliskości służą wyłącznie wyborowi materiału do oceny.

```bash
"$PY" -m research.vehicle_slots_v41 review \
  --queue "$OUT/final/queue.json" \
  --csv "$OUT/ocena-operatora.csv" \
  --output "$OUT/kolejka-po-przegladzie.json"
```

Wynik jest oddzielną rewizją diagnostyczną. Wersja źródłowa pozostaje w
`queue.json`. Test integracyjny można włączyć, wskazując katalog wynikowy przez
`PARKVISION_V41_ARTIFACTS`.
