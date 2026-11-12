# Inicjalizacja geometrii parkingu v4.3

Detektor analizuje jeden obraz referencyjny. `OpenCVSlotDetector.detect(frame)`
zwraca propozycje `SlotProposal`. `analyze(frame)` dodatkowo zachowuje źródłowe
odcinki LSD, grupy, dopasowane krawędzie, osie oznakowania, węzły i ich wsparcie
oraz uzasadnienia decyzji. Dowody są w pikselach pełnego obrazu.
`Polygon` zachowuje konwencję `x/(W−1)`, `y/(H−1)`.

## Przygotowanie propozycji

W katalogu głównym repozytorium, przy zainstalowanym pakiecie silnika:

```bash
PY=apps/engine/.venv/bin/python
IMAGE=/pełna/ścieżka/do/referencji.jpg
REVIEW=$(mktemp -d)/parking-r1-review

"$PY" -m parkvision.geometry_setup propose \
  --image "$IMAGE" --output-dir "$REVIEW" \
  --parking-id parking-a --parking-name 'Parking A' \
  --camera-id kamera-a --camera-name 'Kamera A' --revision 1

xdg-open "$REVIEW/preview.png"
labelme --nodata "$REVIEW/draft.json"
```

`preview.png` zestawia oryginał, odcinki LSD z osiami i węzłami L/T/X oraz
propozycje ROI. Linie ciągłe oznaczają fragmenty wsparte obserwacją, kreskowane
oznaczają oszacowanie, a `?` wskazuje brak rozstrzygnięcia. Numery odpowiadają
`candidate_id` w `evidence.json`. Dowody wskazują linie, ich źródłowe odcinki,
węzły i powód decyzji. Szkic odnosi się do hasha i orientacji obrazu. Jeśli
metadane EXIF zmieniają wyświetlaną orientację, najpierw zapisz osobną
referencję z ustaloną orientacją pikseli.

| Status | Znaczenie |
| --- | --- |
| `observed` | Widoczne odcinki wystarczają do ustalenia granic w tolerancji detektora. |
| `estimated` | Co najmniej jeden fragment został uzupełniony w ograniczonym zakresie lub domknięty między zgodnymi końcami. |
| `unresolved` | Detektor nie wygenerował wiarygodnego poligonu. Szczegóły pozostają w dowodach. |
| `operator_verified` | Projekcja odczytana z jawnie zatwierdzonej konfiguracji. |

W LabelMe popraw wierzchołki, usuń niepoprawne propozycje i dodaj brakujące
miejsca, jeśli masz wystarczające dowody. `label` jest trwałym ID fizycznego
miejsca, a `description` jest nazwą wyświetlaną. Przy kolejnej rewizji pracuj
na kopii poprawionej anotacji lub przypisz istniejące ID w nowym szkicu.
Automatyczne ID szkicu są powtarzalne dla tej samej referencji i geometrii.
Powiązanie miejsca między ujęciami potwierdza operator. Zapisz poprawioną
anotację w katalogu przeglądu jako `reviewed.json` z `imageData: null`.

## Zatwierdzenie i odczyt

```bash
"$PY" -m parkvision.labelme_configuration preview \
  --annotation "$REVIEW/reviewed.json" --image "$IMAGE" \
  --output "$REVIEW/reviewed-preview.png"
xdg-open "$REVIEW/reviewed-preview.png"
```

Po sprawdzeniu podglądu użyj wyświetlonego SHA-256 poprawionej anotacji.
Podaj rzeczywiste dane operatora i czas zatwierdzenia w ISO 8601 ze strefą.

```bash
"$PY" -m parkvision.labelme_configuration approve \
  --annotation "$REVIEW/reviewed.json" --image "$IMAGE" \
  --reviewed-annotation-sha256 "$REVIEWED_SHA256" \
  --approved-by "$OPERATOR" --approved-at "$APPROVED_AT" \
  --output "$REVIEW/parking-r1.json"

"$PY" -m parkvision.geometry_setup verified \
  --configuration "$REVIEW/parking-r1.json" --image "$IMAGE" \
  --camera-id kamera-a --output-dir "$REVIEW/verified"
```

`verified/verified.json` i `verified/preview.png` pokazują konfigurację odtworzoną
z pliku. Zmiana pozycji kamery, kadru lub obiektywu wymaga ponownego przeglądu.

## Zakres

Detektor opiera się na lokalnej powtarzalności separatorów i wspólnej granicy
poprzecznej. Szacuje krótką przerwę albo brakujący fragment w obrębie
potwierdzonego sąsiedztwa. Zakręty, nieregularne końce rzędów, pofragmentowana
farba i długie zasłonięcia wymagają przeglądu. Dopasowany odcinek może łączyć
dwie obserwacje, a rozkład rzeczywistego wsparcia w luce pozostaje jawny.
Geometria nie korzysta z obrysu pojazdu. Wyjściem detektora jest szkic,
którego akceptację zapisuje istniejący workflow konfiguracji.
