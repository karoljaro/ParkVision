# ParkVision · ręczna ocena relacji A/B

Otwórz gotowy `final/public/index.html` w przeglądarce. Strona działa lokalnie
i bez połączenia z siecią. Można też udostępnić sam katalog `public` na
`127.0.0.1`, aby zachować stały adres i miejsce zapisu w przeglądarce.

```bash
BASE='/home/karoljaron/Dokumenty/ChatGPT/System parkingowy'
OUT="$BASE/model-improvement-v4-2-2026-11-10-v1"
xdg-open "$OUT/final/public/index.html"
```

1. Wybierz przykład. Porównaj oryginał z nakładką i w razie potrzeby otwórz
   pełny kadr. Osobno sprawdź położenie auta i wiarygodność poligonu na gruncie.
2. Wybierz status. Zapisanie statusu „Zaakceptowane” wymaga jawnego wyboru
   relacji. Wpisz autora i uzasadnienie, opcjonalnie oznacz geometrię i
   widoczność. Sugestię z przeglądu można odsłonić oddzielnie.
3. Użyj „Zapisz ocenę”. Zapis w przeglądarce jest przypisany do partii.
   „Pobierz postęp JSON” tworzy przenośną kopię wszystkich zapisanych statusów
   wraz z autorem i czasem. Można ją ponownie wczytać.
4. „Pobierz zaakceptowane CSV” zapisuje wyłącznie przyjęte decyzje w formacie
   v4.1. Aby wyeksportować także pełne pochodzenie i zweryfikować plik
   postępu względem źródeł, uruchom poniższą komendę.

```bash
cd /home/karoljaron/Projekty/ParkVision
PY=research/paint-segmentation/.venv/bin/python
"$PY" -m research.vehicle_slots_v42 export \
  --batch "$OUT/final/batch.json" \
  --progress "/ścieżka/do/parkvision-ab-progress.json" \
  --output "$OUT/oceny-operatora"
```

Wynik to `v41-reviews.csv`, `decisions.json` ze wszystkimi zapisanymi
ocenami i dowodami oraz odrębny `reviewed-v41-queue.json`. Zatwierdzenie
relacji zmienia tylko dane diagnostyczne. Moduł odrzuca zmienione pliki
źródłowe, identyfikatory spoza partii i powtórzone decyzje dla tej samej pary.
Nowy katalog wyjściowy chroni wcześniejsze wyniki.

Odtworzenie partii w nowym katalogu wymaga lokalnych wejść v4.0 i v4.1,
Pythona ≥ 3.11 oraz Pillow. Plik wyboru zawiera ręczne obserwacje, a
walidator ponownie sprawdza źródłowe klatki przed generowaniem podglądów.

```bash
"$PY" -m research.vehicle_slots_v42 prepare \
  --v4 "$BASE/model-improvement-v4-0-2026-11-10-v1/final" \
  --v41 "$BASE/model-improvement-v4-1-2026-11-10-v1" \
  --selection "$OUT/selection.json" \
  --output "$OUT/rebuild"
```
