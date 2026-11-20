# Domknięcie Phase 4C — walidacja częściowych hipotez

2026-11-15. Zachowano niezatwierdzoną pracę Phase 4C. Bez zmian rejestracji,
treningu, konfiguracji operatora, commit/push/PR i bez rozpoczynania Phase 5.

## Dlaczego powstała P9

Separatory `5620` i `5629` są ukośnymi kreskami białej strefy nieparkingowej.
T49 łączy `5629` z obwiednią `5626`, a T58 łączy `5620` z obwiednią `5614`.
Te obwiednie spotykają się w narożniku L51. Nie są niezależnymi dowodami rzędu.

Kierunki separatorów różnią się tylko o około 3.6°, odstęp kotwic wynosi około
195 px, a rozstaw przeciwległych końców mieści się w istniejącym limicie.
Obie kotwice mają wsparcie źródłowych fragmentów. Wyjątek dopuszczający dwie
kotwice T oraz liczenie różnych ID kontekstów jako dwóch potwierdzeń pozwoliły
utworzyć otwarty rząd 68.

Zabezpieczenia pełnego ROI działają w `_build_candidate` i przy szacowaniu
zamknięcia. P9 nie miała pełnego obrysu, więc nie przechodziła przez taką walidację.
Wieloklatkowość potwierdziła trwałość kreskowania, nie jego funkcję parkingową.
Nie było niezależnego potwierdzenia stanowiska poza tym samym układem oznaczeń.

## Poprawka

Walidacja częściowych hipotez w `row_reconstruction.py` rozpoznaje powtarzalną
rodzinę ukośnych kresek dochodzących do dwóch ramion **obserwowanego, źródłowo
wspartego narożnika L**. Sprawdza kierunki do wnętrza klina, co najmniej trzy
różne kreski, skończone pokrycie i relacje odstępów. Kopie tej samej kreski nie
zwiększają liczby niezależnych elementów. Dodatkowe niezależne kotwice chronią
hipotezę przed automatycznym przypisaniem do takiego wzoru.

Wykorzystano istniejące `_marking_evidence`, `_supported`, topologię i grupy LSD.
Nie można bezpośrednio zastosować `_has_hatching` wymagającego zamkniętego poligonu,
ponieważ częściowa hipoteza nie ma potwierdzonej głębokości. Nie tworzono fikcyjnego
obrysu do sprawdzenia. Nie zmieniono globalnych progów ani reguł dla skośnych stanowisk.

Odrzucona hipoteza zachowuje `status=unresolved`, `proposal=None`, źródła,
kotwice i ID rzędu. Dostaje `rejection_code=non_parking_hatching` oraz
`rejection_line_ids` — dwa pierwsze ID oznaczają ramiona obwiedni, dalsze kreski
rodziny. Nie jest liczona jako aktywna hipoteza częściowa. Podgląd oznacza ją
`R9 REJECTED`, a węzły i odcinki pozostają widoczne diagnostycznie.
Osobny licznik `rejected_partial_hypotheses` odróżnia odrzucenie od braku dowodów.
Powtórna obserwacja nie przywraca znanego odrzuconego wzoru jako miejsca.

## Wynik rzeczywisty przed/po

| Wielkość | Przed | Po |
| --- | --- | --- |
| Osie | 134 | 134 |
| Przyjęte dodatkowe klatki Ion | 2 | 2 |
| Pełne ROI | 0 | 0 |
| Aktywne częściowe hipotezy | 3 | 2 |
| Jawnie odrzucone częściowe hipotezy | 0 | 1 |
| Aktywne otwarte struktury rzędów | 2 | 1 |

P7 i P8 zachowują dotychczasowe dane i status. P9 pozostaje w dowodach jako
odrzucone kreskowanie. Surowy rekord rzędu 68 jest zachowany z oznaczeniem
odrzucenia i niezgodnym sąsiedztwem. Nie deklarujemy poprawności fizycznej P7/P8
ani niezależnej poprawy skuteczności na CCTV.

## Sprawdzenie

- **1076 testów silnika przeszło**, w tym dziewięć nowych przypadków. Obejmują
  kreskowanie, kontekst krawężnika, powtarzanie obserwacji, sąsiedztwo poprawnych
  miejsc, stanowiska skośne, częściowe T i ograniczoną widoczność separatorów.
  Mała próbka rzeczywistej geometrii P9 zawiera 133 oryginalne fragmenty LSD
  z pochodzeniem. Nie jest zbiorem treningowym ani niezależnym GT.
- Research ma **61 passed / 3 skipped**. Black, Flake8 i `git diff --check` przechodzą.
- Pilot zachowuje siedem pełnych ROI, UFPR05 osiem częściowych i łańcuch dziewięciu
  separatorów, UFPR04 pięć częściowych. Sześć szkiców LabelMe jest identycznych
  bajtowo z poprzednimi. Dawne pola geometrii i statusów pozostają zgodne.
- A/B/C zachowują IoU 0.959984 / 0.961105 / 0.964244 wobec kontroli deweloperskiej.
  Syntetyczna fuzja nadal daje 2 → 3 pełne ROI.
- Rejestracja dwóch klatek Ion jest identyczna z Phase 4C. Ponownie potwierdzono
  pięć odmów ClearPix/Family Dollar. Kod rejestracji jest identyczny bajtowo
  ze stanem sprzed domknięcia.
- Rzeczywisty LabelMe otworzył szkic syntetyczny, poprawiono narożnik, usunięto
  propozycję i zapisano dwie. Zatwierdzenie testowe i odczyt konfiguracji zachowały
  ID i współrzędne do 1e-9 px. Rzeczywista konfiguracja A/B/C nie została zmieniona.

## Pliki i artefakty

Zmiany domknięcia obejmują `detection/initialization.py`, `row_reconstruction.py`,
`multi_frame.py`, `geometry_setup.py`, `geometry_setup_preview.py`,
`geometry_sequence_setup.py`, istniejący test `test_open_rows.py`, nowy
`test_partial_validation.py` i mały fixture `tests/fixtures/partial-hatched-corner.json`.
Dokumentację uzupełniają ten raport, odnośnik w raporcie Phase 4C i przewodnik.

Artefakty są poza Git w
`/home/karoljaron/Dokumenty/ChatGPT/System parkingowy/geometry-initialization-v4-3-phase4c-finalization-2026-11-15/`.
`ion-before-after.jpg` pokazuje zmianę klasyfikacji, `ion-after/` zawiera aktualny
przegląd, `regression.json` porównania, a `before-code/` kopię wcześniejszej pracy.
Logi, `final-verification.json` i `STATUS.json` dokumentują sprawdzenia.

Kontrola jest celowo lokalna i konserwatywna. Same równoległe ukośne kreski bez
potwierdzonego układu obwiedni nie pozwalają odróżnić kreskowania od prawdziwych
skośnych stanowisk. Takie przypadki pozostają nierozstrzygnięte do oceny operatora.
