# Następny evaluation stage: pomoc w jednorazowym setupie

Protokół proponowany 2026-11-01. Ten etap nie uruchamia inferencji, treningu,
strojenia, pobierania danych ani otwierania testu PUCPR. Najpierw utrwalamy
zatwierdzoną geometrię; opis implementacji: `apps/engine/docs/parking-setup.md`.
Plan poniżej zastępuje priorytet ciągłego odtwarzania układu miejsc przyjęty
w starszych planach badań. Nie zmienia zapisanych wyników eksperymentów.

## Co już wiemy, a czego nie zmierzono

Według `binary-paint-experiment.md`, binarny U-Net ResNet50 z transferem całego
binarnego modelu AVM osiągnął na reviewed ROI CNR: **paint IoU 39,81%, Dice
56,95%, precision 64,26%, recall 51,13%**. To 20 142 ocenione piksele w 17
kaflach pochodzących z 3 kadrów jednego parkingu, z bardzo dużym obszarem ignore.
Checkpoint wybrano po IoU CNR; CNR wielokrotnie oglądano przy decyzjach.
Nie jest już niezależnym materiałem do kolejnych wyborów technologii/progów.

IoU pikseli farby nie mierzy kompletności miejsc, poprawności ich ID, jakości
poligonów ani czasu operatora. Część potrzebnych granic stanowią krawężniki,
a ich niski `P(paint)` jest zgodny z zadaniem. Przerwana farba może wystarczyć
człowiekowi do narysowania poprawnego ROI; dobra maska na małym wycinku nie
gwarantuje poprawnego poligonu całego miejsca. Ignore nie jest poprawnie
rozpoznanym tłem. Nie ekstrapolujemy wyników ROI na pełny parking.

Obecnie ML dostarcza mapę farby, **nie propozycje miejsc**. `OpenCVSlotDetector`
także nie składa poligonów. Dlatego najpierw można badać pomoc nakładki dla
operatora. Metryk propozycji miejsc nie wolno przypisywać samemu modelowi
segmentacji ani uzupełniać sztucznie wygenerowanymi miejscami.

## Mała próba niezależna od dotychczasowych decyzji

1. Zacząć od **jednego nowego fizycznego parkingu**, najlepiej własnego/nabytego
   za zgodą, ze znaną lokalizacją, kamerami i historią pozyskania. Pilot sprawdza
   procedurę i użyteczność, nie dowodzi generalizacji. Dopiero gdy procedura
   działa, powtórzyć ją na drugim parkingu o innym układzie/perspektywie.
2. Na parking wystarczy na początek 1–2 stabilnych widoków i 2–3 wybrane ujęcia
   na widok: czytelna referencja do setupu oraz trudniejsze zasłonięcie/cień/mokro,
   jeśli są dostępne. Nie zbieramy tysięcy bliskich klatek. Nie trzeba czekać
   na deszcz, żeby sprawdzić podstawowy workflow.
3. Przed inferencją zamrozić manifest: `parking_id`, `camera_id`, sesję/czas,
   SHA256, wymiary, źródło/zgodę na wykorzystanie i warunki sceny. Wszystkie
   widoki, klatki, cropy i anotacje tego samego parkingu pozostają jedną grupą.
   Zestaw służy ewaluacji, nie trafia automatycznie do treningu.
4. UFPR, CNR i zamknięte PUCPR nie są „nowym parkingiem”. Nie otwieramy PUCPR
   na potrzeby tego pilota. Nie ponawiamy audytu całej puli publicznych danych.
   AVM ma nieznane ID parkingów/sekwencji, więc nie da się bezwarunkowo dowieść
   rozłączności ze wszystkimi scenami pretrainingu. Nowe, udokumentowane własne
   dane zmniejszają niepewność; ograniczenie AVM pozostaje jawne w raporcie.
5. Niezależny operator przed oglądaniem predykcji oznacza fizyczne miejsca i
   ich referencyjne poligony w ustalonym, małym obszarze roboczym (np. 10–20
   kolejnych miejsc, jeśli tyle faktycznie istnieje). Wspólne miejsca w kamerach
   dostają ten sam `slot_id`. Oznaczyć również granice nierozstrzygalne, zamiast
   zgadywać ich GT. Nie wybierać tylko miejsc, na których model wygląda dobrze.
   Do pilota nie potrzeba gęstej maski farby całego parkingu.

Nie pozyskano danych teraz: do sprawdzenia zapisu geometrii wystarczą dane
syntetyczne, a kolejny obraz bez ustalonej roli i zakresu oceny nie rozwiąże
problemu architektury. Pierwsze pozyskanie to mały pakiet referencji z punktów
powyżej, po przygotowaniu podglądu konfiguracji.

## Pomiar dostępny bez składania miejsc przez model

Zamrozić istniejący binarny `best.pt` i jego SHA256 z raportu, preprocessing,
kafelkowanie oraz regułę wyświetlania mapy przed obejrzeniem nowej próby.
Nie wykonywać dalszego wyboru checkpointu/progu na CNR ani na tej próbie.
GPU nie ogranicza wyboru technologii; zapisujemy rzeczywisty sprzęt i czas.

Porównać to samo zadanie zatwierdzenia zakresu miejsc:

- ręczne rysowanie na oryginalnym obrazie;
- ręczne rysowanie z opcjonalną nakładką `P(paint)` z zamrożonego modelu;
- opcjonalnie obecna diagnostyka LSD jako osobna pomoc, jeśli operator jej używa.

Nakładki pozostają niezależne. Nie filtrujemy twardo LSD przez ML i nie
traktujemy farby jako pełnej topologii. Wybór obrazu referencyjnego/alternatywnego
ujęcia jest dozwoloną częścią setupu, a czas tej czynności liczy się do kosztu.

Rejestrować: czas do zatwierdzenia, liczbę dodanych miejsc, usuniętych/ponownie
narysowanych ROI, przesunięć/dodanych/usuniętych wierzchołków, błędów powiązania
kamer i miejsc oraz liczbę miejsc odłożonych do ponownego przeglądu. Jeśli start
jest pusty, „dodanie” jest pracą rysowania, a nie false negative modelu.
Mierzyć jakość końcowych poligonów względem niezależnej referencji.

Operator nie powinien po prostu powtarzać tego samego rysowania z pamięci.
Przy dwóch operatorach zamienić kolejność wariantów i przydzielić porównywalne
obszary; referencję przygotowuje osoba niepatrząca na podpowiedzi. Przy jednej
osobie jawnie raportować efekt uczenia i pilotażowy charakter porównania czasu.
Nie deklarować istotnej przewagi na podstawie kilku powtórzeń tej samej sceny.

## Metryki po powstaniu jawnego generatora propozycji poligonów

To późniejszy, osobno wersjonowany element. Przed badaniem trzeba zamrozić jego
kod/parametry i określić obsługiwany zakres, bez wymyślania poligonów z samej maski.
Wtedy raportujemy **przed poprawkami i po zatwierdzeniu**:

| Pomiar | Sposób raportowania |
| --- | --- |
| Kompletność i fałszywe propozycje | Jednoznaczne dopasowanie 1:1 do referencji według IoU; próg dopasowania ustalony przed oceną (propozycja pilota: 0,7). Nadmiarowe duplikaty są FP, pominięte rozstrzygalne miejsca FN. Próg dopasowania nie jest automatycznym progiem akceptacji operatora. |
| Kształt ROI | Rozkład IoU i odległości granic w pikselach oraz względem przekątnej obrazu, osobno na kamerę/parking; błędy narożników dla porównywalnych czworokątów. Uwzględnić brak propozycji w kompletności, nie ukrywać go za średnią IoU dopasowanych. |
| Koszt korekty | Udział propozycji przyjętych bez zmian, liczba dodanych/usuniętych miejsc, edycji wierzchołków i korekt ID, czas do zatwierdzenia, przypadki wymagające nowej referencji. |
| Powiązania kamer | Odsetek poprawnych powiązań tego samego fizycznego miejsca, brakujących relacji i błędnych scaleń różnych miejsc. |
| Nierozstrzygalne granice | Osobna liczba i przyczyna; nie liczyć jako poprawnych propozycji ani ukrywać poza raportowanym zakresem. |

Nie przyjmować samego IoU jako dowodu użyteczności occupancy. Tolerancję błędu
granicy powinien później uzasadnić osobny moduł zajętości, którego obecnie nie ma.

## Trudne warunki i kryterium decyzji

Oddzielić dwie sytuacje:

- **Trudny setup:** brak widocznych granic w jedynym ujęciu, pojazd na linii,
  refleksy. Mierzymy dodatkowy czas i potrzebę innego ujęcia/ręcznej korekty.
  Odłożenie konfiguracji nie jest awarią stałej pracy, a niepewności nie należy
  zamieniać w wymyślone miejsce.
- **Po zatwierdzeniu:** wybrane trudne klatki mają korzystać z tych samych ID
  i poligonów. Nie odtwarzamy granic i nie mierzymy „sukcesu reinicjalizacji”.
  Ocena zajętości i odporności jej obserwacji to osobny przyszły eksperyment.

Wynik pilota podać per parking/widok i warunek, razem z czasem i wszystkimi
odłożonymi miejscami. Nie traktować kafli/pikseli jako niezależnych parkingów.
Małe ROI paint IoU/Dice/precision/recall można dodać pomocniczo tylko na
niewielkich, wybranych **przed predykcją** obszarach, jeśli wyjaśniają konkretne
błędy; nie oznaczać tysięcy masek bez potrzeby.

Decyzja po pilocie: czy nakładka zmniejsza czas/poprawki przy co najmniej takiej
samej jakości końcowej i nie zwiększa liczby nierozstrzygniętych miejsc? Jeśli
nie, ręczny setup nadal realizuje cel produktu. Najpierw sklasyfikować problem
(obraz referencyjny, UX, braki geometrii, farba), a dopiero potem uzasadniać
nowe dane lub model. Żaden wynik pilota nie uruchamia automatycznie treningu.

Checkpointy `best.pt`, `last.pt`, pretraining AVM, overfit i frozen datasets
pozostają chronione. Status lokalnego akademickiego użycia AVM i nieznanej
formalnej licencji danych pozostaje jak w `avm-experiment.md`; ten plan nie
publikuje obrazów, masek ani wag i nie nadaje nowych warunków prawnych.
