# ParkVision v4.3 Phase 4C — rzeczywista rejestracja CCTV

Domknięcie walidacji częściowych hipotez opisuje
[uzupełnienie Phase 4C](geometry-phase4c-finalization.md). P9 jest obecnie jawnie
odrzucona jako kreskowanie; poniższy raport zachowuje wcześniejszy stan diagnostyki.

Stan z 2026-11-15. Praca na czystym lokalnym `main` po commicie
`26e428eb1742e3dd081d5685e55737ae8be2df07` obejmującym Phase 4B.
Bez commit, push, PR, pobierania danych, treningu i zmian zatwierdzonych konfiguracji.

## Wynik i granice wniosku

Bezpiecznie przyjęto **2 z 7 dodatkowych klatek** — Ion Alarm 240 i 350 s.
ClearPix 160/240/300 s oraz Family Dollar 450/535 s pozostają odrzucone.
Istniejący pipeline wykonał rzeczywistą fuzję Ion, ale **nie poprawił pełnych ROI**.
Powstały trzy nierozstrzygnięte hipotezy częściowe. Nie są potwierdzonymi miejscami.
Jedna leży na białym kreskowaniu, dwie przy niejednoznacznej geometrii prawego
brzegu obrazu, pojazdów i niebieskich oznaczeń. Nie należy ich zatwierdzać automatycznie.

To weryfikacja mechanizmu rejestracji na historycznych materiałach diagnostycznych,
nie niezależny benchmark skuteczności CCTV ani potwierdzenie generalizacji ROI.

## Przyczyny wcześniejszych odmów

- **ClearPix** — poprzednia maska wybierała głównie słupy i skraj kadru. Punkty
  miały tylko 4.4–5.7% pokrycia, a p90 wynosiło 1.524–1.544 px. Trwałe pęknięcia
  asfaltu i końce farby dają dodatkowe odniesienia, lecz część zasłaniają pojazdy
  i piesi. Brakuje wystarczającego, rozłożonego zbioru kontroli. Odmowa pozostaje.
- **Family Dollar** — przy redukcji 3840→1200 jeden piksel roboczy odpowiada
  3.2 px źródłowym. Wstępne punkty ORB bez doprecyzowania nie dawały wystarczającej
  zgodności w limicie 2 px RANSAC. Powtarzalna cegła i kompresja dodatkowo utrudniają
  jednoznaczne dopasowanie. Lokalne pomiary sugerują mały ruch około 2.8 px,
  a nie dowodzą przestawienia kamery. Po poprawce wybrane dopasowania mają 25/28
  inlierów, ale tylko 4/5 komórek przestrzennych. Dla 450 s nie przechodzi też
  kontrola błędu odrębnego obszaru. Brak podstaw do bezpiecznej akceptacji.
- **Ion Alarm** — maska Phase 4B wycinała trwałą farbę, narożniki kreskowania i
  część krawężników. Pozostawały wąskie obszary ogrodzenia. Kontrolowane wskazanie
  stałych, widocznych struktur naprawia pokrycie. Już niezmieniony algorytm Phase 4B
  z tymi regionami przechodził dotychczasową bramkę. Zysk wynika przede wszystkim
  z właściwego doboru odniesień, nie z poluzowania progów.

Wszystkie serie wizualnie zachowują nominalny widok z własnego pliku kamery.
Nie wykryto dowodu istotnej zmiany zoomu/orientacji. Tylko Ion ma jednak dostateczne
odniesienia do ilościowej akceptacji. Niski błąd lokalny ClearPix/Family nie
potwierdza transformacji na całej powierzchni stanowisk.

## Zmiany rejestracji

1. ORB nadal ogranicza maksymalny wymiar do 1200. Współrzędne wracają do oryginału
   z właściwą konwencją środków pikseli po skalowaniu.
2. Punkty są doprecyzowane w pełnej rozdzielczości. Sprawdzane są śledzenie
   przód/tył do 0.75 px, ograniczenie odejścia od deskryptora oraz zgodność lokalnej
   tekstury NCC co najmniej 0.85. Nie łączymy obrazów ani masek samochodów.
3. Margines ORB zmniejszono z 31 do 15 pikseli roboczych, odpowiadających połowie
   okna deskryptora. Dotychczas wycinał stabilne narożniki ogrodzenia przy brzegu.
   Dopasowanie nadal wymaga poprawnej natywnej tekstury i wszystkich kontroli.
   Odrębny test syntetyczny sprawdza ten przypadek.
4. Obserwacje tej samej cechy z poziomów piramidy ORB są usuwane przy odległości
   poniżej 12 px. Ten sam narożnik nie może udawać punktu dopasowania i kontroli.
5. W każdej komórce siatki część punktów jest odkładana **przed RANSAC**. Nie służą
   dopasowaniu. Wymagamy ich rozłożenia, błędów p90 do 1.5 px i maksimum do 3 px.
6. Opcjonalna rozłączna maska kontroli dostarcza **dodatkowe** punkty. Nie zastępuje
   globalnego zbioru odłożonego. Każda grupa ma własne błędy, a jej zła zgodność
   albo brak wystarczających punktów powoduje odmowę. Nie ma cichego pomijania maski.
7. Stabilność transformacji sprawdzamy po odłożeniu kolejnych komórek dopasowania.
   Maksymalna różnica projekcji narożników obrazu nie może przekroczyć 1.5 px.

Niezmienione limity podstawowe to minimum 25 inlierów, udział 0.60, p90 1.5 px,
otoczka co najmniej 15%, sześć komórek 4×3, zmiana skali do 0.5%, obrót do 0.75°
i przesunięcie do `max(8 px, 0.012 × przekątna)`.
Nadal stosujemy transformację podobieństwa. Nie dodano homografii.

Odłożone punkty ORB pochodzą z tej samej metody. Nie nazywamy ich całkowicie
niezależną walidacją. Dodatkowo wynik Ion sprawdzono lokalnym NCC na pięciu
odrębnych, wybranych przez analityka stałych punktach, niewykorzystanych w dopasowaniu.
Także one nie są ręcznym GT ani niezależnym benchmarkiem.

Wariant wstępny używający wyłącznie dodatkowej maski kontroli miał zbyt małe
pokrycie. Finalny protokół zawsze zachowuje rozłożony holdout z puli głównej
i dokłada osobne regiony. Nie usuwa źle dopasowanych kontroli i nie zmienia limitów.
Wyniki wstępne pozostają w osobnych katalogach, wyniki końcowe w `registration-verified`.

## Opcjonalne punkty wskazane przez operatora

`propose-sequence` zachowuje interfejs CLI. Manifest może mieć `validation_mask`
obok `background_mask`. Każda klatka może podać własne maski, gdy widoczność się
zmienia. Maska może zawierać ocenioną trwałą farbę i pęknięcia nawierzchni, lecz
nie całą powierzchnię parkingu bez uzasadnienia. Samochody, odbicia, zmienne cienie
i roślinność pozostają wykluczone. Wymagana jest ocena widoczności w obu obrazach.

Wpis klatki może opcjonalnie zawierać `registration_points` jako obiekt lub ścieżkę
do JSON. Wymagane pola tego obiektu są następujące.

```json
{
  "provenance": "kto wskazał punkty, jakie stałe struktury i ograniczenia",
  "reference_pixel_sha256": "SHA-256 zdekodowanych pikseli BGR referencji",
  "source_pixel_sha256": "SHA-256 zdekodowanych pikseli BGR źródła",
  "fit": [{"id": "fit-1", "reference": [100, 200], "source": [103, 198]}],
  "check": [{"id": "check-1", "reference": [300, 400], "source": [303, 398]}]
}
```

Przykład pokazuje format pojedynczej pary, nie kompletną dopuszczalną kontrolę.
Potrzeba minimum sześciu punktów fit i sześciu check, rozłożonych w co najmniej
sześciu komórkach i obejmujących otoczką 15% obrazu w obu układach. Punkty muszą
być skończone, w obrazie, mieć unikatowe ID i odległość co najmniej 12 px.
`pixel_sha256` można odczytać z `multi_frame.frames` w dowodach. Nie jest to hash pliku PNG.

Transformacja jest dopasowana do wszystkich punktów fit, bez cichego usuwania
niezgodnych wskazań przez RANSAC. Check nie służy dopasowaniu. Wszystkie punkty
przechodzą także kontrolę rzeczywistej tekstury 15×15, więc zgodne liczby na
niepasujących obrazach nie wystarczą. Kontrola geometrii, stabilności i ruchu
kamery pozostaje aktywna. Błędne punkty lub hashe powodują odmowę, bez powrotu
do automatycznej rejestracji. W dowodach metoda ma nazwę `control_points`.
Nie oznacza to zatwierdzenia operatora ani `operator_verified`.
Nie użyto tego trybu jako obejścia odmowy na rzeczywistych klatkach.

## Pomiary Ion Alarm

Wspólna referencja to klatka 0 s, 1920×1440. Pełne współrzędne, flagi inlierów,
grupy kontroli, źródła i hashe znajdują się w dowodach oraz plikach diagnostycznych.
Podane czasy są PTS pliku, nie odstępami odczytanymi z nadruku zegara kamery.

| Źródło | Fit inliery | Fit p50 / p90 / max [px] | Fit otoczka / komórki | Nieużyte check | Check p50 / p90 / max [px] |
| --- | --- | --- | --- | --- | --- |
| 240 s | 37/38 | 0.098 / 0.217 / 0.597 | 35.49% / 6 | 72 | 0.100 / 0.307 / 0.688 |
| 350 s | 37/37 | 0.151 / 0.348 / 0.506 | 35.72% / 6 | 72 | 0.188 / 0.513 / 2.041 |

Check obejmuje odpowiednio 40.72% i 40.45% obrazu oraz siedem komórek.
Stabilność projekcji narożników wynosi 0.143/0.246 px. Skala to
0.999981/1.000050, obrót −0.000628°/−0.001186°, przesunięcie
(0.004, 0.037)/(−0.083, −0.074) px. To niemal nieruchomy kadr.
Maksymalna zmiana końca rzeczywiście wykrytej osi po transformacji to 0.036/0.112 px.

Na pięciu odrębnych punktach lokalnego NCC p90 wynosi 0.150/0.131 px,
a maksimum 0.230/0.134 px. Kontrole obejmują farbę przy analizowanej nawierzchni,
krawężniki i górny element konstrukcyjny. Nie sprowadzają się do jednego obiektu.
P90 starego ORB na nowej masce było mniejsze, około 0.03 px. Zmieniono jednak
zbiór i dokładność pomiaru oraz usunięto duplikaty piramidy. Nie przedstawiamy
tej nieporównywalnej statystyki jako poprawy lub pogorszenia dokładności ROI.

## Rzeczywista fuzja

| Wielkość | Referencja single | Multi 0/240/350 s |
| --- | --- | --- |
| Osie jasnych pasów, także pojazdów/kontekstu | 93 | 134 |
| Geometryczne węzły topologii | 36 | 100 |
| Diagnostyczne rekordy rzędów | 17 | 69 |
| Otwarte struktury rzędów | 0 | 2 |
| Pełne ROI | 0 | 0 |
| Częściowe hipotezy unresolved | 0 | 3 |
| Pozostałe unresolved | 1 | 6 |

85 osi ma źródła z kilku klatek, 49 z jednej. Odrzucono 23 sprzeczne/niejednoznaczne
obserwacje — 14 z powodu położenia/szerokości i 9 z powodu niejednoznacznego
lub powtórnego dopasowania. Konkretne ID, współrzędne i powody są w
`real-fusion-comparison.json` i `ion-alarm-fusion/evidence.json`.

Nie są to liczby potwierdzonych fizycznych separatorów, narożników ani rzędów.
Wiele osi leży na pojazdach. Hipoteza P9 dotyczy nieparkingowego białego kreskowania.
P7/P8 przy prawym brzegu nie mają wystarczająco obserwowalnych fizycznych granic.
Pełne fałszywe ROI i duplikaty wynoszą zero, bo pełnych propozycji nie ma.
FN i pełna liczba błędnych skojarzeń osi pozostają nieustalone bez kompletnej
niezależnej geometrii i audytu wszystkich 85 skojarzeń. Nie deklarujemy zera
błędnych skojarzeń ani odzyskania fizycznych miejsc na podstawie wzrostu liczników.

## Regresja i testy

Pytest silnika ma **1067 passed**, w tym 46 nowych testów Phase 4C.
Regresja research ma **61 passed / 3 skipped**. Black dla sześciu zmienionych
plików Python, Flake8 dla src/tests/scripts i `git diff --check` przechodzą.
Testy obejmują ruch kamery, oświetlenie, powtarzalne wzory, lokalne skupienie,
dominujący poruszający się obiekt, błędne maski i punkty, brak tekstury, degenerację,
osobne kontrole, hashe, brak fallbacku i przyjętą klatkę bez nowej geometrii.
W szczególnej kontroli ruchomy obiekt daje dobry fit, lecz odrębne nieruchome
tło ma błąd 7 px i powoduje odmowę.

Sześć rzeczywistych wyników jednoklatkowych ma identyczną geometrię, dowody i
bajtowe szkice jak Phase 4B. Pilot zachowuje siedem pełnych i dwanaście częściowych,
UFPR05 osiem częściowych i lokalny łańcuch dziewięciu separatorów, UFPR04 pięć
częściowych. A/B/C zachowują IoU 0.959984 / 0.961105 / 0.964244.
Hash rzeczywistej zatwierdzonej konfiguracji jest niezmieniony.
Syntetyczna fuzja nadal daje 2 → 3 pełne ROI bez duplikatów.

Dokładniejsza rejestracja ujawniła błąd numeryczny wcześniejszego zabezpieczenia
przed duplikatami — niemal identyczne krawędzie float32 dawały zerowe pole
przecięcia. Minimalna poprawka w fuzji rozpoznaje obustronną bliskość wierzchołków.
Nie zmieniono algorytmów składania ROI, rekonstrukcji rzędów ani modeli domenowych.

LabelMe 5.9.1 rzeczywiście otworzył trzy syntetyczne propozycje. Przesunięto narożnik,
usunięto propozycję i zapisano dwie. Nieaktualny hash został odrzucony.
Zatwierdzenie testowe i odczyt zachowały ID oraz współrzędne do 1e-9 px.
Odczytano też istniejące A/B/C bez modyfikacji konfiguracji.

## Artefakty i zmienione pliki

Katalog poza Git to
`/home/karoljaron/Dokumenty/ChatGPT/System parkingowy/geometry-initialization-v4-3-phase4c-2026-11-15/`.
`diagnosis/` zawiera pochodzenie, maski, punkty lokalnego NCC oraz obrazy ORB
przed/po selekcji regionów. `registration-verified/` zawiera końcowe pomiary i
podglądy dopasowań, inlierów, odrzuceń, pokrycia oraz krawędzi przed/po wyrównaniu.
`ion-alarm-fusion/`, `clearpix-fusion/`, `family-dollar-fusion/` zawierają rzeczywiste
przeglądy CLI. `ion-comparison.jpg` pokazuje brak poprawy pełnych ROI.
`regression.json`, `real-fusion-comparison.json`, logi, `final-verification.json`,
`STATUS.json` i kopia końcowego kodu pozwalają wznowić pracę bez powtarzania inwentaryzacji.

Zmieniono `detection/frame_registration.py`, `detection/multi_frame.py`,
`geometry_sequence_setup.py`. Dodano `registration_preview.py`,
`tests/detection/test_frame_registration.py`, `tests/test_registration_manifest.py`
oraz ten raport i odnośnik w przewodniku inicjalizacji.

Nie zmieniono Mask R-CNN, ConvNeXt, Temporal Occupancy, reguły D, anotacji, splitów,
manifestów historycznych, oryginalnych archiwów ani zatwierdzonych konfiguracji.
UFPR04/05 pozostały osobnymi kamerami. Nie wykonano ponownego audytu ani Phase 5.
