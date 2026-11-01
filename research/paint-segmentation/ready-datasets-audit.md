# Gotowe maski oznaczeń parkingowych — audyt 2026-11-01

**Nie ma podstaw, by od razu zlecać ręczne oznaczanie dziesiątek tysięcy zdjęć.**
Istnieją rzeczywiste gotowe maski, a najbliższy naszemu zadaniu sprawdzony
pakiet to **AVM SS: 6763 obrazy**. Nie został jednak dopuszczony do treningu:
brakuje potwierdzenia praw do plików danych i wiarygodnego grupowania scen.
W innych pakietach dodatkowym problemem jest semantyka: maska może obejmować
asfalt wewnątrz oznaczenia, a nie wyłącznie widoczną farbę.

Sprawdzono bieżące źródła czterech wskazanych zbiorów i czterech dodatkowych
kandydatów. Przygotowano porównania **20 oryginalnych par**: AVM SS 8,
RLMD 2, CeyMo 10. Kopie źródeł, metadane pobrań, SHA256, statystyki masek,
ranking w JSON i zdjęcia znajdują się w osobnym katalogu:

`/home/karoljaron/Dokumenty/ChatGPT/System parkingowy/ready-datasets-audit-2026-11-01-v1/`.

Nie uruchomiono treningu ani inferencji modelu. Nie zmieniono silnika CV,
anotacji użytkownika, zamrożonego Dataset A, checkpointów ani dawnych wyników.
Brak commita. W repo dodano raport i odnośnik; kod ML i CV pozostał bez zmian.

## 1. Baseline v1 zachowany

Baseline v1 nadal oznacza **18,53% foreground mIoU CNR**, najlepszą epokę 8
eksperymentu `unet-resnet50-imagenet-v2`. `best.pt` sprawdzono przez SHA256
i odczyt `weights_only=True` na CPU, bez inferencji ani ponownego zapisu.
Zgadza się epoka, architektura, wynik i hashe frozen manifestu/lock.

SHA256 `best.pt`:
`f603de439442c51164cdf708f16864678efa294db16d48bbc140c74018b4dcaf`.

SHA256 `last.pt`:
`f591c6219eb73039b76b0afe8d331085c6aa8773188ed349c2b151756ea58f45`.

Zachowano także oba checkpointy overfit:

- `unet-overfit-v1/unet-resnet50.pt`:
  `715a867614b62b70d4d99e740204142458163c1af041203007786790610242c8`.
- `unet-overfit-v2/unet-resnet50.pt`:
  `c7176c50e7a2fe6ad370896ce7b045022633b98b5e5f76a2154f81f4c6c162e6`.

Kontrola obejmuje **107 plików wcześniejszego eksperymentu**: frozen dataset,
konfiguracje, historię, wyniki, predykcje, kod użyty w wykonaniu, oba
checkpointy, wagi ImageNet i poprzedni spis integralności. Osobno objęto nią
wszystkie cztery znalezione wcześniejsze checkpointy. Dowody są w
`baseline-preservation-before.json`, `baseline-preservation-after.json`,
`baseline-v1.json` i `baseline-best.sha256` nowego audytu.

## 2. Ranking dalszej weryfikacji

To kolejność praktycznej przydatności i dostępności dowodów, **nie ranking
jakości modeli**. „Pliki sprawdzone” nie oznacza „licencja danych potwierdzona”.
Żaden z poniższych pakietów nie uzyskał statusu gotowego do naszego treningu.

| Kolejność / źródło | Rzeczywiste maski i dostęp | Licencja plików danych | Decyzja |
| --- | --- | --- | --- |
| **1. [AVM SS](https://github.com/Engro39/avm_dataset)** | Sprawdzony ZIP z 6763 maskami; ponownie sprawdzono 8 par. RGB PNG, marker jest rzeczywistą klasą. | **NIEZWERYFIKOWANA**: brak jednoznacznej deklaracji w README, drzewie repo i ZIP-ie. | Najlepszy kandydat po potwierdzeniu praw i pochodzenia grup. `other` nie daje GT curb. |
| **2. [RLMD / RLMD-AC](https://github.com/stu9113611/RLMD)** | Pobrano 2 pary JPG/PNG z oryginalnego Drive autorów, poprawnie odczytane. | **NIEZWERYFIKOWANA**; MIT na nieoficjalnym mirrorze nie jest pozwoleniem autorów danych. | Potencjalny ograniczony pretraining, ale część masek jest niezgodna z widoczną farbą. |
| **3. [PSV, Wu i in. 2018](https://arxiv.org/abs/1804.07027)** | Publikacja potwierdza segmentację. Oryginalna strona pobrania niedostępna; brak lokalnej surowej pary. | **NIEZWERYFIKOWANA**. | Oryginalne pliki i prawa nadal potrzebne. Linie przerywane zawierają w etykiecie przerwy. |
| **4. [AVP-SLAM, Liu/Yao 2022](https://onlinelibrary.wiley.com/doi/10.1155/2022/9681455)** | Publikacja opisuje GT i podaje Baidu; oryginalne maski nie zostały pozyskane. | **NIEZWERYFIKOWANA** dla osobno hostowanych danych. | Interesujący garaż, lecz dostęp i format GT pozostają niepotwierdzone. |
| **5. [CeyMo](https://github.com/oshadajay/CeyMo)** | Oficjalny sample ZIP: 10 par, maski RGB PNG, poligony JSON i bbox XML. | **NIEZWERYFIKOWANA** dla zdjęć/anotacji. Repo ma MIT z zakresem „Software”; sample nie zawiera licencji danych. | Maski części klas obejmują cały obszar symbolu. Nie importować wszystkich foreground jako paint. |
| **6. [SeRM](https://github.com/yiliu-coding/SeRMdataset)** | Autorzy opisują pixel GT; pełny link wydają po prośbie o dostęp. Nie pobrano oryginalnego zestawu. | **NIEZWERYFIKOWANA**; wymagane wyjaśnienie z autorami. | Duże źródło drogowe do rozważenia po uzyskaniu plików i praw. Nie wysłano żadnej wiadomości. |
| **7. [DLR SkyScapes](https://www.dlr.de/en/eoc/about-us/remote-sensing-technology-institute/photogrammetry-and-image-analysis/public-datasets/dlr-skyscapes)** | Autorzy opisują dense GT, także typy oznaczeń parkingowych. Oryginalne pliki niepobrane. | **NIEZWERYFIKOWANA** w tym audycie. Credit licencyjny ilustracji na stronie nie jest licencją całego zbioru. | Potencjalnie ciekawsza perspektywa lotnicza; warunki i maski trzeba pozyskać u właściciela. |
| **8. [LCRM](https://imak.ai/news/lcrm-open-dataset-assessing-wear-road-markings)** | Autorzy ogłaszają 20 tys.+ zdjęć. Na odczytanych stronach autorów nie znaleziono linku do oryginalnego pakietu. | **NIEZWERYFIKOWANA** dla plików udostępnionego zbioru. | Ciekawe zużycie farby; zapowiedź/artykuł nie zastępuje weryfikacji wydania i masek. |

### Rozmiar, pochodzenie i warunki

| Zbiór | Obrazy / rozdzielczość | Parkingi, kamery i warunki |
| --- | --- | --- |
| AVM SS | 6763, **320×160 sprawdzone** | 3614 outdoor, 3149 indoor; autorzy pokazują dzień/deszcz/odbicia w garażu. Liczba fizycznych parkingów, kamer i sesji niepodana w sprawdzonych metadanych. Nie utożsamiać liczby klatek z parkingami. |
| RLMD | **2137** według aktualnego README; historyczna publikacja mówi o 700. Próbka **1920×1080 sprawdzona**. | Obrazy z jazdy drogowej; parkingi/kamery/sesje niezweryfikowane. Dwa odczytane pliki mają numeryczne nazwy. RLMD-AC dodaje noc i deszcz. |
| PSV | 4249; 2550 train, 425 val, 1274 test według publikacji | 4 fisheye po 640×480, składane do **1000×1000**. To rozdzielczość surowych kamer i BEV, nie jeden format 640×480 całego PSV. Późniejsza praca opisuje też warianty 600×600. Jiading/Tongji, indoor/outdoor, światło/cienie, farba świeża/zużyta, biała/żółta. Liczba obiektów i oryginalne IDs niezweryfikowane. |
| AVP Liu/Yao | Liczba obrazów niezweryfikowana; artykuł podaje BEV **1090×860**, faktycznych plików nie odczytano | **Jeden garaż**, 4 kamery: front 1920×1080, pozostałe 640×480. Ciemne wnętrze, odbicia i zasłonięcia. Bez potwierdzonego outdoor/deszczu. |
| CeyMo | 2887, oficjalnie 2099 train/788 test; **1920×1080 sprawdzone** | 2 kamery według README, sample JSON ma camera/vehicle, lecz nie kompletne drive/session IDs. Normal/crowded/dazzle/night/rain/shadow. Brak inwentarza niezależnych parkingów; to zbiór drogowy. |
| SeRM | Tekst autorów: 25 158; tabela w README: 25 157; nie przeliczono lokalnie. 19 998 train/5160 test. | Korea, kamera monocular; sprzętowe camera IDs i pełne warunki niezweryfikowane. Oddzielne 21 tys. klatek odometrii z 3 tras **nie są automatycznie** tym samym zbiorem segmentacji. Rozdzielczość oryginalnego wydania niezweryfikowana. |
| SkyScapes | **16 dużych scen 5616×3744** według [oryginalnej publikacji](https://openaccess.thecvf.com/content_ICCV_2019/papers/Azimi_SkyScapes__Fine-Grained_Semantic_Understanding_of_Aerial_Scenes_ICCV_2019_paper.pdf), nie lokalnego przeliczenia | Okolice Monachium, lot śmigłowcem, 13 cm/piksel; rig 3 DSLR, wybrano zdjęcia nadir. Nie ustalono liczby parkingów ani bezpiecznych grup geograficznych z oryginalnych plików. |
| LCRM | 20 tys.+ i 12 typów oznaczeń/3 poziomy zużycia według autorów; rozdzielczość niezweryfikowana | Kamera liniowa, Moskwa/Petersburg/obwód moskiewski, różne pory dnia. Liczba kamer, tras, parkingów i metadane podziału niezweryfikowane. |

Oryginalna PSV strona `http://cs1.tongji.edu.cn/tiev/resourse/` zwróciła HTTP 500,
HTTPS timeout. Wiley i próby pobrania suplementu Liu/Yao zwróciły 403.
Przeglądarka zablokowała `pan.baidu.com` regułą site-safety; nie podjęto obejścia.
Strona DLR miała błąd pobrania i błąd łańcucha certyfikatów w bezpośrednim HTTPS;
nie wyłączano walidacji certyfikatów. To stan dostępu w tym środowisku,
nie twierdzenie, że dane nigdzie nie istnieją. Szczegóły w `evidence/access-status.json`.

## 3. Co ujawniły oryginalne maski

**AVM SS.** ZIP użytkownika ma ponownie potwierdzony SHA256
`b9a8f70fcb9f57bb1c8360034cbf09b06277b2361453edfa5039960cd33c52ef`.
Zgadza się ze wcześniejszym pełnym audytem wszystkich 6763 tripletów.
W tym etapie skopiowano bez zmiany bajtów 8 oryginalnych tripletów do audytu.
Oryginalne GT ma marker, free space, vehicle, other i ego. Ma też wcześniej
wykryte **7071 żółtych pikseli w 540 maskach**: znaczenie niepotwierdzone.
W dwóch sprawdzonych przypadkach żółty pokrywa się z nakładaniem poligonów
curb/vehicle, ale nie nadajemy na tej podstawie nowej klasy.

Propozycja po ewentualnym dopuszczeniu danych: RGB biały → paint=1;
niebieski free space i czerwony vehicle → background=0;
zielony other, czarny ego, żółty i nieznane kolory → ignore=255.
**Żadna klasa nie daje tu pewnego osobnego curb=2**. Klucz YAML nazwany curb
nie rozstrzyga znaczenia zbiorczej klasy opisanej jako curb/pillar/wall.
To propozycja mapowania, nie zatwierdzony importer ani nowe GT Dataset A.

**RLMD.** Z oryginalnego linku autorów pobrano `images/train/10.jpg`, `20.jpg`
i odpowiadające `labels/train/10.png`, `20.png`. Obie maski są **RGB uint8 PNG**.
Wymaganie P-mode w [RMSeg-UDA](https://github.com/stu9113611/RMSeg-UDA)
dotyczy formatu po konwersji do ich treningu, nie tych pobranych plików.
W próbce 10 etykieta `dashed single white`, ID 9, jest ciągłym pasmem także
nad widocznymi przerwami asfaltu. Nie wolno mapować jej w całości na paint.
Próbka nie zawiera klasy 24 `parking lot`; jej nazwa **nie dowodzi**, że
etykietuje szerokość farby, a nie pole miejsca. Klasy 9/10 i niezweryfikowane
24 powinny pozostać ignore. Inne klasy wymagają przeglądu szerokości, przerw,
zużycia i położenia na fizycznym krawężniku; brak automatycznego mapowania 1–24.

**PSV.** Oryginalna publikacja w sekcji III.B wprost opisuje etykietowanie
całej linii przerywanej wraz z przerwami. Solidne i parkingowe oznaczenia są
obiecujące, ale ich surowych konturów nie sprawdzono. Nie włączono pliku
dodatkowych anotacji VPS-Net: autorzy opisują go jako anotacje wolnych miejsc,
co nie stanowi dowodu maski farby. Brak własnej, fikcyjnej rasteryzacji slotów.

**AVP Liu/Yao.** W pracy *The Robust Semantic SLAM System for Texture-Less
Underground Parking Lot* są osobne klasy parking line/lane line/speed bump/
traffic signs. Bez oryginałów nie zweryfikowano palety, mapowania, widocznych
granic ani kompletności zasłonięć. Speed bump i inne niepewne klasy pozostają
poza naszym nadzorem. Nie łączymy tego źródła z AVP-SLAM Qin 2020 ani AVM-SLAM.

**CeyMo.** Oficjalna próbka autorów ma 45 wpisów ZIP i 10 kompletnych czwórek
JPG / maska PNG / poligony JSON / bbox XML. Wszystkie obrazy i maski odczytano.
W `141a` klasa junction box ma **cztery wierzchołki obejmujące całą kratownicę**;
oryginalna maska wypełnia także asfalt pomiędzy żółtymi kreskami. Porównanie
na `samples/ceymo/141a-junction-envelope-detail.png` pokazuje dokładnie tę
różnicę. Nie wygenerowano „poprawionej” farby progiem jasności ani inną heurystyką.
Strzałki mogą być pomocniczym podzbiorem po sprawdzeniu praw i jakości;
nie oznacza to akceptacji całego foreground, crosswalk czy regionu lane-word.

W źródłach bez osobnej klasy krawężnika **czarne/unoznaczone tło nie jest
automatycznie naszym background=0**: może zawierać curb i nieopisane oznaczenia.
Najbezpieczniejsza konwersja takich niepewnych obszarów to 255. Zbiór mający
wyłącznie pozytywne paint i resztę ignore nie zastąpi samodzielnie pełnego
uczenia trzech klas z wiarygodnymi negatywami.

## 4. Przeciek i źródła mylone z gotową segmentacją farby

AVM ma 4057/2706 train/test, ale `split_db.py` losuje pojedyncze indeksy,
bez parking/camera/session. Nie uruchamiano tego skryptu. Wcześniejszy audyt
tego samego ZIP-a wykazał brak dokładnych duplikatów, ale para **3756 test /
3757 train** to prawie ten sam widok garażu (MAD RGB 1,63/255); ponownie
obejrzano kopie obu obrazów. Nie jest to niezależny test scen.

RLMD: foldery train/val istnieją, ale nazwy 10/20 nie odtwarzają tras i sesji.
RLMD-AC według autorów ma nieopisane treningowe klatki nocy/deszczu; nie wolno
liczyć ich jako gotowych par GT. W PSV podano podział scen, ale bez pobranych
IDs nie da się potwierdzić odtwarzalności. AVP Liu/Yao to jeden garaż: wszystkie
jego kamery/sesje muszą pozostać w jednej grupie. CeyMo ma camera/vehicle,
co nie wyklucza ponownych przejazdów tą samą ulicą. SkyScapes wymaga podziału
całymi scenami/obszarami przed wycinaniem kafli; LCRM całymi przejazdami przed
dzieleniem sąsiednich pasów skanu. Same train/test nazwy nie są dowodem braku
przecieku. Nie przypisano wymyślonych parking/session IDs.

Odrzucono także niewłaściwe zadania, mimo słowa „segmentation” w nazwie:

- [ParkSeg12k](https://github.com/UTEL-UIUC/ParkSeg12k): 12 617 par 512×512,
  ok. 35 tys. parkingów w 45 miastach; maska **całego terenu parkingu**, nie farby.
- [APKLOT](https://github.com/langheran/APKLOT): 500 obrazów, 7000+ poligonów
  bloków miejsc; deklaracja MIT dla datasetu z dodatkowymi warunkami obrazów
  Google Maps nie zmienia znaczenia tych poligonów na paint.
- [RVD-UPL / AVP_Dataset](https://github.com/wuhuateng/AVP_Dataset): 3 tys.+ BEV,
  jeden garaż, 4 fisheye; README opisuje cztery narożniki całych miejsc
  i ich widoczność. Nie potwierdzono oryginalnych masek szerokości farby/licencji.
- [AVM-SLAM 2024](https://github.com/yale-cv/avm-slam_dataset): sekwencje
  4 kamer 1280×960 i BEV 1354×1632 jednego garażu, nie dowód wydania ręcznego
  paint GT. Autorzy podają academic-only/GPLv3 i kontakt w sprawie komercyjnego
  użycia; nie przenosimy tych słów na nieudostępnione maski ani inne zbiory.

## 5. Konkretny plan bez masowego opisywania zdjęć

1. **Odblokować AVM SS u właściciela danych**: potwierdzenie praw do zdjęć
   i masek z `avm_seg_db.zip`, do uczenia modeli i docelowego sposobu użycia;
   informacja o parkingach/sekwencjach, zbiorczej klasie other i żółtej palecie.
   Udostępniony ZIP, README „public dataset” i licencja kodu nie zastępują
   tej informacji. Nie wysłano wiadomości ani zgłoszenia bez polecenia użytkownika.
2. Po spełnieniu warunków zrobić minimalny importer z **oryginalnych masek**
   do osobnego external research datasetu: jawna paleta, nieznane kolory=255,
   other=255, brak GT curb, kontrola rozmiarów i SHA256, zachowane źródło,
   licencja i grupy. Nie tworzyć LabelMe `reviewed=true` w imieniu użytkownika.
   Testy obejmą paletę/ignore, zmianę źródła, nakładanie, geometrię i przeciek.
3. **Pretraining na dodatkowych danych**, w nowym katalogu, bez zmiany baseline:
   najlepiej zadanie marker kontra pewne negatywy z AVM. Osobna głowica
   pomocnicza i jawny brak klasy curb; encoder/decoder można potem przenieść
   do naszego U-Net z głowicą trzech klas. Nie utożsamiać non-paint pomocniczego
   zadania z pełnym background/curb naszego zbioru.
4. **Dostrajanie wyłącznie na UFPR train**; CNR zachować wyłącznie do walidacji
   i wyboru checkpointu. Porównanie z zachowanym baseline v1 na dokładnie
   tych samych ROI i metrykach. Z góry ograniczony protokół, bez wielokrotnego
   wybierania zbiorów i klas pod wynik CNR.
5. **PUCPR pozostawić zamknięte** do końcowej oceny wybranego, zamrożonego
   modelu. Nie mieszać jego klatek ani innych widoków CNR z external train.
   Przed przyjęciem źródła potwierdzić wykluczenie tych fizycznych obiektów;
   brak identycznych hashów zdjęć nie wyklucza powiązanej sceny.

Nie ufać losowemu splitowi AVM. Gdy źródło ma służyć wyłącznie jako dodatkowy
train, można trzymać cały sprawdzony korpus w jednej roli treningowej i nie
raportować jego wewnętrznego „testu”. Nadal potrzebne jest potwierdzone
pochodzenie wykluczające CNR/PUCPR. Gdy potrzebna jest dodatkowa walidacja
źródła, wymaga ona prawdziwych grup; klastry podobieństwa nie są potwierdzonymi IDs.

Alternatywa po uzyskaniu praw RLMD/PSV: najpierw selekcja zgodnych **klas**
i kontrola małej próbki konturów, nie poprawianie wszystkich obrazów od zera.
Nie włączać obszarów całych miejsc, przerw w dashed, nominalnych obwiedni
zużytych oznaczeń ani niezweryfikowanych parking-lot regionów jako farby.
Mogą ewentualnie służyć osobnemu, jawnie innemu zadaniu pomocniczemu;
nie są pełnowartościowym GT widocznej farby Dataset A.

**Importer treningowy nie powstał, bo warunek potwierdzonych danych i praw
nie został spełniony.** Nie rozbudowywano pipeline na podstawie niepewnych
etykiet. W tym etapie nie potrzeba nowych ręcznych anotacji ani kolejnego
treningu. Najbardziej oszczędny następny krok to uzyskanie warunków AVM SS,
zamiast zamawiania tysięcy nowych poligonów.

## 6. Artefakty i kontrola

W katalogu audytu: `dataset-registry.json` zawiera wszystkie statusy, liczby
z rozróżnieniem lokalnych pomiarów/relacji autorów oraz proponowane mapowania
niezatwierdzone do uczenia. `samples/{avm-ss,rlmd,ceymo}/` zawiera oryginały,
maski, metadane i porównania oryginał / oryginalne GT / overlay.
`mask-audit.json` w każdym katalogu opisuje paletę i konkretne niezgodności.
`evidence/` zachowuje oryginalne README, CSV, odpowiedzi konektora, PDF-y
PSV i metadane udanych/nieudanych pobrań. Wydania danych nie pobierano masowo.

Weryfikacja objęła odczyt 20 par, zgodność wymiarów/dtype, paletę i liczniki,
CRC pobranej próbki ZIP oraz źródłowe SHA256. Obejrzano overview AVM i RLMD,
zbliżenie dashed RLMD i rzeczywistą obwiednię junction box CeyMo. Przykłady
dobrano do kontroli semantyki, nie do wyliczania metryk modelu ani estymacji
jakości całego datasetu. Nie powtarzano testów/Black/Flake8/Mypy niezmienionego
kodu. Sprawdzono `git diff --check`, zachowanie baseline i poprzednich wejść.
