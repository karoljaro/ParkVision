# Farba parkingowa: pierwszy krok CV + ML

**Wykorzystanie AVM SS 2026-11-01:** wykonano [import i trening AVM → UFPR](/home/karoljaron/Projects/ParkVision/research/paint-segmentation/avm-experiment.md).
6763 pary sprawdzone w całości; 3 epoki pretrainingu i osobny fine-tuning.
Na identycznych ROI CNR foreground mIoU **18,53% → 18,94%**, ale IoU paint
spadło do **15,18%**, a recall do **15,46%**. Wynik mieszany, bez promowania modelu.
AVM pozostaje osobnym źródłem badawczym, bez sztucznego curb i bez używania
losowego splitu do niezależnej walidacji. Nie potwierdzono formalnej licencji danych;
podstawę lokalnego akademickiego wykorzystania opisano w raporcie.
PUCPR, wcześniejsze checkpointy, anotacje i silnik zachowano. Starsze wpisy poniżej
opisują poprzednie etapy.

**Audyt gotowych masek 2026-11-01:** [AVM SS, AVP-SLAM Liu/Yao, PSV, RLMD
i źródła dodatkowe](ready-datasets-audit.md). Sprawdzono 20 rzeczywistych par
obraz–maska. AVM SS jest najbliższym kandydatem; nadal wymaga potwierdzenia
praw do danych i grupowania. RLMD/PSV/CeyMo zawierają także etykiety nominalnych
obszarów z asfaltem, nie tylko widocznej farby. Brak nowego treningu/importu.
Baseline v1 (18,53% CNR) i wszystkie wcześniejsze checkpointy zachowane;
SHA256 best.pt oraz kontrola 107 artefaktów zapisane w osobnym katalogu audytu.

**Pierwszy trening 2026-11-01:** ukończono [U-Net ResNet50 z ImageNet
i niezależną walidacją CNR](training-results.md). Zamrożono podział UFPR train /
CNR val / PUCPR test. Early stopping po 23 epokach, najlepsza epoka 8:
IoU paint **22,28%**, curb **14,77%**; dużo fałszywych alarmów, wynik
eksperymentalny. Zachowano checkpointy, przebieg i wszystkie predykcje CNR.
PUCPR poza oceną i strojeniem. [Pipeline i odtworzenie](training.md).
Bez zmian silnika, dawnych anotacji i artefaktów oraz bez commita.
Starsze wpisy poniżej opisują stan sprzed treningu.

**Najnowszy stan 2026-11-01:** [odzyskano pochodzenie wcześniejszych PKLot/CNR
i przygotowano pulę z trzech parkingów](dataset-a-provenance.md): **8 obrazów,
150 reviewed poligonów, 106 037 nadzorowanych pikseli**. Potwierdzono pełną
zgodność pobranych JPEG-ów oraz deklaracje CC BY 4.0 / ODbL 1.0 dla danych.
Nowy eksport zachowuje identyczne maski. Przygotowano osobną propozycję podziału
całymi parkingami (UFPR/CNR/PUCPR, 2/3/3); aktywna pula nadal pilot, bez treningu.
Poniższe wcześniejsze wpisy zachowują historyczny stan przed tą weryfikacją.

**Najnowszy etap 2026-11-01:** po ręcznej korekcie użytkownika wykonano
[eksport v2 trzeciej partii i porównanie masek](dataset-a-batch03-review.md).
265 pikseli asfaltu zmieniono z curb na background, 61 niepewnych paint na ignore;
mały krawężnik zachowano. Wskazane uwagi zostały zamknięte. Bieżąca pula ma
11 masek, 198 poligonów i 386 413 pikseli nadzorowanych. Całość nadal pilot;
jedyny parking z potwierdzonym oryginalnym pochodzeniem plików to PUCPR.

**Przygotowanie 2026-11-01:** przygotowano [trzecią partię: 3 pełne kadry
PUCPR pobrane bezpośrednio od autorów PKLot](dataset-a-batch03-pucpr.md),
z oryginalnymi ścieżkami, datami, hashami i licencją danych CC BY 4.0.
Nowe obrazy czekają na ręczną anotację; wszystkie trzy należą do jednego parkingu.
Łącznie wybrano 11 obrazów: 8 reviewed + 3 nieoznaczone. Treningu nie uruchamiano.

**Wcześniej 2026-11-01:** wykonano [eksport ręcznie zatwierdzonej drugiej
partii i przegląd nakładek](dataset-a-batch02-review.md): 4 nowe maski,
107 poligonów użytkownika. Razem ze starym pilotem jest 8 obrazów, 136 poligonów
i 349 094 oznaczone piksele. Całość pozostaje w pilot; brak niezależnego podziału
i potwierdzonej proweniencji/licencji lokalnych plików. AVM nadal poza A.
[Przygotowanie partii i audyt AVM z 2026-10-30](dataset-a-expansion.md) oraz
poniższe starsze sekcje zachowują swój historyczny zakres.

**Aktualizacja 2026-10-30:** po ręcznym przeglądzie pilota wyeksportowano cztery
maski i wykonano osobny [techniczny test U-Net](overfit.md). Poniższy opis audytu
i wyniki pierwszego etapu zachowują swój wcześniejszy stan. Silnik pozostaje bez zmian.

Stan i weryfikacja źródeł: **2026-10-30**. To działający workflow przygotowania
danych, jeszcze bez treningu, predykcji i metryk modelu. Nie ma lokalnego GT farby.
Wszystkie polecenia poniżej zakładają katalog główny repozytorium ParkVision.

## Audyt przed zmianami

- Właściwe repo: `/home/karoljaron/Projects/ParkVision`; bieżący workspace
  `Dokumenty/ChatGPT/System parkingowy` zawiera artefakty, nie jest repozytorium Git.
- Czyste `git status --short`, HEAD `7f231302a09d0f38bd1d44acb029d8cc31561953`
  (CLAHE). Brak `AGENTS.md` w repo i sprawdzonych katalogach nadrzędnych.
- Repo zawiera `LICENSE`, `.gitignore`, konfigurację edytora i `apps/engine`.
  Silnik: `src/parkvision/{capture,detection,domain}`, `scripts`, `tests`, `docs`,
  `samples` (9 JPEG), `.venv` oraz ignorowany `tmp` z diagnostyką dla 9 obrazów.
  `markings.json` i `segments.json` zawierają geometrie, **nie maski farby**.
- Przejrzano również pliki ignorowane poza środowiskiem i cache: brak anotatora,
  masek pikselowych, zbiorów A/B, podziału train/val/test, wag modeli lub treningu.
  Obrazy i nakładki w drugim workspace są artefaktami wcześniejszych eksperymentów.
- Python 3.14.8, NumPy 2.5.4, OpenCV 5.0.0, pytest 9.1.1. Bazowo **755 testów**.
  `parkvision-engine` wymaga tylko NumPy/OpenCV. Nie zmieniono jego zależności.
- `apps/engine/docs/clahe-experiment.md` potwierdza diagnostyczny, opcjonalny CLAHE;
  nie włączono go do danych ani przetwarzania tej próby.

Nowy kod jest wyłącznie w `research/paint-segmentation`. Nie importuje `parkvision`.
Nie zmienia LSD → grupowanie → dopasowanie → parowanie → topologia → wsparcie,
`OpenCVSlotDetector`, `detect()`, `SlotProposal`, `Polygon` ani `SlotProjection`.
Nie utworzono commita.

## Publiczne dane: co rzeczywiście oznaczono

Poniższe linki zostały otwarte/sprawdzone. Opis publikacji nie oznacza, że pobrano
archiwum i sprawdzono wszystkie jego etykiety. W tej iteracji nie pobierano zbiorów.

| Zbiór / źródło | Dostępne etykiety i warunki | Licencja / dostęp | Przydatność do farby |
| --- | --- | --- | --- |
| [PKLot, strona autorów](https://web.inf.ufpr.br/vri/databases/parking-lot-database/) i [README](https://www.inf.ufpr.br/lesoliveira/download/pklot-readme.pdf) | JPEG + XML: położenie całego miejsca, occupied/vacant; różna pogoda. „Segmented spaces” to wycięte miejsca. | Autorzy podają **CC BY 4.0** i cytowanie publikacji; link do archiwum na stronie. | Dobre źródło obrazów do ręcznej anotacji; **brak masek farby**. UFPR04/05 to dwa widoki tego samego parkingu: jeden split. |
| [CNRPark-EXT](http://cnrpark.it/), [publikacja autorów](https://falchi.isti.cnr.it/Draft/2017-ESWA-Draft.pdf) | Obrazy kamer, obszary miejsc i zajętość, także deszcz/cienie; nie kontury widocznej farby. | Strona źródłowa zwróciła timeout; **nie potwierdzono licencji u źródła ani warunków w oryginalnym archiwum**. | Ręczna anotacja po potwierdzeniu licencji; 9 kamer jednego parkingu nie daje 9 niezależnych parkingów. |
| [Tongji ps2.0 / DeepPS, autorzy](https://cslinzhang.github.io/deepps/) | Surround-view; marking points, powiązania miejsc i narzędzie anotacji punktów. | Strona zawiera link Drive; nie znaleziono tam jednoznacznej licencji danych. | Punkty/geometria nie wyznaczają szerokości farby. Nie rasteryzować ich jako GT. Inna domena niż kamera stała. |
| [AVM SS, repo autorów](https://github.com/Engro39/avm_dataset) | Rzeczywiste kolorowe maski 320×160: free space, **marker**, vehicle, other objects; ego ignorowane. Opis obejmuje outdoor-rainy i odbicia w garażu. | Link do folderu Drive działa jako strona; repo nie zawiera pliku LICENSE ani jasnych warunków danych. Nie sprawdzono zawartości archiwum. | Najbliższy kandydat: są maski marker. Potrzebne pozwolenie/licencja oraz identyfikatory parkingów/sekwencji. Gotowy train/test nie potwierdza braku przecieku. „Other” łączy krawężnik ze ścianą/słupem, nie jest GT osobnej klasy curb. |
| [PSV / publikacja autorów](https://arxiv.org/abs/1804.07027), [VPS-Net](https://github.com/weili1457355863/VPS-Net) | Publikacja opisuje segmentację oznaczeń; VPS-Net udostępnia też adnotacje **wolnych miejsc**, co jest innym zadaniem. | Nie potwierdzono pobrania oryginalnych masek, licencji ani identyfikatorów podziału. | Kandydat do sprawdzenia, **nie potwierdzony gotowy dataset farby**. Nie utożsamiać wszystkich wersji PSV. |
| [APKLOT, repo autorów](https://github.com/langheran/APKLOT) | Poligony LabelMe obszarów parkingowych/bloków na obrazach satelitarnych i kamerowych. | Autorzy podają MIT oraz dodatkowe warunki zdjęć Google Maps. | Otwarty format, ale **obszar miejsca/bloku nie jest farbą**; nie importować tych poligonów jako paint. |
| [CRPS-D, repo autorów](https://github.com/zzh362/CRPS-D) | Nowszy zbiór detekcji miejsc, różna pogoda; linki do danych i anotatora. | Nie znaleziono jasnej licencji danych; nie sprawdzono archiwum ani masek. | Nie ma podstaw, by traktować opis slot detection jako potwierdzenie masek farby. |

Wniosek: **nie ma obecnie zweryfikowanego, lokalnie dostępnego zestawu masek z
licencją i wiarygodnym podziałem grupowym**. AVM wymaga dalszej weryfikacji; nie
uruchamiamy treningu na podstawie samego README. Nie oceniano porad prawnych ani
nie przenoszono licencji kodu na obrazy.

## Działający wycinek i kontrakt danych

`prepare.py` ma trzy komendy: `init`, `inspect`, `convert`. Jedyny importer obsługuje
rzeczywisty, otwarty format **LabelMe JSON z poligonami**. Nie wymaga LabelMe, Qt,
PyTorch, Transformers ani instalacji silnika do samej konwersji.

| Wartość PNG | Etykieta | Znaczenie |
| --- | --- | --- |
| 0 | `background` | Ręcznie przejrzany obszar negatywny; np. asfalt, cienie, refleksy, pęknięcia, auta. |
| 1 | `paint` | Widoczna farba oznaczeń parkingowych, również numery i kreskowanie w obrębie parkingu. |
| 2 | `curb` | Fizyczna powierzchnia krawężnika, także pomalowanego. Osobny sygnał kontekstowy. |
| 255 | `ignore` | Nieoznaczone, zasłonięte lub niepewne piksele. Wyłączone z loss i metryk. |

Nie rekonstruujemy farby pod autem, w przerwach zużycia ani w odbiciu. W przypadku
pomalowanego krawężnika pierwszeństwo semantyczne ma `curb`; jego kolor można
opisać tagiem. Nie ma nakładających się klas paint/curb. Wątpliwości to `ignore`.

Maska: jednopasmowy **uint8 PNG**, dokładnie H×W wejścia, bez skalowania, CLAHE,
wygładzania i antyaliasingu. Koordynaty `(x,y)` oznaczają środki pikseli, zakres
`0..W-1`, `0..H-1`. Rasteryzacja: zaokrąglenie do najbliższego środka (`np.rint`,
remisy do parzystego), wypełnienie wraz z brzegiem przez `cv2.fillPoly/LINE_8`.
Rysuj proste, nieprzecinające się poligony. Bardzo cienkie/małe fragmenty trzeba
sprawdzić w powiększeniu; wielokąt o zerowym polu po zaokrągleniu jest odrzucany.

Poza poligonami domyślnie jest **255**, nigdy domniemane tło. Kolejność obiektów
w LabelMe nie wpływa na wynik: background → paint/curb → ignore. Nakładanie
paint/curb poza ignore powoduje błąd. Negatywne ROI można zakreślić dużym
poligonem background, a w jego wnętrzu dokładnie obrysować farbę i krawężnik.
Zużyte fragmenty obrysuj osobno, bez łączenia ich przez widoczny asfalt.

Globalna flaga `reviewed: true` jest ręcznym potwierdzeniem przeglądu **oznaczonych
ROI**, nie twierdzeniem o kompletności całego obrazu. Puste adnotacje, inne
etykiety/typy kształtów, punkty poza obrazem, błędne wymiary, inny `imagePath`
i osadzone `imageData` są odrzucane. Obrazy odczytujemy przez `cv2.IMREAD_COLOR`,
tak jak batch CV; weryfikuj orientację w GUI i overlay. Nie obracaj obrazów w trakcie.

Manifest v1 jest tworzony przez `init`. Ścieżki odnoszą się do jego katalogu.
Każdy rekord ma `id`, `image`, `annotation`, `image_sha256`, `parking_id`,
`camera_id`, `group_id`, `split`, `surface`, `conditions`, `source`, `license`.
`surface`: wet/dry/unknown; warunki np. shadow/reflection/worn_paint/occlusion.
`camera_id` musi być globalnie unikatowe, np. `cnrpark:camera3`.
`group_id` łączy sekwencję, sesję oraz wszystkie kadry pochodne/augmentacje.

- Domyślnie `split=pilot`, tożsamość i licencja unknown. Taki pilot nie jest
  zbiorem do raportowania generalizacji. Nie zgadujemy pochodzenia z nazwy pliku.
- Dla train/val/test wymagane są uzupełnione identyfikatory, źródło i licencja.
  Są to deklaracje kuratora; program nie ustala sam praw do danych.
- Ten sam parking, kamera lub group_id nie mogą trafić do różnych splitów,
  także między pilot a train/val/test. Przenoś całą grupę razem.
- Kontrola ścieżek, SHA256 oryginału i identycznych **zdekodowanych pikseli**
  wykrywa zmienione obrazy i dokładne duplikaty. Nie wykrywa wszystkich podobnych
  JPEG-ów lub kadrów: rzetelne ID/proweniencja i ręczny przegląd są konieczne.
  Przy braku pewności zachowaj całość w pilot; nie losuj splitów klatkami.

`convert` sprawdza cały manifest przed zapisem, wymaga nowego katalogu wyjściowego,
zapisuje `<id>.mask.png`, `<id>.overlay.png` oraz `dataset.json`: klasy, ścieżki,
identyfikatory splitów, hashe źródeł/anotacji/masek, wymiary, wersje bibliotek,
liczby pikseli klas. Sprawdza identyczność maski po ponownym odczycie PNG.
Overlay: paint magenta, curb pomarańczowy, background niebieski; ignore zachowuje
oryginalne piksele. **Liczby pikseli to statystyki adnotacji, nie wyniki modelu**.

## Najbliższy eksperyment: ręczne ROI na czterech obrazach

Gotowy pilot poza repozytorium:
`/home/karoljaron/Dokumenty/ChatGPT/System parkingowy/paint-segmentation-pilot-2026-10-30/`.
Zawiera 4 kopie JPEG identyczne bajtowo, manifest i `overview.jpg`. Nie zawiera
adnotacji ani masek. Obrazy: `04-cnrext-camera3-wet`, `parking`,
`02-empty-parking-shadows`, `01-pklot-ufpr05-empty` (wymiary w manifeście inspekcji).
UFPR05 pozostaje surface=unknown: sama nazwa nie rozstrzyga stanu nawierzchni.

Obecne środowisko silnika wystarcza do uruchomienia narzędzia bez nowych instalacji:

```bash
apps/engine/.venv/bin/python research/paint-segmentation/prepare.py inspect \
  '/home/karoljaron/Dokumenty/ChatGPT/System parkingowy/paint-segmentation-pilot-2026-10-30/manifest.json'
```

Aby odtworzyć pilot w **nowym** katalogu:

```bash
apps/engine/.venv/bin/python research/paint-segmentation/prepare.py init \
  --images apps/engine/samples/04-cnrext-camera3-wet.jpg \
    apps/engine/samples/parking.jpg apps/engine/samples/02-empty-parking-shadows.jpg \
    apps/engine/samples/01-pklot-ufpr05-empty.jpg \
  --output /tmp/parkvision-paint-pilot
```

Izolowane środowisko narzędzi danych można odtworzyć tak (Python 3.14.8,
przypięte wersje bez zależności od pakietu engine):

```bash
python3.14 -m venv /tmp/parkvision-paint-data-env
/tmp/parkvision-paint-data-env/bin/python -m pip install \
  -r research/paint-segmentation/requirements.txt
```

Do ręcznej anotacji użyj [LabelMe 5.9.1](https://github.com/wkentaro/labelme/tree/v5.9.1)
(Qt, [GPL-3.0](https://github.com/wkentaro/labelme/blob/v5.9.1/LICENSE)), **osobno od
silnika i narzędzi danych**. Wariant 5.9.1 przypięto dla powtarzalnego formatu/CLI;
nie instalowano ani nie testowano GUI w tej sesji. Przykład z Pythonem 3.12:

```bash
python3.12 -m venv /tmp/parkvision-labelme-env
/tmp/parkvision-labelme-env/bin/python -m pip install labelme==5.9.1
/tmp/parkvision-labelme-env/bin/labelme \
  '/home/karoljaron/Dokumenty/ChatGPT/System parkingowy/paint-segmentation-pilot-2026-10-30/images' \
  --nodata --labels background,paint,curb,ignore --flags reviewed
```

1. Create Polygons: na każdym obrazie wybierz 2–3 niewielkie ROI. Zaznacz
   przejrzane tło, dokładne powierzchnie farby, krawężnik i niepewności. Nie zaznaczaj
   wnętrza całego miejsca jako paint. Pozostała część obrazu zostanie ignore.
2. Na wet obejmij słabą farbę i osobno odbicie/mokrą plamę; na shadows linię
   przecinającą cień; na parking wyraźną farbę i spękania; na UFPR05 farbę przy
   krawężniku. Dodaj także ROI negatywne bez farby.
3. Obejrzyj przy 200–400%, sprawdź brak fantomowej farby pod przeszkodami i
   przez przerwy. Dopiero wtedy zaznacz globalny checkbox reviewed i zapisz JSON
   obok obrazu. Zachowaj `imageData=null` (`--nodata`) i poprawny względny imagePath.
4. Uzupełnij metadane warunków i rzeczywistego źródła w manifeście. Pozostaw
   split=pilot do zebrania niezależnych parkingów i potwierdzenia licencji.
5. Po opisaniu wszystkich czterech obrazów uruchom konwersję; brak choć jednej
   adnotacji zakończy ją błędem **bez utworzenia pustych masek**:

```bash
apps/engine/.venv/bin/python research/paint-segmentation/prepare.py convert \
  '/home/karoljaron/Dokumenty/ChatGPT/System parkingowy/paint-segmentation-pilot-2026-10-30/manifest.json' \
  --output /tmp/parkvision-paint-export-v1
```

Obejrzyj każdą pełną nakładkę i cropy 1:1 przy brzegach farby/krawężnika. Do poprawy
edytuj JSON w LabelMe i eksportuj do kolejnego nowego katalogu. Nie używaj nakładki
RGB jako etykiety treningowej. Na razie to test kontraktu i instrukcji anotacji;
cztery ROI/obrazy nie pozwalają wybrać najlepszego modelu.

## U-Net vs SegFormer vs DeepLabV3+: protokół następnej iteracji

**Wszystkie trzy potrzebują tych samych masek**, w tym negatywnych pikseli i ignore.
Żadna architektura nie usuwa potrzeby GT ani uczciwego podziału. Kolejność jest
propozycją badawczą, nie wynikiem jakościowym i nie wynika z możliwości laptopa.

| Kolejność / model | Dlaczego | Implementacja / trening | Inferencja i ryzyko do zmierzenia |
| --- | --- | --- | --- |
| 1. U-Net + ResNet50, wagi ImageNet | Prosty baseline encoder/decoder ze skip connections, szybkie sprawdzenie całego protokołu i brzegów cienkiej farby. | Jeden wspólny loader/loss/ewaluator; `smp.Unet`, 3 klasy. Mniej decyzji wdrożeniowych; nie gwarantuje mniejszej liczby parametrów od każdego konkurenta. | Pełna mapa klas; zachowanie cienkich linii, koszt aktywacji dekodera i kafelkowania. |
| 2. SegFormer MiT-B2; potem B4 jeżeli wyniki uzasadnią | Hierarchiczny transformer daje inny rodzaj kontekstu; interesujący kandydat wobec cieni/refleksów. To hipoteza do testu. | `smp.Segformer` + `mit_b2`, pretraining ImageNet. Ten sam protokół; oddzielne, z góry ustalone strojenie LR i budżet. Wymaga GPU dobranego do wejścia/modelu, nie do laptopa. | Kontrola rekonstrukcji rozdzielczości i cienkich oznaczeń, pamięć/czas przy większych wejściach. Nie zakładać odporności na deszcz po samej nazwie modelu. |
| 3. DeepLabV3+ + ResNet50, ImageNet | ASPP i decoder sprawdzają wpływ kontekstu wieloskalowego, ze wspólnym encoderem jak U-Net. | `smp.DeepLabV3Plus`, jawny output stride 8/16, ten sam loader. Atrous/ASPP dodają decyzje konfiguracji; przy małym batchu kontrolować BatchNorm. | Wpływ output stride na wąską farbę, interpolacja, realny koszt ASPP. |

Architektury: [U-Net](https://arxiv.org/abs/1505.04597),
[SegFormer](https://arxiv.org/abs/2105.15203),
[DeepLabV3+](https://arxiv.org/abs/1802.02611).
[Segmentation Models PyTorch](https://github.com/qubvel-org/segmentation_models.pytorch)
udostępnia wszystkie trzy rodziny; planujemy jedną bibliotekę, zamiast trzech
różnych frameworków. Jeszcze jej nie instalowano i nie tworzono pustego train.py.
Przy pierwszym treningu przypiąć Python/PyTorch/SMP/CUDA, wersję i hash wag oraz
sprawdzić licencje wybranych checkpointów osobno od licencji biblioteki.

Warunki rozpoczęcia i wspólny protokół:

1. Zbiór **A**: ręcznie zatwierdzone maski z wielu parkingów, różne kamery,
   mokro/sucho, cień/refleks/zużycie oraz trudne negatywy. Minimum trzech niezależnych
   grup parkingowych umożliwia jedynie formalne train/val/test; potrzeba większej
   różnorodności do wiarygodnych wniosków. Zamrozić manifest przed treningiem;
   całe parkingi, powiązane kamery, sesje i pochodne obrazy muszą być w jednym splicie.
   Pilot opisany powyżej nie spełnia jeszcze tych warunków.
2. Docelowy zbiór **B**: przyszła adaptacja do konkretnego parkingu. Osobny protokół,
   z rozłącznymi kamerami/sesjami i nietkniętym testem; wynik adaptacji B nie jest
   dowodem generalizacji na nowy parking. Nie mieszać B z zamrożonym testem A.
3. Najpierw test techniczny na kilku oznaczonych treningowych ROI (czy loss maleje),
   potem U-Net na właściwym podziale. Dopiero po kontroli anotacji i walidatora
   uruchomić SegFormer i DeepLabV3+. Nie raportować overfitu jako wyniku walidacji.
4. Wspólne cropy np. 512×512 w natywnej skali + padding ignore, losowanie wyłącznie
   z train. Część cropów z farbą, część trudnych negatywów; sampling/augmentacje
   zapisane w konfiguracji. Bez CLAHE na starcie. Walidacja pełnych obrazów przez
   ustalone kafelki z overlapem i uśrednianiem prawdopodobieństw; bez losowych cropów
   oceny. Uważać, by downsampling nie usuwał cienkiej farby.
5. Trzy klasy softmax: background/paint/curb; CE z ignore_index=255 + Dice farby
   i krawężnika z wyłączeniem ignore. Wagi klas wyłącznie z train. Ten sam limit
   aktualizacji, te same dane, augmentacje, zestaw seedów (np. 17/42/2026) i budżet
   strojenia dla każdego modelu; konfiguracja LR osobna, lecz jawna. Wybór checkpointu
   i ewentualnego progu paint tylko na val. Test otworzyć po zamrożeniu decyzji.
6. IoU/Dice **per klasa**, precision/recall i FP piksele oraz FPR per klasa;
   w szczególności paint na curb, refleksach/cieniach i tle. Raportować wet/dry
   osobno, unknown osobno, liczbę parkingów/obrazów/pikseli ocenianych i ignore.
   Macierz pomyłek i średnie per parking obok sum pikseli. Puste klasy: N/A przy
   zerowym mianowniku, nie sztuczne 1.0; uwzględnić FP na negatywnych obrazach.
   Przy ROI metryka dotyczy tylko opisanych ROI, nie całego kadru. Małą próbę
   wyraźnie oznaczyć; nie traktować skorelowanych pikseli jak niezależnych prób.
7. Zmierzyć realny czas treningu/GPU-godziny, peak VRAM, koszt wg użytej stawki,
   medianę/p95 inferencji po rozgrzewce, batch=1, rozdzielczość, preprocessing,
   kafelkowanie i transfer danych. Bez zmyślonych benchmarków/liczb FPS.

Docelowe wyjście niezależnego ML: float32 H×W `P(paint)` w `[0,1]`, dodatkowo
opcjonalne prawdopodobieństwo curb, z identyfikatorem obrazu, hashem modelu,
oryginalnymi wymiarami i opisem odwróconych transformacji/paddingu. Maska progowa
jest pochodna. Porównanie z oryginalnymi geometriami LSD pozostaje diagnostyką;
brak twardego filtrowania i zmian modelu domenowego. Niski P(paint) na krawężniku
nie jest dowodem, że geometria jest błędna.

## Sprawdzenia tej iteracji

```bash
apps/engine/.venv/bin/python -m pytest -q apps/engine/tests research/paint-segmentation
apps/engine/.venv/bin/python -m flake8 --jobs 1 --config apps/engine/.flake8 \
  apps/engine/src apps/engine/tests apps/engine/scripts research/paint-segmentation
apps/engine/.venv/bin/python -m black --check research/paint-segmentation/prepare.py
apps/engine/.venv/bin/python -m black --check research/paint-segmentation/test_prepare.py
```

Testy narzędzia używają **wyłącznie sztucznych rastrów testowych**, nie etykiet do
uczenia. Sprawdzają dokładne piksele klas i ignore, wymiary, formaty, brak mutacji,
lossless PNG, hashe, błędne poligony/etykiety, reviewed, brak adnotacji, kolizje
train/val/test, duplikaty, niezmienność źródeł i uruchomienie CLI z innego katalogu.
Mypy jest dostępne tylko w zastanym katalogu `/tmp/parkvision-clahe-mypy`, bez
zmiany zależności silnika:

```bash
PYTHONPATH=/tmp/parkvision-clahe-mypy apps/engine/.venv/bin/python -m mypy \
  --follow-imports=silent research/paint-segmentation/prepare.py \
  research/paint-segmentation/test_prepare.py
```

Końcowy wynik: **807 passed** (755 silnika + 52 narzędzia danych, 1.57 s).
Pełny Flake8 dla silnika i research przechodzi; Black dla obu nowych plików Python
przechodzi; Mypy nie zgłasza błędów w tych 2 plikach. `git diff --exit-code --
apps/engine` jest pusty, HEAD pozostał bez zmian. Jedyny nowy katalog repo to
`research/paint-segmentation` (4 pliki źródłowe/dokumentacyjne, bez wag i danych).

Na rzeczywistym pilocie uruchomiono init i inspect: odczytano wszystkie 4 obrazy,
potwierdzono identyczność bajtową kopii. Próba convert bez JSON-ów została poprawnie
odrzucona i nie utworzyła katalogu masek. `inspection.json` i `workflow-check.json`
w katalogu pilota zapisują ten stan. Nie wygenerowano rzeczywistego GT.

Nie można jeszcze twierdzić, że ML poprawia pokrycie farby, redukuje FP LSD,
działa lepiej na mokrym asfalcie lub że któraś architektura wygrywa. Nie ma
treningu, checkpointu, predykcji ani benchmarku segmentacji. Następne działanie
to ręczna anotacja/podgląd i zebranie niezależnych, legalnie dostępnych grup danych.
