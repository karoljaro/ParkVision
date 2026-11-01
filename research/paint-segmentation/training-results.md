# Pierwszy trening: wynik na niezależnym CNR

2026-11-01. Ukończono eksperyment U-Net ResNet50 z wagami ImageNet,
augmentacją i niezależną walidacją. **Pipeline działa, lecz wynik modelu jest
słaby: średnie IoU paint/curb na oznaczonych ROI CNR wynosi 18,53%.**
Nie uruchomiono oceny PUCPR, nie zmieniono silnika CV i nie wykonano commita.

## Zakres wykonania

Zamrożono osobne kopie 8 obrazów, ich zatwierdzonych JSON-ów i masek wraz
z manifestem i SHA256. UFPR04/05 pozostają razem w train: **2 obrazy,
48 576 nadzorowanych pikseli**. CNR: **3 kamery jednego innego parkingu,
20 142 nadzorowane piksele**, wyłącznie walidacja. PUCPR: 3 obrazy trzeciego
parkingu, integralność sprawdzona, poza loaderem modelu i metrykami.

Użyto oficjalnych wag **ImageNet1K V2** ResNet50; decoder był inicjalizowany
losowo, seed=42. Brak checkpointu overfit. Natywne kafle 256×256: 10 train,
17 val. Augmentacja geometryczna bez interpolacji etykiet oraz kontrast/jasność
tylko w train. Ważone CE + Dice, ignore=255 wykluczone z loss i gradientów.
AdamW, LR encoder=0,0001, decoder=0,001, batch=4, statystyki BN encodera
zamrożone. Szczegóły i polecenia są w [opisie pipeline](training.md).

Wykonanie na RTX 4070 zakończyło się przez **early stopping po 23 epokach
(69 kroków optymalizatora)**. Najlepszy checkpoint pochodzi z **epoki 8**,
wybranej według z góry zapisanego średniego IoU paint/curb CNR. Limit 100 epok,
patience=15, min_delta=0,001. Czas od startu pętli do eksportu wyników: 131,5 s.
Bez przeszukiwania parametrów. Przy pierwszej próbie poprawiono wyłącznie
implementację deterministycznej redukcji straty CUDA, przed pierwszą aktualizacją
wag; oba wykonania i protokoły zachowano. Nie zmieniono podziału ani hiperparametrów.

## CNR — najlepszy checkpoint, wszystkie reviewed ROI

Procenty z macierzy pomyłek łączącej 20 142 oznaczone piksele trzech obrazów.
Ignore nie jest tłem i nie uczestniczy w mianownikach.

| Klasa | IoU | Dice | Precision | Recall | Piksele GT |
| --- | ---: | ---: | ---: | ---: | ---: |
| background | 39,94% | 57,08% | 95,15% | 40,77% | 15 967 |
| paint | **22,28%** | **36,45%** | **25,65%** | **62,95%** | 1811 |
| curb | **14,77%** | **25,74%** | **16,31%** | **61,08%** | 2364 |

Średnie IoU foreground = **18,53%**, średnia dwóch IoU z macierzy łącznej;
nie jest to średnia trzech obrazów. Przed uczeniem decoder był losowy i ta
metryka wynosiła 5,47%. Loss CNR dla wybranego checkpointu to 2,715, wobec
2,431 przed uczeniem; wybór następował według zadeklarowanego IoU, nie loss.

| Obraz CNR | IoU paint | IoU curb | Średnie IoU foreground |
| --- | ---: | ---: | ---: |
| camera3-wet | 35,37% | 23,77% | 29,57% |
| camera8 | 19,80% | 32,28% | 26,04% |
| camera1-occupied | 25,00% | 5,96% | 15,48% |

Macierz pomyłek, wiersze GT i kolumny predykcji w kolejności background/paint/curb:

```text
6510  2562  6895
 155  1140   516
 177   743  1444
```

Dominują fałszywe alarmy: **3305 FP paint i 7411 FP curb**. Na camera1 model
oznaczył jako curb 5321 pikseli ręcznie opisanego tła. Jest to widoczne na
zapisanych predykcjach jako duży pomarańczowy fragment asfaltu. Na camera8
część asfaltu nad farbą dostaje paint, a granice paint/curb są mylone. Model
odzyskuje część oznaczeń, ale przy tej precision nie nadaje się do wdrożenia.
Spadek loss na train do 0,110 nie przełożył się na dalszą poprawę CNR.

To wynik **wybranego na CNR** checkpointu, a nie niezależnego testu końcowego.
Próba obejmuje dwa obrazy uczące jednego parkingu i nieliczne ręcznie wybrane
ROI. Nie obejmuje wiarygodnej oceny całych scen ani dowodu jakości na nowych
parkingach jako populacji. Przyczyny błędów mogą obejmować niedostateczną
różnorodność train i różnice obrazu/GT; jeden eksperyment ich nie rozdziela.

## Rzeczywiste predykcje i kontrola

Wynik:
`/home/karoljaron/Dokumenty/ChatGPT/System parkingowy/paint-segmentation-training-2026-11-01/unet-resnet50-imagenet-v2/`.

- `validation/overview.jpg`: trzy obrazy CNR, kolumny oryginał / GT / model w ROI.
- `validation/*.comparison.png`: te same porównania w pełnej rozdzielczości.
- `validation/tiles/*.comparison.png`: wszystkie 17 kafli, powiększenie 2×.
- `validation/*.prediction.png`: indeksowane predykcje na przetworzonych kaflach;
  `*.roi-prediction.png`: tylko oceniane ROI, reszta 255.
- `validation/*.errors.png`: błędne oceniane piksele zaznaczone czerwono.
- `result.json`: wszystkie metryki i macierze, także per obraz; `history.json`
  i `learning-curves.png`: przebieg 0–23.
- `best.pt`, `last.pt`: model, optymalizator, konfiguracja i stan RNG.

Metryki przeliczono **niezależnie w NumPy z zapisanych PNG predykcji ROI**:
zgodność macierzy i wartości dla każdego obrazu. Powtórne wczytanie checkpointu
do świeżego modelu daje identyczne predykcje. Zamrożone dane, osiem wejść puli
pilot ze zweryfikowanymi źródłami oraz **104 chronione pliki wcześniejszych
partii pozostały niezmienione**. Obejrzano overview, trzy powiększenia (po jednym
na kamerę) i krzywe. Wszystkie ROI podlegają kontroli ilościowej, nie tylko
pokazane wycinki. Nie oceniano modelu na PUCPR.

Walidacja kodu: **81 testów research**, w tym rzeczywista deterministyczna CUDA,
oraz **755 testów silnika** przeszło. Black, Flake8 i Mypy nowych plików,
`pip check` oraz `git diff --check` przeszły. Kod i zależności silnika bez zmian;
HEAD `552808e84be3a4e1445d2841253169ae937763e7`. Całość bez commita.

Kolejny krok danych: zwiększyć różnorodność train i pewnych trudnych negatywów
(asfalt o różnych jasnościach, cienie, odbicia, auta, krawężniki), zachowując
CNR poza uczeniem i PUCPR jako niewykorzystany test. Nie poprawiać GT pod
predykcje modelu ani przenosić zdjęć tych parkingów do train. Ten wynik nie
uzasadnia wdrażania fuzji CV+ML ani wyboru innej architektury na podstawie samego
modelu dopasowanego do dwóch zdjęć.
