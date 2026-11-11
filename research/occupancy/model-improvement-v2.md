# Model Improvement v2

## Obserwator

`observe.py` obsługuje dwa jawne backendy: `--model acpds-rcnn` (domyślny)
i `--model convnext-roi4`. Detektor Faster R-CNN i reguła D są wspólne.
Progi 0.10/0.90, operator unreadable, geometria i temporal v1 pozostają te same.
Backend ConvNeXt używa bez kopiowania `training_v1.data.encode_slot`,
`training_v1.model.tensor_input` i `build_model`. Import Torch jest odroczony;
odczyt błędnych klatek i replay nie wymagają zależności treningowych.
Pełne zależności inferencji są w `training_v1/requirements.txt`.

Wagi są lokalne, poza repo. Dla ConvNeXt JSON assets zawiera dokładnie
`detector_weights` i `per_slot_weights`. Dla RCNN obowiązuje dotychczasowy
plik z dodatkowym `acpds_pooler`. Ścieżki względne są liczone od pliku assets.
Nie ma pobierania ani fallbacku. Niezgodne wagi dają `inference_error`,
unknown dla całej klatki i kod wyjścia 1, zgodnie z kontraktem v1.

Przypięty ConvNeXt:

- SHA-256: `5f5672cfdea28cdb53942675d748d5f9d6c3abda5bc7fc18abe209d1d1331a23`.
- Architektura `convnext_tiny_imagenet1k_v1_rgb_roi4`, epoka 13,
  klasy w kolejności `[free, occupied]`, stem 4 kanały, head 2 klasy.
- RGB+ROI letterbox 224, kontekst 25%, normalizacja RGB ImageNet,
  maska 0/1. Sprawdzane są hash, metadane i ścisła zgodność state dict.
- Hash manifestu uczenia:
  `30e7adc215dbdbf9501b27ea3d554857716eb5ffdd3373087e08d25463c9ec74`.

Zachowany `schema_version=1` i wszystkie dotychczasowe pola. Jedyne rozszerzenie
to `provenance.selected_model`; tożsamość faktycznie załadowanych wag jest
w `provenance.models.per_slot`. Przy błędzie ładowania models pozostaje null.
Istniejący `baseline_id` identyfikuje niezmienioną bazową politykę obserwacji;
wybrany klasyfikator trzeba odczytywać z metadanych modelu. ConvNeXt dodaje
tam architekturę, epokę, recipe, klasy i hash manifestu uczenia.

Przykłady z repo, po ustawieniu ścieżek na własną zatwierdzoną konfigurację,
klatkę oraz jej rzeczywisty czas pozyskania:

```bash
python=research/paint-segmentation/.venv/bin/python
artifacts='/home/karoljaron/Dokumenty/ChatGPT/System parkingowy/model-improvement-v2-2026-11-09-v1'

"$python" research/occupancy/observe.py \
  --model acpds-rcnn --assets "$artifacts/assets-rcnn.json" \
  --configuration "$configuration_path" --camera-id "$camera_id" \
  --image "$image_path" --captured-at "$capture_time" \
  --session-id "$session_id" --max-age-seconds 60 --view-confirmed \
  --device cuda --output "$new_rcnn_result_path"

"$python" research/occupancy/observe.py \
  --model convnext-roi4 --assets "$artifacts/assets-convnext.json" \
  --configuration "$configuration_path" --camera-id "$camera_id" \
  --image "$image_path" --captured-at "$capture_time" \
  --session-id "$session_id" --max-age-seconds 60 --view-confirmed \
  --device cuda --output "$new_convnext_result_path"
```

`--device cuda` wymaga dostępnej karty. `--device cpu` również jest obsługiwane.
Dla odtworzenia historycznego dodaj `--evaluated-at` zgodnie z wcześniejszym
kontraktem. Każde wyjście musi wskazywać nowy plik.

## Mała próba integracyjna

`integration_smoke.py` wybiera pierwszą klatkę ACPDS i pierwszą CNR camera1,
po cztery pierwsze źródłowe ROI. Porównuje nowy adapter z zamrożonym kodem
ConvNeXt oraz RCNN ze starym `baseline_models.py`. Granica różnicy score
wynosi 1e-5; sprawdza też stany progowe i identyczność detekcji.
Tworzy osobne konfiguracje diagnostyczne, bez edycji R1/R2/R3.
Zegar próby jest jawnie syntetyczny; nie oznacza czasu wykonania zdjęcia.
To sprawdzenie połączenia elementów, bez metryk jakości i doboru parametrów.

```bash
"$python" -m research.occupancy.integration_smoke \
  --protocol '/home/karoljaron/Dokumenty/ChatGPT/System parkingowy/dataset-model-improvement-2026-11-08-v1/final-comparison-v1/protocol.json' \
  --rcnn-assets "$artifacts/assets-rcnn.json" \
  --convnext-assets "$artifacts/assets-convnext.json" \
  --device cuda --output "$artifacts/new-integration-smoke"
```

## Widoczność

Osobny moduł [visibility_v2](visibility_v2/README.md) przygotowuje manifest,
sprawdza pochodzenie, deduplikuje i udostępnia małą kolejkę anotacji.
Nie trenuje modelu i nie wprowadza automatycznej bramki do obserwatora.
Softmax occupancy nie jest miarą widoczności.

Pomysł odejmowania masek aut sąsiadów pozostaje w backlogu: rozdzielenie
instancji samochodów, przypisanie do miejsc i traktowanie zasłoniętych pikseli
jako nieznanych wymagają odrębnego projektu. Nie wdrażamy tego w v2.
