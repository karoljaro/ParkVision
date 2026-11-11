# Model Improvement v3 — scene evidence

This is a separate research module. It adds two trained models and
an auxiliary CLI beside the existing occupancy observer. ConvNeXt, ACPDS,
rule D and Temporal v1 remain unchanged. Nothing here rewrites occupancy,
approves parking geometry, infers legal parking, or gates visibility.

Artifacts and the Polish audit report:
`/home/karoljaron/Dokumenty/ChatGPT/System parkingowy/roboflow-expansion-2026-11-09-v1/`.
The work began on 9 November and finished on 10 November 2026.

## Audited source roles

| Original export | Images / unique pixels | Actual annotation role | Used in training |
| --- | ---: | --- | --- |
| parking spaces v1 | 2884 / 2874 | Occupied often vehicle silhouette; Vacant ground region; vehicle masks | No: mixed semantics/provenance and old-source overlaps |
| parking spaces and tree canopies v4 | 400 / 335 | Vehicle/tree masks and broad parking regions | Yes, separated by city |
| Parking v8 | 2518 / 2518 | 10702 RLE and 10 polygon car masks in a mechanical garage | Vehicle masks, train only |
| Parking lot line v2 | 1659 / 1659 | parkinglane is a parking-region polygon, not thin paint | Region channel, train only |
| Free Parking Space Detection v9 | 544 / 544 | Four active region classes; no verified legal/free-slot contract | No automatic remapping |
| vehicle occluded v1 | 5046 / 5046 | Eight object-box classes; no degree-of-occlusion labels | Vehicle boxes; person excluded |

All six original exports declare CC BY 4.0. Publisher links, original metadata
and attribution are retained in `raw-audit/`, `audit/license-evidence.json` and
the report. This records publisher declarations, not an independently verified
copyright chain for every source photograph. No new datasets or weights were
downloaded. The only added dependency was the small official pycocotools wheel.

Source filenames are family hints, not independent-image counts. The aerial
25 parents each have 16 spatial crops, visually consistent with 4x4 tiling;
they are not 16 rotations. The line export explicitly has three training
augmentations per original. Other exports contain filename/grayscale variants
despite incomplete augmentation histories. Exact pixels, source families and
256-bit dHash retrieval are audited separately. Exporter train/valid/test
partitions are not used as proof of physical-site independence.

Only one Parking.v8 ZIP was present under the full local ParkVision dataset
directory. Its SHA-256 is recorded; equality of two v8 archives is unverified.

## Models and limits

**Vehicle instances:** Mask R-CNN ResNet50 FPN v2, RGB min-size 640 / max-size
960. It uses the already local, hash-pinned COCO Faster R-CNN backbone/FPN/box
initialization and a new mask head. Genuine RLE/polygon targets supervise masks;
box-only batches never receive invented rectangular masks. The explicit generic
transport class includes cars, bikes, motorcycles, buses, trucks, trailers,
rickshaws, cng and other vehicles, and excludes people. Parking-state classes
are never merged into it.

Training: 614 real-mask and 512 box-only images; validation: 11 real-mask aerial
images (one crop per Herne parent) and 80 road images from two held-out filename
sessions. All garage images are one train-only facility group; Paderborn is
train, Herne validation. Road session names do not establish physical-site,
camera or date independence. Six epochs, seed 42, batch 2, AdamW 1e-4; selected
epoch 2 by predeclared mean mask AP50 and box AP50. Validation mask AP=34.24,
mask AP50=58.41; road box AP=49.84, AP50=80.64 (COCO percentages).

`runs/vehicle-instance-v3/best.pt`

SHA-256: `c9162dcf0111bd7f1020f88faf5afe5f7ee8b825ad376060896da3a8b2e1616a`.

**Scene maps:** U-Net ResNet50, local ImageNet initialization, RGB letterbox
512 with ImageNet normalization. Independent sigmoid channels: `tree_canopy`
and `annotated_parking_region_candidate`. The latter can span multiple slots;
it is not paint, a single ground footprint or legal parking status. Unlabelled
pixels are not confirmed nonparking. The line corpus supervises only the region
channel, and black/padded pixels are ignored. Ambiguous aerial `parking lot`
regions are excluded. Training: 162 Paderborn crops plus 629 canonical line
images; validation: 11 Herne parent representatives. Ten epochs, seed 43,
batch 4, AdamW 1e-4, selected epoch 10 by mean validation IoU. Tree IoU=62.93%,
region IoU=19.35%. **The region head is too weak for automatic ROI generation.**
These are development validation results on a small domain; there is no new
independent final test and no measured improvement to occupancy false signals.

`runs/canopy-parking-region-v3/best.pt`

SHA-256: `7aea067ea0bc5670ca7fa87ded4c6e41745299e3eca591a3cb9740fe78528b6e`.

Each run retains its manifest hash, protocol, all epoch metrics and the exact
source-code snapshot used for training. CUDA AMP ran on the available RTX 4070;
there was no model selection for laptop CPU performance. New models have not
been promoted into the default observer.

## Run auxiliary evidence

Run the existing observer first, selecting ACPDS or ConvNeXt as documented in
`research/occupancy/README.md`. Then pass its saved v1 observation and the same
configuration to this command. The following concrete command replays the
already prepared integration fixture and creates a new output:

```bash
cd /home/karoljaron/Projekty/ParkVision
PY=research/paint-segmentation/.venv/bin/python
ART='/home/karoljaron/Dokumenty/ChatGPT/System parkingowy/roboflow-expansion-2026-11-09-v1'
"$PY" -m research.scene_v3.evidence \
  --observation "$ART/integration/observation-convnext.json" \
  --configuration "$ART/integration/configuration.json" \
  --checkpoint "$ART/runs/vehicle-instance-v3/best.pt" \
  --sha256 c9162dcf0111bd7f1020f88faf5afe5f7ee8b825ad376060896da3a8b2e1616a \
  --map-checkpoint "$ART/runs/canopy-parking-region-v3/best.pt" \
  --map-sha256 7aea067ea0bc5670ca7fa87ded4c6e41745299e3eca591a3cb9740fe78528b6e \
  --device cuda --output /tmp/parkvision-v3-scene-evidence.json
```

The output and its `-masks` directory must not already exist. Omit both map
arguments for vehicle evidence only. `--device cpu` is available for inference.
Checkpoints must match their supplied SHA, architecture, classes and recipe;
there is no fallback. Input frame/configuration hashes and slot IDs are checked.

Every detected instance is compared with every ROI, using many-to-many image
plane overlap. A vehicle may overlap multiple spaces; there is no forced owner
or subtraction of neighbouring masks. This is evidence for developing the next
assignment model, not a proof of ground contact or actual blocking. Some source
masks may span hidden vehicle parts. No instance does not imply a free space.
Canopy overlap does not by itself imply unreadability. `visibility=unassessed`
and `occupancy_override=null` remain explicit, and original states are preserved.
Probability PNGs are uncalibrated model scores.

The fixture is one new, nontraining source image with two source polygons.
Its IDs are diagnostic, its capture timestamp is explicitly synthetic and its
geometry/parking legality are not operationally approved. It proves functional
compatibility, not accuracy on a target camera.

## Reproduce preparation and training

Install `requirements.txt` only into the research environment if needed.
Raw archives remain untouched. Use fresh output directories for every command.
The commands below show all stages; do not overwrite the recorded results:

```bash
"$PY" -m research.occupancy.training_v1.roboflow \
  --input /home/karoljaron/Datasets/ParkVision/09.11 --output /tmp/pv3-raw
"$PY" -m research.scene_v3.audit --audit /tmp/pv3-raw \
  --previous '/home/karoljaron/Dokumenty/ChatGPT/System parkingowy/dataset-model-improvement-2026-11-08-v1/roboflow-prepared-v1' \
  --output /tmp/pv3-audit
"$PY" -m research.scene_v3.quality --audit /tmp/pv3-raw \
  --output /tmp/pv3-quality.json
```

Both preparers accept repeated `--protected PATH` arguments. Supply all three
protected manifests recorded in the existing task manifests: frozen paint/PUCPR,
ACPDS and CNR. Their file/image hashes are verified and matching image bytes
excluded. No protected labels enter learning or validation. Unknown source
provenance remains unknown; exact matching is not proof of absence of all
reencoded protected imagery. Mixed-source Parking Spaces is excluded entirely.

```bash
BASE='/home/karoljaron/Dokumenty/ChatGPT/System parkingowy'
PAINT="$BASE/paint-segmentation-training-2026-11-01/frozen-v1/dataset.json"
ACPDS="$BASE/dataset-model-improvement-2026-11-08-v1/final-comparison-v1/frozen/acpds-manifest.json"
CNR="$BASE/dataset-model-improvement-2026-11-08-v1/final-comparison-v1/frozen/cnr-manifest.json"
"$PY" -m research.scene_v3.prepare --audit /tmp/pv3-raw \
  --extended /tmp/pv3-audit --output /tmp/pv3-vehicles \
  --protected "$PAINT" --protected "$ACPDS" --protected "$CNR"
"$PY" -m research.scene_v3.maps prepare --audit /tmp/pv3-raw \
  --output /tmp/pv3-maps \
  --protected "$PAINT" --protected "$ACPDS" --protected "$CNR"
"$PY" -m research.scene_v3.train --manifest "$ART/vehicle-data/manifest.json" \
  --detector '/home/karoljaron/Dokumenty/ChatGPT/System parkingowy/occupancy-pilot/model-assets-v1/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth' \
  --epochs 6 --output /tmp/pv3-vehicle-training
"$PY" -m research.scene_v3.maps train --manifest "$ART/scene-map-data/manifest.json" \
  --pretrained '/home/karoljaron/Dokumenty/ChatGPT/System parkingowy/paint-segmentation-training-2026-11-01/pretrained/resnet50-11ad3fa6.pth' \
  --sha256 11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca \
  --epochs 10 --output /tmp/pv3-map-training
```

Training requires CUDA. Small numerical variation can occur with GPU kernels;
source group membership, initialization, hyperparameters and checkpoint selection
are recorded rather than claiming byte-identical future training checkpoints.

Next task: prepare a bounded, reviewed set of adjacent ground-space polygons
and vehicle-to-space relations, distinguishing perspective overlap from real
two-space blockage. This supplies the missing target for a neighbour-aware
assignment model; existing segmentation alone cannot establish that relation.
