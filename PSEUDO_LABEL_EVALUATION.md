# IR pseudo-label diagnostics

This opt-in observer measures **dense teacher rows selected by the same rule
as KD** on a fixed labeled IR validation subset. Each pass uses the current
teacher weights after the training/EMA update, without random augmentation.
It runs additional teacher-only forwards and does not compute detection AP.

## Run

```bash
python -B example_flir.py --data_root /path/to/flir/align --output_dir outputs/pseudo_audit --eval-pseudo-labels
```

Optional arguments:

| Argument | Default | Purpose |
| --- | --- | --- |
| `--eval-pseudo-labels` | Off | Enable observation without changing supervision |
| `--pseudo-label-eval-interval` | `500` | Global training iteration interval |
| `--pseudo-label-iou-threshold` | `0.5` | Minimum IoU for a valid selected row |
| `--pseudo-label-eval-samples` | `256` | Maximum labeled validation images in each periodic audit |
| `--pseudo-label-eval-seed` | `0` | Local seed for repeatable subset selection |
| `--eval_batch_size` | `4` | Fixed audit and student validation batch size |

Annotations always come from `data_root/Annotations`, sharing the dataset root.
For each IR sample ID, the reader looks up `<sample_id>.xml`. The class mapping
is `person=0`, `car=1`, `bicycle=2`, matching the existing FLIR loaders. Small
boxes are retained, as in IR validation. Optional XML image dimensions must
match the original IR image; malformed known-class boxes invalidate the file.
Missing/invalid files are reported as unavailable, not as background images.
A valid XML file with no supported objects is an annotated empty image.
Both `<annotation>` and the FLIR-exported `<Annotation>` roots are accepted.

## Cadence and output

An audit is due at completed iterations 500, 1000, 1500, etc., once EMA teachers
are initialized. It is independent of the current RGB/IR training route because
it uses separate IR validation loaders. With the default EMA start at 2000,
the first eligible audit is normally at completed step 2001, followed by 2500,
3000, etc. Missed warmup ticks are not replayed. Each enabled teacher gets its
own pass (`rgb`, `ir`, or both according to `--teacher-mode`).

Periodic audits use the same subset of up to 256 validly annotated IR images
from the validation split. Selection uses a local seeded shuffle and retains
at least one image containing each supported class when the sample budget
allows, then fills remaining slots from shuffled candidates. It does not
guarantee exact class proportions. If fewer valid images exist, all are used.
Assembly fails clearly if no valid validation annotations are available.

Image IDs, order, and batch size stay fixed across passes. Selection reads
image headers/XML at startup; images and predictions are not cached. After
normal training completion, each enabled teacher also gets one full pass over
all existing IR validation images, retaining the final incomplete batch.
Missing/invalid labels in that full pass are reported explicitly.

Records are appended to `output_dir/logs/pseudo_label_eval.jsonl`. A
`[pseudo-label-eval]` summary also appears in the existing console/train log,
independently of `--log_interval`. Records include step, phase, teacher, scope
(`fixed_ir_validation` or `full_ir_validation`), image/batch counts, sample IDs,
KD settings, resize, IoU threshold, and aggregate metrics. Console summaries
include scope and image count. Reusing an output directory appends to this file;
use a different output directory for an independent run.

Observer progress is saved in training checkpoints to avoid duplicate audits
in an already evaluated interval after resume. The fixed image manifest and
batch size are stored too; resume rejects a changed subset or batch grouping.
Older baseline checkpoints remain loadable without observer state. A completed
final checkpoint skips an already recorded full pass on resume. An older
completed checkpoint without that flag receives the full pass without any
further optimizer steps.

## Metric definitions

Default selection schedules keep the top-k budgets and use stricter HM floors:

| Phase | RGB top ratio / min HM | IR top ratio / min HM |
| --- | --- | --- |
| 1 | 1% / 0.55 | 0.2% / 0.70 |
| 2 | 2% / 0.45 | 0.5% / 0.60 |
| 3 | 4% / 0.40 | 1.5% / 0.50 |
| 4 | 6% / 0.40 | 3% / 0.50 |

Phase 1 does not run KD; phase 4 does not use the RGB teacher for KD. Raising
the HM floor only removes rows whose scores fall below it; a top-k group that
already exceeds the new floor remains unchanged. These are training-selection
thresholds, separate from the GT IoU threshold used only for diagnostics.

Evaluation uses the final configured short-edge size and no random flip. GT
follows the actual teacher resize. Each selected decoded box is compared with
all same-class GT boxes in its image and matched to the highest-IoU one. A row
is valid when that IoU meets the configured threshold (default 0.5). FCOS
location, pyramid-scale, and center-sampling rules are not used by diagnostics;
supervised training keeps its existing assignment unchanged.

Training and evaluation share the batch-global top-k/HM selection helper,
before NMS. GT never affects selection. Keep audit batch size fixed when
comparing runs because batch-global selection depends on batch grouping.

| Metric | Meaning |
| --- | --- |
| `available_images`, `missing_images`, `invalid_images` | Label availability in the observed batch |
| `selected_rows_all` | Selected KD rows across all images, including unavailable labels |
| `selected_rows` | Selected KD rows in images with available labels; denominator for validity |
| `selected_foreground_rows`, `finite_foreground_rows` | Finite selected rows with IoU at or above the threshold to any GT, ignoring class |
| `valid_rows`, `valid_fraction` | Rows matching any same-class GT at the IoU threshold; fraction over all `selected_rows` |
| `foreground_class_accuracy` | Fraction of geometrically matched rows also matching a same-class GT at the threshold |
| `foreground_box_iou_mean` | Mean best IoU to any GT among geometrically matched rows, including wrong-class predictions |
| `quality_mae_foreground` | Mean absolute error between sigmoid quality and best geometric GT IoU, on geometrically matched rows |
| `gt_count`, `covered_gt`, `gt_coverage` | GT objects covered by at least one valid selected row, counted once per image/object |
| `nonfinite_selected_rows` | Selected predictions with non-finite box, class, or quality values; they remain in the validity denominator |

Undefined ratios are `null`, not zero. For example, a batch without available
labels has no validity score; an annotated empty image with selected rows has
validity zero. An image with GT but no selected rows has coverage zero.
Multiple dense rows can legitimately cover one object. These values are not
post-NMS precision/recall or mAP. Coverage counts each row's best same-class GT
once per image/object. Foreground class accuracy is not tautologically 100%:
its denominator includes high-IoU wrong-class predictions. Ratios use pooled row/object counts; IoU and
quality means are weighted by finite selected foreground counts. Small/empty
batches therefore do not receive the same weight as large batches.

Records identify this definition with `matching="same_class_box_iou"`. Do not
compare these validity/foreground values directly with older FCOS-assignment
audits, even if some metric field names are unchanged.

## Training isolation

Audit targets are read through a separate validation path. IR training targets
remain `None`. Audit labels/metrics never control KD masks, loss weights,
optimizer/LR steps, EMA, student validation, or best-checkpoint selection.
Numeric policy lives in `PseudoLabelEvalConfig`, cadence in the curriculum
scheduler, execution in the reader/adapter/evaluator, and output in
`training/observability.py`.

Evaluation restores Python, NumPy, CPU/CUDA Torch RNG and every model module's
previous train/eval mode in a `finally` block. Dedicated loaders own seeded
generators whose states are restored too. Training loaders/cursors are never
consumed. No student forward, backward, optimizer/LR step, or EMA update occurs
inside an audit. The baseline's 20% supervised RGB phase-4 routing is retained.

Tests exercise real FCOS with a small backbone, checking geometry, metrics,
availability, scheduling, resume, forward counts, and identical losses,
gradients, weights, optimizer/LR state, and RNG with diagnostics on/off and
with changed/missing audit GT. Full FLIR/GPU training is a separate check.
