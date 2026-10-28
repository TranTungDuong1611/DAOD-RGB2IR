# IR pseudo-label diagnostics

This opt-in observer compares the **dense rows actually selected by KD** with
IR ground truth from the current training batch. It evaluates the teacher
outputs already produced for distillation, before the optimizer/EMA update.
It does not run another forward or compute teacher detection AP.

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

Annotations always come from `data_root/Annotations`, sharing the dataset root.
For each IR sample ID, the reader looks up `<sample_id>.xml`. The class mapping
is `person=0`, `car=1`, `bicycle=2`, matching the existing FLIR loaders. Small
boxes are retained, as in IR validation. Optional XML image dimensions must
match the original IR image; malformed known-class boxes invalidate the file.
Missing/invalid files are reported as unavailable, not as background images.
A valid XML file with no supported objects is an annotated empty image.

## Cadence and output

An audit is due at completed iterations 500, 1000, 1500, etc. If that iteration
uses RGB, or no initialized/enabled teacher performs KD, evaluation waits for
the next eligible real IR KD batch. RGB-backed phase-2 routes are excluded,
including routes whose names contain `ir`. If several ticks pass before IR
becomes eligible, one audit samples the current batch; it does not replay old
ticks. A deferred audit does not shift subsequent global boundaries.

For example, RGB at step 500 followed by IR KD at step 501 produces an audit at
501. The next boundary remains 1000. Each teacher actually used for KD on that
batch gets a separate record; teachers outside the selected route are not run.

Records are appended to `output_dir/logs/pseudo_label_eval.jsonl`. A
`[pseudo-label-eval]` summary also appears in the existing console/train log,
independently of `--log_interval`. JSONL records include step, phase, route,
teacher, sample IDs, annotation availability, flip, KD selection settings,
IoU threshold, and metrics. Reusing an output directory appends to this file;
use a different output directory for an independent run.

Observer progress is saved in training checkpoints to avoid duplicate audits
in an already evaluated interval after resume. Older baseline checkpoints
remain loadable; their first eligible IR batch can trigger a pending audit.

## Metric definitions

GT follows the batch's already sampled horizontal flip and the actual resize
recorded by the teacher forward. The existing FCOS location, pyramid-scale,
and center-sampling assignment determines each row's corresponding GT object.
The criterion's batch-global top-k/HM mask is reused directly, before NMS.

| Metric | Meaning |
| --- | --- |
| `available_images`, `missing_images`, `invalid_images` | Label availability in the observed batch |
| `selected_rows_all` | Selected KD rows across all images, including unavailable labels |
| `selected_rows` | Selected KD rows in images with available labels; denominator for validity |
| `selected_foreground_rows` | Selected rows assigned to a GT object by FCOS |
| `valid_rows`, `valid_fraction` | Selected rows with foreground assignment, correct class, and sufficient box IoU; fraction over `selected_rows` |
| `foreground_class_accuracy` | Correct-class fraction among finite selected foreground rows |
| `foreground_box_iou_mean` | Mean IoU with assigned GT among finite selected foreground rows |
| `quality_mae_foreground` | Mean absolute error between sigmoid quality and assigned-GT box IoU, on finite selected foreground rows |
| `gt_count`, `covered_gt`, `gt_coverage` | GT objects covered by at least one valid selected row, counted once per image/object |
| `nonfinite_selected_rows`, `finite_foreground_rows` | Invalid prediction count and foreground denominator for class/IoU/quality metrics |

Undefined ratios are `null`, not zero. For example, a batch without available
labels has no validity score; an annotated empty image with selected rows has
validity zero. An image with GT but no selected rows has coverage zero.
Multiple dense rows can legitimately cover one object. These values are not
post-NMS precision/recall or mAP. Individual batches vary; compare trends with
their sample IDs and denominators.

## Training isolation

Audit targets are read through a separate side channel. IR training targets
remain `None`. Audit labels/metrics never control KD masks, loss weights,
optimizer/LR steps, EMA, student validation, or best-checkpoint selection.
Numeric policy lives in `PseudoLabelEvalConfig`, cadence in the curriculum
scheduler, execution in the reader/adapter/evaluator, and output in
`training/observability.py`.

Tests exercise real FCOS with a small backbone, checking geometry, metrics,
availability, scheduling, resume, forward counts, and identical losses,
gradients, weights, optimizer/LR state, and RNG with diagnostics on/off and
with changed/missing audit GT. Full FLIR/GPU training is a separate check.
