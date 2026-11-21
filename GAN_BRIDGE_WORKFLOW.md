# FCOS: RGB, SAGA, GAN-IR and real IR

Use `--workflow gan_bridge` to train a detector with two intermediate domains.
Each iteration uses one domain batch. The scheduler spreads domains according to
the ratios below, rather than training on long contiguous domain blocks.

| Stage | End of run | RGB | SAGA | GAN-IR | Real IR |
| --- | --- | --- | --- | --- | --- |
| 1: RGB warmup | 20% | 100% | 0% | 0% | 0% |
| 2: introduce bridges | 40% | 50% | 30% | 20% | 0% |
| 3: gradual transition | 60% | 30% | 30% | 30% | 10% |
| 4: GAN-heavy adaptation | 80% | 20% | 20% | 40% | 20% |
| 5: target focus | 100% | 10% | 10% | 20% | 60% |

Counts approach the requested ratios over time; short stages may not contain a
complete ratio period. Stage boundaries round upward to whole iterations. For
fewer than five iterations, some later stages can be empty.

## Run

The existing project environment must provide PyTorch, Torchvision and the
evaluation dependencies. No additional production packages are needed.

```bash
python example_flir.py \
  --data_root /path/to/flir/align \
  --workflow gan_bridge \
  --gan-weights /path/to/latest_net_G_A.pth \
  --output_dir output_flir/gan_bridge \
  --total_iters 10000 --batch_size 4 --workers 4 --device cuda \
  --bridge-ratios '10:0:0:0,5:3:2:0,3:3:3:1,2:2:4:2,1:1:2:6' \
  --bridge-phase-fractions 0.2 0.4 0.6 0.8 1.0
```

In PowerShell, put the command on one line or replace the shell continuation
backslashes with PowerShell backticks. Paths above are placeholders.

`--bridge-phase-fractions` gives five cumulative stage end fractions. For example,
`0.15 0.35 0.60 0.80 1.0` allocates more time to the middle stage. The legacy
`--phase1-end`, `--phase2-end` and `--phase3-end` flags are rejected for this workflow
because they cannot describe its five stages. Teacher initialization defaults to
the bridge warmup end; an explicit `--ema-start` must equal that boundary.

`--domain-sampling` controls the original curriculum workflow. The GAN bridge
always uses per-iteration interleaving. `--workflow curriculum` and
`--workflow rgb_baseline` still work without GAN weights.

The original curriculum's `--phase4-rgb-ratio` controls its final-stage RGB
supervision. GAN bridge uses its own four-domain ratios instead, including in
the final stage. Its source GT weight remains `--bridge-source-sup-weight`.

Optional fixed-subset and final full IR teacher audits are available with
`--eval-pseudo-labels`; see [pseudo-label evaluation](PSEUDO_LABEL_EVALUATION.md).
Their cadence is independent of the domain chosen for a training iteration.

## Required GAN artifact

Supply a pretrained **RGB -> IR** generator with 3 input channels, 1 output
channel, `ngf=64`, 9 residual blocks, reflection padding and instance normalization
without affine parameters, running statistics or dropout. Its tensor names match
the [`ResnetGenerator` in the upstream CycleGAN project](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py).
The local implementation is in `models/cyclegan_generator.py`.

The loader accepts a generator state dict, a `state_dict` wrapper, or a state dict
with a uniform `module.` prefix. Loading is strict; a different architecture or
the wrong translation direction must be corrected at the artifact/configuration
level. No repository is cloned and no pretrained teacher detector checkpoint is
required.

The GAN stays frozen, in evaluation mode and outside the detector optimizer.
Only GAN-IR iterations call it. Images enter as RGB floats in `[0,1]`, are mapped
to `[-1,1]` for the generator and back to `[0,1]` afterward. The single output
channel is repeated three times for FCOS. Translation is done one image at a time
to limit generator activation memory. Odd spatial sizes are padded on the bottom
and right, then cropped back to their original size.

## Views, GT and teacher updates

RGB, SAGA and GAN-IR each consume a batch from the same labeled RGB loader. SAGA
and GAN-IR are generated independently from RGB. Real IR uses the unlabeled IR
loader. Shared horizontal flipping happens before appearance conversion, and the
teacher/student detector forwards reuse the student's selected resize.

| Appearance | GT | Teacher / EMA after warmup |
| --- | --- | --- |
| RGB | Yes | RGB |
| SAGA in stage 2 | Yes | RGB |
| SAGA in stages 3–5 | Yes | IR |
| GAN-IR | Yes, inherited from RGB | IR |
| Real IR | No | IR |

SAGA blends object regions toward grayscale using source GT boxes; its background
stays RGB before student photometric augmentation. The student and teacher alpha
defaults, per stage, are:

```text
student: 1.0 0.75 0.50 0.25 0.25
teacher: 1.0 0.75 0.50 0.50 0.25
```

They are configurable with `--bridge-saga-student-alphas` and
`--bridge-saga-teacher-alphas`, each taking five values in `[0,1]`.
`--bridge-saga-teachers` defaults to `none rgb ir ir ir`.

The student starts with the existing FCOS initialization options. Stage 1 trains
the student with RGB GT, then copies its warmed weights into both teachers at the
first post-warmup iteration. Selected teacher parameters/buffers subsequently
follow the existing EMA implementation after a successful student update. Both
teachers are required for `gan_bridge`.

The five domain stages map to the existing four loss/HM policies as
`phase1, phase2, phase3, phase3, phase4`. Dense FCOS distillation, HM filtering and
component loss weights are retained. In the final stage, all source-derived
batches keep a positive GT weight, controlled by `--bridge-source-sup-weight`
(default `1.0`). Real IR never receives GT. RTC and Direct Gate are not part of
this workflow.

## Resume and logs

Repeat the same training command and add:

```bash
--resume output_flir/gan_bridge/final_model.pth
```

Checkpoints contain student, both teachers, optimizer, LR scheduler, domain
allocation progress, global step, teacher initialization state and evaluation
history. They record the GAN file SHA-256 and architecture, all five ratios and
durations, SAGA policies, source GT weight and training settings. Incompatible
bridge settings or generator bytes are rejected before model restoration.
Identical GAN bytes at a different path can resume. Changing `total_iters` changes
the stage boundaries and requires a new run rather than resume.

Domain sequence continuation is deterministic. Complete numerical reproduction is
not promised: existing data-loader cursor/RNG state is not checkpointed. If a
forward throws or training is interrupted before an iteration completes, its
domain allocation is rolled back so the final checkpoint still matches the last
completed global step; consumed loader samples are not rolled back.

Iteration logs show the appearance domain, source loader, stage `1/5`–`5/5`, ratio,
teacher names, EMA target, loss components and SAGA alpha when applicable. Evaluation
continues using the project's existing AP50 and loss/HM phase labels; stages 3/4
share the adaptation policy. A real GAN checkpoint and FLIR data are needed to
measure accuracy; automated smoke tests do not establish an AP gain.

Dataset splits retain the source branch's behavior. This change does not introduce
a disjoint RGB/IR split. Inspect translated real samples before a long experiment:
reusing RGB boxes assumes the GAN preserves object geometry, which a tensor shape
check alone cannot establish.

## Verification

```bash
python -m unittest discover -s tests -v
```

The bridge tests include range/shape contracts, strict weight loading, SAGA masks,
all five domain allocations, full short training with small model doubles, dense
distillation with a real Torchvision FCOS head and lightweight backbone, freezing,
GT on source routes, teacher/EMA routing, saved progress and incompatible resume.
