# DAOD-RGB2IR

See [the SAGA + GAN bridge workflow](GAN_BRIDGE_WORKFLOW.md) for four-domain FCOS
training with five stages, per-iteration interleaving and checkpoint/resume usage.

## RandomMono + BTF experiment

Base: `origin/feat/randommono-btf-kaggle` at `feff39f`.
This branch replaces source-image SAGA views with a shared RandomMono + Bilateral
Texture Filtering (BTF) endpoint. Detection/KD losses, teacher EMA, routing,
phase-four RGB supervision, dataset roles and existing evaluation remain intact.

For each source image, draw three positive random weights and normalize their
sum. Use one weight vector across the whole image and reuse the same endpoint
for its teacher and student views:

```text
M = wR * R + wG * G + wB * B
Z = (1 - rho) * M + rho * BTF(M)
view(alpha) = alpha * RGB + (1 - alpha) * repeat_three_channels(Z)
```

Defaults: `rho=0.25`, BTF radius `2`, one BTF iteration; existing curriculum
alpha levels are retained. `alpha=1` bypasses bridge processing and random draws.
Raw IR follows the existing path. This is an appearance bridge: it removes color
and suppresses texture without claiming to synthesize physically correct heat.
The defaults are starting values for experiments, not validated FLIR settings.

### Object color consistency on P3-P5

The student and pure RGB EMA teacher expose FPN levels P3, P4 and P5. For
each level, `DetectionLossCriterion.color_consistency` uses source GT boxes
after the shared flip/resize, ROIAlign (7x7, strides 8/16/32), spatial averaging,
L2 normalization and mean L1. The level losses are averaged, then multiplied by
`color_consistency_weight` (default 0.1). No projection head is added; gradients
flow into the student FPN/backbone, while the teacher reference is detached.

The loss runs after RGB teacher initialization on labeled source views with
student alpha below 1. Pure RGB, RGB baseline, GAN-IR and real IR steps skip it.
The GAN workflow's appearance-bridge domain also supports it, including its final
stage. A matching pure RGB teacher forward is reused when available; otherwise
one extra RGB teacher forward supplies the reference. Per-level `color_loss_p3`,
`color_loss_p4`, `color_loss_p5` and weighted `loss_color_consistency` are logged.

Settings live in `LossConfig` in `configs/distillation.py`. CLI and notebook
settings expose the same options:

```python
color_consistency_weight = 0.1       # Set 0.0 to disable, including extra forward.
color_consistency_levels = ("p3", "p4", "p5")
color_consistency_level_weights = (1.0, 1.0, 1.0)
```

CLI flags: `--color-consistency-weight`, `--color-consistency-levels` and
`--color-consistency-level-weights`. Level weights must match the selected levels;
their sum is normalized by the criterion. The default formula is
`L_total = L_existing + 0.1 * (L_p3 + L_p4 + L_p5) / 3`.

### Kaggle: upload one notebook and Run All

1. Import `notebooks/flir_randommono_btf_10k.ipynb` into Kaggle.
2. Attach the existing dataset and enable **GPU** and **Internet** for uncached
   optional dependencies and pretrained FCOS weights. Start with a fresh kernel.
3. Check the **Experiment settings** cell, then **Run All**.

The defaults are:

```python
data_root = "/kaggle/input/datasets/tungduong123qwe/flir-dataset/align"
output_dir = "/kaggle/working/output_randommono_btf"
total_iters = 10000
batch_size = 4
workers = 0
```

All required project source is included as visible Python module cells. Each
source cell begins with `%%flir_module <module_name>`, followed directly by its
imports, classes and functions. Keep this first line when editing; after a code
change, Run All again so the loader imports the edited modules. The setup cell
registers this Jupyter cell magic automatically. Source code is not wrapped in
quoted strings. No repository clone,
project `.py` files, or shell training command is needed. The notebook calls the
same `example_flir.main(args)` entrypoint as the CLI. It checks the dataset,
CUDA/NMS, real BTF, COCO dependency and pretrained weights before training, and
saves `effective_config.json`, `source_manifest.json` and `bridge_preview.png`.
The setup cell installs missing OpenCV contrib / pycocotools in a private
runtime directory and preserves Kaggle's torch/torchvision stack.
`workers=0` avoids subprocess import problems with the embedded modules.

Training retains existing evaluation and checkpoint behavior, saving
`best_model.pth`, `final_model.pth` and periodic checkpoints under the output
directory. The last executable cell reloads the saved student and runs official
COCO evaluation on IR validation. It selects `best_model.pth`, falling back to
`final_model.pth`, and writes `coco_ir_<checkpoint-stem>.json`. Set
`SETTINGS["eval_checkpoint"]` to evaluate another saved checkpoint. Set
`SETTINGS["resume"]` before the training cell to use the existing resume path;
keep the intended training/augmentation settings consistent with that run.

For ablations, set `texture_strength=0.0` for RandomMono alone, or
`intermediate_domain="saga"` for the previous augmentation. Optional pseudo-label
audits remain available through `eval_pseudo_labels=True`. Reduce batch sizes in
settings if the selected GPU runs out of memory; GPU memory and throughput have
not been measured locally.

### Existing CLI

```bash
python example_flir.py --data_root /path/to/align --output_dir output_randommono_btf --total_iters 10000 --intermediate-domain randommono_btf --texture-strength 0.25 --btf-radius 2 --btf-iterations 1
```

The new BTF path requires an OpenCV contrib build exposing
`cv2.ximgproc.bilateralTextureFilter` (not regular OpenCV). The notebook installs
`opencv-contrib-python-headless==4.11.0.86` and `pycocotools==2.0.11` if needed.
Use a single OpenCV distribution in a CLI environment. No fallback blur or
random-weight initialization is substituted when a requirement is missing.

After editing canonical Python modules, regenerate the notebook:

```bash
python tools/build_kaggle_notebook.py
python -m pytest tests/test_kaggle_notebook.py -q
```

The tests check source hashes, visible class/function definitions, deterministic
generation, and real IPython execution/editing without a project checkout. Local validation uses CPU and synthetic VOC data;
the actual 10k Kaggle run and FLIR AP must be measured on the attached dataset.
