# DAOD-RGB2IR

## RandomMono + BTF experiment

Base: freshly fetched `origin/feat/fcos-distillation` at `1cb91c5`.
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

All required project source is embedded as module cells. No repository clone,
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

The tests check embedded source hashes, deterministic generation and imports
without a project checkout. Local validation uses CPU and synthetic VOC data;
the actual 10k Kaggle run and FLIR AP must be measured on the attached dataset.