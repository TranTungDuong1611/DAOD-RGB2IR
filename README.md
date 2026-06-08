# Curriculum Domain Adaptation RGB → IR (FLIR-aligned)

Huấn luyện object detector thích nghi miền **RGB → IR (thermal)** trên bộ **FLIR ADAS Aligned**,
theo kiểu curriculum nhiều phase với **dual-teacher EMA** + **pseudo-label**.

Có **2 entry-point** dùng chung framework, chỉ khác **ảnh trung gian (bridge)**:

| Script | Bridge (ảnh trung gian) | Cần GAN? |
|--------|-------------------------|----------|
| `train_kaggle.py` | **SoftSAGA** — gray-hoá vùng object, nền giữ RGB | ❌ Không |
| `training_full_v4.py` | **GAN-IR blend** — `α·RGB + (1−α)·CycleGAN(RGB)` | ✅ Có |

Hai bản **giống hệt** về model / scheduler / loss / EMA / eval — chỉ khác cách tạo 3 domain "mid".

---

## 1. Yêu cầu

- Python 3.8+, `torch` + `torchvision` (≥ 0.13). Kaggle có sẵn.
- Đánh giá mAP **tự viết** (không cần `pycocotools` / `torchmetrics`).
- `training_full_v4.py` cần thêm repo **pytorch-CycleGAN-and-pix2pix** (tự `git clone` nếu thiếu) và file weight CycleGAN `G_A`.

---

## 2. Dữ liệu

Trỏ `--data-root` tới thư mục `align/` có cấu trúc:

```
align/
├── JPEGImages/
│   ├── FLIR_XXXXX_PreviewData.jpeg   ← ảnh IR (thermal)
│   ├── FLIR_XXXXX_RGB.jpg            ← ảnh RGB (đã căn chỉnh với IR)
│   └── ...
├── Annotations/
│   └── FLIR_XXXXX_PreviewData.xml    ← nhãn VOC (dùng cho cả RGB & IR)
├── align_train.txt                   ← danh sách stem "FLIR_XXXXX_PreviewData"
└── align_validation.txt
```

- **Classes** (0-indexed): `person=0, car=1, bicycle=2`. Nhãn `FLIR`, `dog` bị bỏ qua.
- Train RGB lấy nhãn từ XML căn chỉnh; val/eval chạy trên **IR thật**.

---

## 3. Setup trên Kaggle

```bash
# Nếu code nằm trong /kaggle/input (read-only) → copy ra working để ghi được
cp -r /kaggle/input/<dataset-code>/codeRGB_IRDAOD-RGB2IR /kaggle/working/daod
cd /kaggle/working/daod
```

> Có thể chạy thẳng từ `/kaggle/input` (script tự thêm thư mục của nó vào `sys.path`),
> chỉ cần `--output-dir` trỏ vào `/kaggle/working/...`.

---

## 4. Chạy train

### 4a. Bản SAGA bridge — `train_kaggle.py`

```bash
python train_kaggle.py \
  --data-root  /kaggle/input/aligned-flir/aligned_flir/align \
  --output-dir /kaggle/working/out_daod \
  --model      faster_rcnn \
  --total-iters 7500 \
  --eval-period 500 \
  --batch-size 2
```

### 4b. Bản GAN-IR bridge — `training_full_v4.py`

```bash
python training_full_v4.py \
  --data-root   /kaggle/input/aligned-flir/aligned_flir/align \
  --gan-weights /kaggle/input/gan-weights/latest_net_G_A.pth \
  --output-dir  /kaggle/working/out_v4 \
  --model       faster_rcnn \
  --total-iters 7500 \
  --eval-period 500 \
  --batch-size  2
```

> **Đổi detector:** thêm `--model fcos` để dùng FCOS thay vì Faster R-CNN.

---

## 5. Tham số chính

| Cờ | Mặc định | Ý nghĩa |
|----|----------|---------|
| `--data-root` | (bắt buộc) | Thư mục `align/` |
| `--gan-weights` | (bắt buộc, chỉ v4) | CycleGAN `G_A` `.pth` (RGB→IR) |
| `--cyclegan-repo` | `/kaggle/working/pytorch-CycleGAN-and-pix2pix` | Repo CycleGAN (tự clone nếu thiếu) |
| `--output-dir` | `./out_*` | Nơi lưu checkpoint + log |
| `--model` | `faster_rcnn` | `fcos` \| `faster_rcnn` |
| `--total-iters` | `7500` | Tổng số iteration (step) |
| `--phase-ends` | auto `0.2/0.4/0.65/0.85` | Override ranh giới 5 phase: `"p1,p2,p3,p4"` |
| `--eval-period` | `500` | Eval IR + lưu best mỗi N step |
| `--batch-size` | `4` | **Faster R-CNN nên để 2** (xem Lưu ý) |
| `--img-height` / `--img-width` | `512` / `640` | Kích thước resize (giữ object nhỏ) |
| `--lr` | `5e-5` | Adam |
| `--ema-alpha` | `0.999` | Momentum EMA teacher |
| `--pseudo-conf` | `0.7` | Ngưỡng pseudo-label (khi tắt adaptive-thresh) |
| `--no-adaptive-thresh` | off | Dùng ngưỡng cố định thay vì per-phase |
| `--coco-map` | off | Tính mAP@0.5:0.95 (chậm) thay vì chỉ mAP@0.5 |
| `--init-weights` | none | Init student+2 teacher từ checkpoint (xem Lưu ý) |

---

## 6. Curriculum 5 phase (phân bố domain)

Cả 2 bản dùng chung scheduler. Với `--total-iters 7500`, phase-ends mặc định `(1500,3000,4875,6375)`:

| Phase | % tổng | Domain (trong phase) |
|-------|--------|----------------------|
| 1 RGB warmup | 20% | `rgb` 100% |
| 2 RGB+near_rgb | 20% | `rgb` 67% / `mid_near_rgb` 33% |
| 3 Intermediate | 25% | `mid_intermediate` 100% |
| 4 near_ir+IR | 20% | `mid_near_ir` 67% / `ir` 33% |
| 5 IR focus | 15% | `ir` 100% |

`mid_*` = ảnh bridge (SAGA hoặc GAN-IR). Alpha 3 mức: `near_rgb=0.70`, `intermediate=0.50`, `near_ir=0.25`
(tỉ lệ RGB trong blend; càng về sau càng giống IR).

---

## 7. Đầu ra

Trong `--output-dir`:

- `best_student.pth` — checkpoint **mAP@0.5 cao nhất** (lưu tự động).
- `last_student.pth` — checkpoint cuối.
- Log: mAP IR + per-class (`person`/`car`/`bicycle`) mỗi `eval_period` step, kèm bảng history cuối.

Checkpoint là dict `{"student": state_dict, "global_step", "mAP@0.5"}`.

---

## 8. Lưu ý quan trọng

1. **Faster R-CNN → `--batch-size 2`.** Nặng hơn FCOS nhiều ở 512×640; batch 4 dễ **OOM** trên GPU Kaggle.

2. **`--init-weights` phải khớp detector.** Checkpoint FCOS **không** nạp được vào Faster R-CNN (head khác).
   Khi chạy `--model faster_rcnn`, **bỏ `--init-weights`** (init từ COCO head — đủ tốt), hoặc chỉ truyền checkpoint Faster R-CNN.

3. **Theo dõi cú sụp ở phase IR (step ~5000+).** Cả 2 bản dùng chung `losses.py`: khi pseudo-label IR rỗng,
   detector có thể học "không phát hiện gì" → mAP tụt mạnh. `best_student.pth` vẫn giữ checkpoint tốt nhất nên
   không mất kết quả. (FCOS nhạy với lỗi này hơn Faster R-CNN.)

4. **Resolution quan trọng.** Giữ `512×640`; hạ xuống quá thấp (vd 224) làm object nhỏ (person/bicycle) gần như
   biến mất → mAP tụt mạnh.

---

## 9. Cấu trúc code

| File | Vai trò |
|------|---------|
| `train_kaggle.py` | Entry-point — bridge **SAGA** |
| `training_full_v4.py` | Entry-point — bridge **GAN-IR** (subclass, chỉ override `_next_mid`) |
| `trainer.py` | Vòng train curriculum, dual-teacher EMA, dispatch domain step |
| `scheduler.py` | `CurriculumScheduler` — chọn domain mỗi iteration |
| `config.py` | Dataclass cấu hình (phase, loss, EMA, alpha SAGA…) |
| `fcos_wrapper.py` / `faster_rcnn_wrapper.py` | Bọc detector + factory `build_*_trio` |
| `saga.py` | SoftSAGA (bridge của bản SAGA) |
| `losses.py` | Loss RGB / MID / IR + lọc pseudo-label |
| `ema.py` | Cập nhật EMA teacher |
| `adaptive_threshold.py` | Ngưỡng pseudo-label theo phase/teacher |
| `evaluator.py` | mAP (VOC/AUC) + `PhaseEvaluator` (auto eval + best ckpt) |
| `datasets/flir.py` | Dataset FLIR RGB / IR / IR-val + collate |
