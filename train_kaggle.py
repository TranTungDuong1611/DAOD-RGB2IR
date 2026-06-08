"""
train_kaggle.py — Entry-point train framework Curriculum-DAOD (RGB→MID(SAGA)→IR)
trên bộ FLIR-aligned của Kaggle (cùng dataset dùng cho training_full_v3).

Khác training_full_v3 (D3T + CycleGAN bridge):
  - Bridge ở đây là SAGA (Semantic-Aware Grayscale) — KHÔNG cần GAN weights.
  - Dùng nguyên bộ máy có sẵn: CurriculumScheduler (5 phase), AdaptiveThresholdScheduler,
    dual-teacher EMA, PhaseEvaluator (mAP + auto eval + best checkpoint).
  - 2 teacher khởi tạo GIỐNG NHAU từ student (build_fcos_trio) → đúng tinh thần D3T;
    teacher IR chỉ chuyên biệt hóa dần qua EMA.

Chạy (Kaggle):
  cd /kaggle/working/codeRGB_IRDAOD-RGB2IR   # thư mục chứa trainer.py, config.py, ...
  python train_kaggle.py \
    --data-root /kaggle/input/aligned-flir/aligned_flir/align \
    --output-dir /kaggle/working/out_daod \
    --total-iters 7500 --batch-size 4 --eval-period 500

  # (tùy chọn) khởi tạo student từ checkpoint RGB đã train sẵn — đúng "burn-in" D3T:
    --init-weights /kaggle/input/teacher-weights/final_model_epoch_25_rgb.pth
"""

import os
import sys
import argparse
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

# Cho phép chạy từ thư mục khác — thêm thư mục script vào sys.path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from config import (
    TrainingConfig, EMAConfig, CurriculumConfig, LossConfig,
)
from trainer import CurriculumDomainAdaptationTrainer
from fcos_wrapper import build_fcos_trio
from faster_rcnn_wrapper import build_faster_rcnn_trio
from ema import copy_student_to_teacher
from adaptive_threshold import AdaptiveThresholdScheduler
from evaluator import DetectionEvaluator, PhaseEvaluator
from datasets.flir import (
    FLIRRGBDataset, FLIRIRDataset, FLIRIRValDataset,
    rgb_collate, ir_collate, ir_val_collate,
    FLIR_CLASSES, NUM_CLASSES, FLIR_TO_COCO_IDX,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_kaggle")


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper: resize ảnh về kích thước cố định VÀ scale box theo (đảm bảo batching)
# ─────────────────────────────────────────────────────────────────────────────

class ResizeWithBoxes(Dataset):
    """
    Bọc dataset FLIR (ảnh đã là tensor [C,H,W], box ở toạ độ pixel gốc).

    rgb_collate / ir_val_collate dùng torch.stack → mọi ảnh phải CÙNG kích thước.
    FLIR-aligned thường đồng kích thước, nhưng để chắc chắn (và bound memory) ta
    resize về (H,W) cố định và scale box tương ứng.

    Hỗ trợ cả item dạng (img, target) lẫn (img,) (IR train không nhãn).
    """

    def __init__(self, base: Dataset, size_hw: Tuple[int, int]) -> None:
        self.base = base
        self.H, self.W = size_hw

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        item = self.base[idx]
        img = item[0]
        _, h0, w0 = img.shape
        img_r = TF.resize(img, [self.H, self.W], antialias=True)

        if len(item) == 1:                       # IR train: (img,)
            return (img_r,)

        target = dict(item[1])                   # (img, target)
        boxes = target.get("boxes")
        if boxes is not None and boxes.numel() > 0:
            sx, sy = self.W / w0, self.H / h0
            scaled = boxes.clone().float()
            scaled[:, [0, 2]] *= sx
            scaled[:, [1, 3]] *= sy
            target["boxes"] = scaled
        return img_r, target


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Curriculum-DAOD (RGB→SAGA→IR) trên FLIR-aligned (Kaggle)"
    )
    # Paths
    p.add_argument("--data-root", type=str, required=True,
                   help="Thư mục 'align' (JPEGImages/, Annotations/, align_train.txt, align_validation.txt)")
    p.add_argument("--output-dir", type=str, default="./out_daod")
    p.add_argument("--init-weights", type=str, default=None,
                   help="(Tùy chọn) state_dict fcos_resnet50_fpn(num_classes=3) để init student "
                        "(và 2 teacher) — đóng vai burn-in nguồn. Nếu bỏ → init từ COCO-pretrained head.")

    # Training length / scheduler
    p.add_argument("--total-iters", type=int, default=7500, help="Tổng số iteration (default: 7500)")
    p.add_argument("--phase-ends", type=str, default=None,
                   help="Override ranh giới phase (iter), 4 số 'p1,p2,p3,p4'. "
                        "Bỏ → tự tính theo tỉ lệ 0.2/0.4/0.65/0.85 của total-iters.")
    p.add_argument("--eval-period", type=int, default=500, help="Eval + save checkpoint mỗi N iter (default: 500)")
    p.add_argument("--log-interval", type=int, default=50)

    # Model
    p.add_argument("--model", type=str, default="faster_rcnn", choices=["fcos", "faster_rcnn"],
                   help="Detector backbone (default: faster_rcnn)")
    p.add_argument("--trainable-backbone-layers", type=int, default=3, help="Số tầng backbone unfreeze (0-5)")
    p.add_argument("--img-height", type=int, default=512)
    p.add_argument("--img-width",  type=int, default=640)

    # Optim / EMA / pseudo
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--ema-alpha", type=float, default=0.999, help="EMA momentum teacher (default: 0.999)")
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--pseudo-conf", type=float, default=0.7,
                   help="Ngưỡng pseudo-label dùng KHI tắt adaptive-thresh (default: 0.7)")
    p.add_argument("--no-adaptive-thresh", action="store_true",
                   help="Tắt AdaptiveThresholdScheduler, dùng ngưỡng cố định --pseudo-conf.")

    # Eval
    p.add_argument("--coco-map", action="store_true",
                   help="Tính mAP@0.5:0.95 (10 ngưỡng IoU) thay vì chỉ mAP@0.5 (chậm hơn).")
    return p.parse_args()


def _compute_phase_ends(total: int, override: Optional[str]) -> Tuple[int, int, int, int]:
    if override:
        vals = [int(x) for x in override.split(",")]
        assert len(vals) == 4, "--phase-ends cần đúng 4 số 'p1,p2,p3,p4'"
        return tuple(vals)  # type: ignore
    return (
        int(0.20 * total),   # phase1 end — RGB warmup
        int(0.40 * total),   # phase2 end — RGB + mid_near_rgb
        int(0.65 * total),   # phase3 end — mid_intermediate
        int(0.85 * total),   # phase4 end — mid_near_ir + IR  (phase5 = IR focus tới hết)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if not os.path.exists(args.data_root):
        raise FileNotFoundError(f"--data-root: '{args.data_root}' không tồn tại")
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size_hw = (args.img_height, args.img_width)
    p1, p2, p3, p4 = _compute_phase_ends(args.total_iters, args.phase_ends)

    logger.info(f"device={device}  data_root={args.data_root}")
    logger.info(f"total_iters={args.total_iters}  phase_ends=({p1},{p2},{p3},{p4})  "
                f"img={size_hw}  batch={args.batch_size}")

    # ── Datasets & loaders ───────────────────────────────────────────────────
    rgb_ds = ResizeWithBoxes(FLIRRGBDataset(args.data_root, split="train"), size_hw)
    ir_ds  = ResizeWithBoxes(FLIRIRDataset(args.data_root,  split="train"), size_hw)
    val_ds = ResizeWithBoxes(FLIRIRValDataset(args.data_root, split="validation"), size_hw)
    logger.info(f"RGB train={len(rgb_ds)}  IR train={len(ir_ds)}  IR val={len(val_ds)}")
    if min(len(rgb_ds), len(ir_ds), len(val_ds)) == 0:
        raise RuntimeError("Một trong các split rỗng — kiểm tra --data-root / file align_*.txt")

    rgb_loader = DataLoader(rgb_ds, batch_size=args.batch_size, shuffle=True,
                            collate_fn=rgb_collate, num_workers=args.num_workers,
                            pin_memory=True, drop_last=True)
    ir_loader  = DataLoader(ir_ds, batch_size=args.batch_size, shuffle=True,
                            collate_fn=ir_collate, num_workers=args.num_workers,
                            pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=ir_val_collate, num_workers=args.num_workers,
                            pin_memory=True)

    # ── Model trio (student + 2 teacher giống hệt) ───────────────────────────
    from_coco = args.init_weights is None
    build_trio = build_faster_rcnn_trio if args.model == "faster_rcnn" else build_fcos_trio
    logger.info(f"Detector: {args.model}")
    student, rgb_teacher, ir_teacher = build_trio(
        num_classes=NUM_CLASSES,
        pretrained_backbone=True,
        trainable_backbone_layers=args.trainable_backbone_layers,
        min_size=min(size_hw),
        max_size=max(size_hw),
        ir_to_rgb=True,
        from_coco=from_coco,
        coco_src_indices=FLIR_TO_COCO_IDX if from_coco else None,
    )

    if args.init_weights:
        if not os.path.exists(args.init_weights):
            raise FileNotFoundError(f"--init-weights: '{args.init_weights}' không tồn tại")
        state = torch.load(args.init_weights, map_location="cpu")
        if isinstance(state, dict) and "student" in state:
            state = state["student"]
        # checkpoint là state_dict của fcos_resnet50_fpn → nạp vào .model của wrapper
        try:
            student.model.load_state_dict(state)
        except RuntimeError:
            # phòng trường hợp checkpoint đã có prefix 'model.'
            student.load_state_dict(state)
        copy_student_to_teacher(rgb_teacher, student)
        copy_student_to_teacher(ir_teacher, student)
        logger.info(f"Đã init student + 2 teacher từ {args.init_weights}")
    else:
        logger.info("Init student từ COCO-pretrained head (slice person/car/bicycle); "
                    "2 teacher = deepcopy student.")

    for m in (student, rgb_teacher, ir_teacher):
        m.to(device)

    optimizer = torch.optim.Adam(
        [p for p in student.parameters() if p.requires_grad], lr=args.lr
    )

    # ── Config ───────────────────────────────────────────────────────────────
    config = TrainingConfig(
        ema=EMAConfig(alpha=args.ema_alpha, use_warmup=True),
        curriculum=CurriculumConfig(
            phase1_end=p1, phase2_end=p2, phase3_end=p3, phase4_end=p4,
        ),
        loss=LossConfig(),                 # trọng số mặc định (rgb_gt=1, mid 0.5/0.5, mid_gt=1, ir=1)
    )
    config.pseudo_label_conf_thresh = args.pseudo_conf
    config.grad_clip = args.grad_clip
    config.device = str(device)
    config.log_interval = args.log_interval

    # ── Adaptive threshold (per-phase, per-teacher) ──────────────────────────
    thresh_sched = None if args.no_adaptive_thresh else AdaptiveThresholdScheduler()

    # ── Evaluator + PhaseEvaluator (auto eval mỗi eval_period + chuyển phase) ─
    iou_list = [0.5 + 0.05 * i for i in range(10)] if args.coco_map else [0.5]
    evaluator = DetectionEvaluator(
        num_classes=NUM_CLASSES, class_names=FLIR_CLASSES,
        iou_thresholds=iou_list, interp="auc",
    )
    phase_eval = PhaseEvaluator(
        evaluator=evaluator,
        ir_val_loader=val_loader,
        device=device,
        eval_every_n=args.eval_period,
        class_names=FLIR_CLASSES,
        thresh_scheduler=thresh_sched,
        log_fn=logger.info,
    )

    best_path = os.path.join(args.output_dir, "best_student.pth")

    def _save_best(results: Dict) -> None:
        torch.save(
            {"student": student.state_dict(),
             "global_step": results.get("global_step"),
             "mAP@0.5": results.get("mAP@0.5")},
            best_path,
        )
        logger.info(f"  ↳ saved best → {best_path}")

    phase_eval.register_best_fn(_save_best)

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = CurriculumDomainAdaptationTrainer(
        student=student,
        rgb_teacher=rgb_teacher,
        ir_teacher=ir_teacher,
        optimizer=optimizer,
        config=config,
        rgb_loader=rgb_loader,
        ir_loader=ir_loader,
        threshold_scheduler=thresh_sched,
        phase_evaluator=phase_eval,
    )

    logger.info("Bắt đầu training ...")
    trainer.train(total_iterations=args.total_iters)

    # ── Eval cuối + lưu last ─────────────────────────────────────────────────
    phase_eval.evaluate(student, trainer.global_step,
                        trainer.scheduler.get_phase(trainer.global_step),
                        trigger_reason="final")
    last_path = os.path.join(args.output_dir, "last_student.pth")
    torch.save({"student": student.state_dict(), "global_step": trainer.global_step}, last_path)
    logger.info(f"Last model → {last_path}")
    phase_eval.print_history()
    logger.info(f"Best mAP@0.5 = {phase_eval.best_map50:.4f}")


if __name__ == "__main__":
    main()
