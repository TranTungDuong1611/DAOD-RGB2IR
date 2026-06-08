"""
training_full_v4.py — GIỐNG HỆT framework DAOD (curriculum RGB→MID→IR, 5 phase,
dual-teacher EMA, adaptive threshold, PhaseEvaluator) NHƯNG ảnh trung gian (MID)
được tạo bằng GAN-IR thay vì SAGA.

Khác biệt DUY NHẤT so với train_kaggle.py:
  - MID domain = blend tuyến tính giữa RGB gốc và ảnh GAN-IR (CycleGAN RGB→IR),
    thay cho SoftSAGA (gray-hoá vùng object).
        mid = alpha · RGB  +  (1 - alpha) · GAN_IR
        alpha = 1.0 → thuần RGB ;  alpha = 0.0 → thuần GAN-IR
    3 mức curriculum tái dùng đúng alpha của SoftSAGAConfig:
        mid_near_rgb    : alpha = soft_saga.alpha_near_rgb     (~0.70)
        mid_intermediate: alpha = soft_saga.alpha_intermediate (~0.50)
        mid_near_ir     : alpha = soft_saga.alpha_near_ir      (~0.25)
  - GAN giữ nguyên hình học → GT box của RGB vẫn hợp lệ trên ảnh MID (như SAGA).

Mọi thứ còn lại (scheduler 5 phase, loss, EMA routing, eval, checkpoint) GIỮ NGUYÊN
bằng cách subclass CurriculumDomainAdaptationTrainer và CHỈ override `_next_mid`.

Chạy (Kaggle):
  cd /kaggle/working/codeRGB_IRDAOD-RGB2IR
  python training_full_v4.py \
    --data-root   /kaggle/input/aligned-flir/aligned_flir/align \
    --gan-weights /kaggle/input/gan-weights/latest_net_G_A.pth \
    --output-dir  /kaggle/working/out_daod_v4 \
    --total-iters 7500 --batch-size 4 --eval-period 500

  # (tùy chọn) khởi tạo student + 2 teacher từ checkpoint RGB đã train:
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

from config import TrainingConfig, EMAConfig, CurriculumConfig, LossConfig
from trainer import CurriculumDomainAdaptationTrainer
from batch_types import MidBatch
from scheduler import DomainStep
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("train_v4")


# ─────────────────────────────────────────────────────────────────────────────
# GAN bridge helper
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def rgb_to_gan_ir_batch(gan: torch.nn.Module, rgb_images: torch.Tensor) -> torch.Tensor:
    """
    rgb_images : [B, 3, H, W] float [0,1]  →  fake-IR [B, 3, H, W] float [0,1].
    CycleGAN G_A: input 3ch [-1,1] → output 1ch [-1,1] (tanh).
    """
    gan_in  = rgb_images * 2.0 - 1.0          # [0,1] → [-1,1]
    gan_out = gan(gan_in)                       # [B,1,H,W] in [-1,1]
    gan_out = (gan_out + 1.0) / 2.0            # → [0,1]
    if gan_out.shape[1] == 1:
        gan_out = gan_out.repeat(1, 3, 1, 1)   # → [B,3,H,W]
    return gan_out.clamp(0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Trainer — giống hệt DAOD, CHỈ override _next_mid để dùng GAN-IR thay SAGA
# ─────────────────────────────────────────────────────────────────────────────

class GANBridgeTrainer(CurriculumDomainAdaptationTrainer):
    """
    CurriculumDomainAdaptationTrainer với ảnh MID tạo bằng blend RGB↔GAN-IR.

    Chỉ override `_next_mid`. Các alpha mức MID tái dùng config.soft_saga (ngữ nghĩa:
    alpha = tỉ lệ RGB trong blend; alpha=1 thuần RGB, alpha=0 thuần GAN-IR).
    """

    def __init__(self, *args, gan: torch.nn.Module, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.gan = gan.to(self.device).eval()
        for p in self.gan.parameters():
            p.requires_grad = False

    def _next_mid(self, mid_level: DomainStep = "mid_near_rgb") -> MidBatch:
        rgb = self._next_rgb()                         # ảnh RGB [0,1] + GT (đã trên device)

        alpha_map = {
            "mid_near_rgb":     self.config.soft_saga.alpha_near_rgb,
            "mid_intermediate": self.config.soft_saga.alpha_intermediate,
            "mid_near_ir":      self.config.soft_saga.alpha_near_ir,
        }
        alpha = alpha_map.get(mid_level, self.config.soft_saga.alpha_near_rgb)

        gan_ir = rgb_to_gan_ir_batch(self.gan, rgb.images)        # [B,3,H,W] [0,1]
        mid_images = alpha * rgb.images + (1.0 - alpha) * gan_ir  # blend, geometry giữ nguyên

        return MidBatch(images=mid_images, targets=rgb.targets, source_images=rgb.images)


# ─────────────────────────────────────────────────────────────────────────────
# Resize ảnh + scale box (giống train_kaggle.py)
# ─────────────────────────────────────────────────────────────────────────────

class ResizeWithBoxes(Dataset):
    """Resize ảnh về (H,W) cố định và scale box theo — đảm bảo batching + bound memory."""

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
        target = dict(item[1])
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
        description="DAOD curriculum (RGB→GAN-IR bridge→IR) trên FLIR-aligned (Kaggle) — v4"
    )
    # Paths
    p.add_argument("--data-root", type=str, required=True,
                   help="Thư mục 'align' (JPEGImages/, Annotations/, align_train.txt, align_validation.txt)")
    p.add_argument("--gan-weights", type=str, required=True,
                   help="CycleGAN G_A checkpoint .pth (RGB→IR). VD: latest_net_G_A.pth")
    p.add_argument("--cyclegan-repo", type=str, default="/kaggle/working/pytorch-CycleGAN-and-pix2pix",
                   help="Repo pytorch-CycleGAN-and-pix2pix (tự clone nếu thiếu)")
    p.add_argument("--output-dir", type=str, default="./out_daod_v4")
    p.add_argument("--init-weights", type=str, default=None,
                   help="(Tùy chọn) state_dict fcos_resnet50_fpn(num_classes=3) để init student+2 teacher.")

    # Scheduler / length
    p.add_argument("--total-iters", type=int, default=7500)
    p.add_argument("--phase-ends", type=str, default=None,
                   help="Override 'p1,p2,p3,p4' (iter). Bỏ → 0.2/0.4/0.65/0.85 của total.")
    p.add_argument("--eval-period", type=int, default=500)
    p.add_argument("--log-interval", type=int, default=50)

    # Model / image
    p.add_argument("--model", type=str, default="faster_rcnn", choices=["fcos", "faster_rcnn"],
                   help="Detector backbone (default: faster_rcnn)")
    p.add_argument("--trainable-backbone-layers", type=int, default=3)
    p.add_argument("--img-height", type=int, default=512)
    p.add_argument("--img-width",  type=int, default=640)

    # Optim / EMA / pseudo
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--ema-alpha", type=float, default=0.999)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--pseudo-conf", type=float, default=0.7)
    p.add_argument("--no-adaptive-thresh", action="store_true")

    # Eval
    p.add_argument("--coco-map", action="store_true",
                   help="Tính mAP@0.5:0.95 (10 ngưỡng IoU) thay vì chỉ mAP@0.5.")
    # Visualization (lưu ảnh infer khi eval)
    p.add_argument("--no-vis", action="store_true",
                   help="Tắt lưu ảnh inference khi eval (mặc định: bật, lưu vào <output-dir>/viz).")
    p.add_argument("--vis-samples", type=int, default=8,
                   help="Số ảnh val IR vẽ GT+pred mỗi lần eval (default: 8)")
    p.add_argument("--vis-score-thresh", type=float, default=0.3,
                   help="Ngưỡng score tối thiểu để vẽ box dự đoán (default: 0.3)")
    p.add_argument("--vis-compare-teachers", action="store_true",
                   help="Vẽ lưới so sánh student vs rgb_teacher vs ir_teacher thay vì chỉ student.")
    return p.parse_args()


def _compute_phase_ends(total: int, override: Optional[str]) -> Tuple[int, int, int, int]:
    if override:
        vals = [int(x) for x in override.split(",")]
        assert len(vals) == 4, "--phase-ends cần đúng 4 số 'p1,p2,p3,p4'"
        return tuple(vals)  # type: ignore
    return (int(0.20 * total), int(0.40 * total), int(0.65 * total), int(0.85 * total))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if not os.path.exists(args.data_root):
        raise FileNotFoundError(f"--data-root: '{args.data_root}' không tồn tại")
    if not os.path.exists(args.gan_weights):
        raise FileNotFoundError(f"--gan-weights: '{args.gan_weights}' không tồn tại")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── CycleGAN repo + GAN ──────────────────────────────────────────────────
    repo = args.cyclegan_repo
    if not os.path.isdir(repo):
        logger.info(f"CycleGAN repo not found at {repo}, cloning...")
        os.system(f"git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git {repo}")
    sys.path.insert(0, repo)
    from models import networks  # noqa: E402

    gan = networks.define_G(input_nc=3, output_nc=1, ngf=64, netG="resnet_9blocks", norm="instance")
    gan.load_state_dict(torch.load(args.gan_weights, map_location="cpu"))
    gan.eval()
    for prm in gan.parameters():
        prm.requires_grad = False
    logger.info(f"GAN loaded (frozen) ← {args.gan_weights}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size_hw = (args.img_height, args.img_width)
    p1, p2, p3, p4 = _compute_phase_ends(args.total_iters, args.phase_ends)
    logger.info(f"device={device}  data_root={args.data_root}")
    logger.info(f"total_iters={args.total_iters}  phase_ends=({p1},{p2},{p3},{p4})  "
                f"img={size_hw}  batch={args.batch_size}  MID=GAN-IR blend")

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
                            collate_fn=ir_val_collate, num_workers=args.num_workers, pin_memory=True)

    # ── Model trio (student + 2 teacher giống hệt) ───────────────────────────
    from_coco = args.init_weights is None
    build_trio = build_faster_rcnn_trio if args.model == "faster_rcnn" else build_fcos_trio
    logger.info(f"Detector: {args.model}")
    student, rgb_teacher, ir_teacher = build_trio(
        num_classes=NUM_CLASSES, pretrained_backbone=True,
        trainable_backbone_layers=args.trainable_backbone_layers,
        min_size=min(size_hw), max_size=max(size_hw), ir_to_rgb=True,
        from_coco=from_coco, coco_src_indices=FLIR_TO_COCO_IDX if from_coco else None,
    )

    if args.init_weights:
        if not os.path.exists(args.init_weights):
            raise FileNotFoundError(f"--init-weights: '{args.init_weights}' không tồn tại")
        state = torch.load(args.init_weights, map_location="cpu")
        if isinstance(state, dict) and "student" in state:
            state = state["student"]
        try:
            student.model.load_state_dict(state)
        except RuntimeError:
            student.load_state_dict(state)
        copy_student_to_teacher(rgb_teacher, student)
        copy_student_to_teacher(ir_teacher, student)
        logger.info(f"Init student + 2 teacher từ {args.init_weights}")
    else:
        logger.info("Init student từ COCO-pretrained head (slice person/car/bicycle); "
                    "2 teacher = deepcopy student.")

    for m in (student, rgb_teacher, ir_teacher):
        m.to(device)

    optimizer = torch.optim.Adam([p for p in student.parameters() if p.requires_grad], lr=args.lr)

    # ── Config ───────────────────────────────────────────────────────────────
    config = TrainingConfig(
        ema=EMAConfig(alpha=args.ema_alpha, use_warmup=True),
        curriculum=CurriculumConfig(phase1_end=p1, phase2_end=p2, phase3_end=p3, phase4_end=p4),
        loss=LossConfig(),
    )
    config.pseudo_label_conf_thresh = args.pseudo_conf
    config.grad_clip = args.grad_clip
    config.device = str(device)
    config.log_interval = args.log_interval

    thresh_sched = None if args.no_adaptive_thresh else AdaptiveThresholdScheduler()

    iou_list = [0.5 + 0.05 * i for i in range(10)] if args.coco_map else [0.5]
    evaluator = DetectionEvaluator(num_classes=NUM_CLASSES, class_names=FLIR_CLASSES,
                                   iou_thresholds=iou_list, interp="auc")

    # Lưu ảnh inference mỗi lần eval → <output-dir>/viz/step*.png (GT xanh, pred đỏ)
    vis_dir = None if args.no_vis else os.path.join(args.output_dir, "viz")
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)
        logger.info(f"Vis eval images → {vis_dir}  ({args.vis_samples} ảnh/lần, "
                    f"thresh={args.vis_score_thresh}"
                    f"{', so sánh teachers' if args.vis_compare_teachers else ''})")

    phase_eval = PhaseEvaluator(
        evaluator=evaluator, ir_val_loader=val_loader, device=device,
        eval_every_n=args.eval_period, class_names=FLIR_CLASSES,
        thresh_scheduler=thresh_sched, log_fn=logger.info,
        vis_dir=vis_dir,
        vis_num_samples=args.vis_samples,
        vis_score_thresh=args.vis_score_thresh,
        # Khi bật so sánh, truyền teachers → vẽ lưới student/rgb_teacher/ir_teacher
        rgb_teacher=rgb_teacher if args.vis_compare_teachers else None,
        ir_teacher=ir_teacher if args.vis_compare_teachers else None,
    )

    best_path = os.path.join(args.output_dir, "best_student.pth")

    def _save_best(results: Dict) -> None:
        torch.save({"student": student.state_dict(),
                    "global_step": results.get("global_step"),
                    "mAP@0.5": results.get("mAP@0.5")}, best_path)
        logger.info(f"  ↳ saved best → {best_path}")

    phase_eval.register_best_fn(_save_best)

    # ── Trainer (GAN bridge) ─────────────────────────────────────────────────
    trainer = GANBridgeTrainer(
        student=student, rgb_teacher=rgb_teacher, ir_teacher=ir_teacher,
        optimizer=optimizer, config=config,
        rgb_loader=rgb_loader, ir_loader=ir_loader,
        threshold_scheduler=thresh_sched, phase_evaluator=phase_eval,
        gan=gan,
    )

    logger.info("Bắt đầu training (MID = GAN-IR blend) ...")
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
