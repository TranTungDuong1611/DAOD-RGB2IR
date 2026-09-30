"""
training_full_d3t_v2_method1_direct_gate.py — PHƯƠNG PHÁP 1: DIRECT DIFFERENTIABLE GATE
(Layer-wise Adaptation Gate) đặt TRÊN nền block scheduler + D3T dense distillation.

== Ý tưởng (theo method_1_direct_differentiable_gate.html) ==
Với MỖI conv của student, trọng số hiệu dụng được tái tham số hoá:

    W_l^eff = W_l^RGB (đóng băng) + σ(a_l) · ΔW_l

  • W_l^RGB : trọng số nguồn (= teacher RGB, cũng là init student) — GIỮ CỐ ĐỊNH.
  • ΔW_l    : phần thay đổi, HỌC ĐƯỢC (khởi tạo ~N(0,1e-4)).
  • a_l     : gate logit của NHÓM block l; g_l = σ(a_l) ∈ [0,1] (init a_l=-2 → g≈0.12).
  • g_l≈0 → block giữ nguyên kiến thức RGB; g_l≈1 → block thích nghi mạnh sang IR.

Gate + delta được HỌC TRỰC TIẾP bằng chính loss detection/distillation (một backward,
KHÔNG cần meta-learning). Chống suy biến bằng regularizer:

    L = L_det + λ_g·L_gate + λ_Δ·L_Δ + λ_F·L_Fisher

  • L_det    : GIỮ NGUYÊN từ pipeline gốc = loss_sup + λ·loss_distill (D3T dense).
  • L_gate  = mean_l σ(a_l)                    → phạt bật gate (thưa hoá adaptation).
  • L_Δ     = Σ_l ‖ΔW_l‖²                       → chống nhập nhằng scale g·ΔW.
  • L_Fisher= Σ_i F_i·(σ(a)·ΔW_i)²  (tùy chọn) → giữ trọng số quan trọng với RGB.

8 nhóm gate (Stem, C2, C3, C4, C5, FPN, Cls-head, Reg-head). Khi kết thúc in giá trị
gate để biết layer nào cần adaptation.

Cài đặt: dùng torch.nn.utils.parametrize chèn W_eff = base + σ(a)·ΔW vào từng conv mà
KHÔNG phải viết lại forward torchvision FCOS. base đóng băng; chỉ ΔW + gate logits (và
bias/BN nếu bật --train-others) là trainable.

== Nền kế thừa: block scheduler + D3T dense distillation ==

== Khác biệt so với training_full_d3t_v2_blocksched.py ==
Thay cơ chế UNSUPERVISED: bỏ hard pseudo-label (lọc box theo ngưỡng conf 0.5 rồi
train như GT) → dùng DENSE DISTILLATION đúng như repo gốc D3T:
  • Lấy output DÀY ĐẶC của head torchvision FCOS (cls_logits, bbox_regression,
    bbox_ctrness) cho student (ảnh strong) và teacher (ảnh weak) — KHÔNG NMS.
  • Region selection theo Harmony Measure hm = cls_prob^α · iou^β, chọn top RATIO.
  • Uncertainty weighting loss_weight = exp(-(1-hm)).
  • 4 loss mềm: QFLv2 (logits, ×4) + GIoU (deltas, ×1) + BCE (quality/ctrness, ×1)
    + VFL/HMfocalLoss (uhl, ×1).
  • nhánh quality của torchvision = CENTERNESS (đóng vai "iou" trong hm — torchvision
    FCOS không có nhánh IoU riêng như cvpods).
  • ir_gan coi như gán với IR-teacher (t_match = teacher_ir), như blocksched.
Phần supervised (rgb, ir_gan có GT) và lịch BlockDomainScheduler giữ nguyên.

== [gốc] training_full_d3t_v2_blocksched.py — D3T-style RGB→IR + lịch domain THEO KHỐI LIỀN.

== Khác biệt DUY NHẤT so với training_full_d3t_v2_base.py ==

Thay BridgeDomainScheduler (error-diffusion — xen kẽ từng iter theo tỉ lệ) bằng
BlockDomainScheduler theo CHU KỲ NGẮN: trong MỖI phase, các iter được chia thành
nhiều CHU KỲ độ dài `--cycle-iters` (mặc định 200). Mỗi chu kỳ lại chia theo tỉ
lệ (rgb : ir_gan : ir_real) của phase thành BA KHỐI LIỀN, train tuần tự HẾT rgb →
HẾT ir_gan → HẾT ir_real TRONG chu kỳ đó, rồi LẶP sang chu kỳ kế tiếp (giống cấu
trúc chu kỳ 200-iter của D3T). Nhờ vậy khối ir_real không bao giờ dài quá một chu
kỳ → luôn quay lại neo rgb sau mỗi cycle. Mọi phần còn lại giữ nguyên bản base.

== v2: SỬA TRIỂN KHAI DỮ LIỆU CHO ĐÚNG PAPER (arXiv:2403.09359, mục 4.1) ==

Khác biệt DUY NHẤT so với training_full_d3t.py là cách bố trí dữ liệu train/test:

  • DISJOINT SPLIT (mới): Paper dùng RGB source và thermal target KHÔNG trùng
    ảnh ("first 2,064 RGB images as source domain and a separate set of 2,064
    thermal target images for training ... preventing overfitting"). FLIR vốn
    là ảnh aligned (RGB-IR từng cặp khớp pixel) nên nếu RGB source và real-IR
    target dùng chung danh sách ảnh thì model "ăn gian" qua sự tương ứng cảnh.
    → v2 chia align_train.txt thành 2 nửa RỜI NHAU:
        - nửa A  : RGB source (và ir_gan sinh ra TỪ nửa A — bridge, có chủ đích)
        - nửa B  : ir_real target (không giao với nửa A)
    ir_gan vẫn là cặp khớp của RGB (đó là vai trò cầu nối), nhưng REAL-IR target
    giờ tách rời khỏi RGB source — đúng tinh thần disjoint của paper.

  • TEST (giữ đúng paper): đánh giá trên ảnh THERMAL có nhãn (align_validation),
    đã loại ảnh không nhãn — khớp giao thức "RGB→Thermal evaluation" của paper.

Mọi phần còn lại GIỮ NGUYÊN training_full_d3t.py:
  • KHÔNG warmup: hai teacher pretrain sẵn (teacher_rgb trên RGB, teacher_ir
    trên ảnh GAN-IR); student init = đúng weight teacher_rgb.
  • Ba domain (rgb, ir_gan, ir_real), ir_gan làm bridge; tỉ lệ đổi theo phase.
  • λ động cho pseudo-label; ir_gan dùng cả supervised + unsupervised.
  • Unsupervised loss có trọng số: teacher cùng domain (w_match) > chéo (w_cross).
  • Ảnh đưa vào FCOS ở [0,1]; FCOS tự normalize; min_size=max_size=img_size.

Usage (Kaggle):
    python training_full_d3t_v2.py \
        --data-root           /kaggle/input/.../align \
        --gan-weights         /kaggle/input/.../latest_net_G_A.pth \
        --teacher-rgb-weights /kaggle/input/.../best_rgb.pth \
        --teacher-ir-weights  /kaggle/input/.../best_ir_gan.pth \
        --total-iters 20000 --batch-size 8 --img-size 640
"""

import os
import sys
import random
import json
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P
import torchvision.transforms.v2 as T
from torchvision import tv_tensors
from torchvision.models.detection import fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights
from torchvision.ops import generalized_box_iou_loss, box_iou
from PIL import Image
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

FLIR_CLASSES: List[str] = ["person", "car", "bicycle"]
FLIR_CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(FLIR_CLASSES)}
NUM_CLASSES = len(FLIR_CLASSES)
_IGNORE_LABELS = {"FLIR", "dog"}


# ─────────────────────────────────────────────────────────────────────────────
# VOC XML helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_voc_xml(xml_path: Path) -> List[Dict]:
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return []
    root = tree.getroot()
    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name", default="").strip()
        if name in _IGNORE_LABELS or name not in FLIR_CLASS_TO_IDX:
            continue
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        try:
            xmin = float(bnd.findtext("xmin"))
            ymin = float(bnd.findtext("ymin"))
            xmax = float(bnd.findtext("xmax"))
            ymax = float(bnd.findtext("ymax"))
        except (TypeError, ValueError):
            continue
        if xmax > xmin and ymax > ymin:
            objects.append({"label": name, "box": [xmin, ymin, xmax, ymax]})
    return objects


def _objects_to_tensors(objects: List[Dict], min_area: float = 16.0):
    boxes, labels = [], []
    for obj in objects:
        x1, y1, x2, y2 = obj["box"]
        if (x2 - x1) * (y2 - y1) < min_area:
            continue
        boxes.append([x1, y1, x2, y2])
        labels.append(FLIR_CLASS_TO_IDX[obj["label"]])
    if boxes:
        return (torch.as_tensor(boxes, dtype=torch.float32),
                torch.as_tensor(labels, dtype=torch.int64))
    return (torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0,),   dtype=torch.int64))


def _ir_stem_to_rgb_filename(stem: str) -> str:
    return stem.replace("_PreviewData", "") + "_RGB.jpg"


_IR_EXTS = (".jpeg", ".jpg", ".png")

def _find_image(folder: Path, stem: str) -> Optional[Path]:
    for ext in _IR_EXTS:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _read_split_file(split_file: Path) -> List[str]:
    with open(split_file, "r") as f:
        return [line.strip() for line in f if line.strip()]


def _split_disjoint(stems: List[str], source_frac: float = 0.5,
                    seed: int = 42) -> Tuple[List[str], List[str]]:
    """
    Tách danh sách stems thành 2 phần KHÔNG giao nhau: (source, target).
    Theo paper (mục 4.1): RGB source và thermal target không được dùng cặp ảnh
    trùng nhau → chống overfit trên dữ liệu aligned.
      - seed >= 0 : xáo trộn xác định (cân bằng ngày/đêm) rồi cắt.
      - seed <  0 : giữ thứ tự gốc, lấy "first N" như mô tả paper.
    """
    s = list(stems)
    if seed >= 0:
        random.Random(seed).shuffle(s)
    k = int(round(len(s) * source_frac))
    return s[:k], s[k:]


# ─────────────────────────────────────────────────────────────────────────────
# Transforms — output [0,1]; KHÔNG normalize (FCOS tự normalize nội bộ).
#   weak   : chỉ geometric (resize + flip) → dùng cho teacher sinh pseudo + sup.
#   strong : chỉ photometric → dùng cho student (consistency), box giữ nguyên.
# ─────────────────────────────────────────────────────────────────────────────

def get_rgb_weak_transform(img_size: int = 640) -> T.Compose:
    return T.Compose([
        T.ToImage(),
        T.Resize((img_size, img_size)),
        T.ToDtype(torch.float32, scale=True),
        T.RandomHorizontalFlip(p=0.5),
    ])


def get_rgb_strong_transform() -> T.Compose:
    """Photometric only — áp trên tensor [0,1], không đổi toạ độ box."""
    return T.Compose([
        T.ColorJitter(0.4, 0.4, 0.4, 0.1),
        T.RandomGrayscale(p=0.2),
        T.GaussianBlur(kernel_size=5),
    ])


def get_ir_weak_transform(img_size: int = 640) -> T.Compose:
    return T.Compose([
        T.ToImage(),
        T.Resize((img_size, img_size)),
        T.ToDtype(torch.float32, scale=True),
        T.Grayscale(num_output_channels=3),
        T.RandomHorizontalFlip(p=0.5),
    ])


def get_ir_val_transform(img_size: int = 640) -> T.Compose:
    return T.Compose([
        T.ToImage(),
        T.Resize((img_size, img_size)),
        T.ToDtype(torch.float32, scale=True),
        T.Grayscale(num_output_channels=3),
    ])


def get_ir_strong_transform() -> T.Compose:
    return T.Compose([
        T.GaussianBlur(kernel_size=5),
        T.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────────────────────────────────────

class FLIRRGBDataset(Dataset):
    """Labeled RGB source. Returns (image [0,1] 3ch, target có GT).

    v2: có thể truyền `stems` (danh sách stem cụ thể) để dùng phần disjoint;
    nếu None thì đọc toàn bộ align_{split}.txt như cũ.
    """

    def __init__(self, root: str, split: str = "train",
                 transform: Optional[Callable] = None, min_area: float = 16.0,
                 img_size: int = 640, stems: Optional[List[str]] = None) -> None:
        self.root      = Path(root)
        self.transform = transform or get_rgb_weak_transform(img_size)
        self.min_area  = min_area
        base = stems if stems is not None else _read_split_file(self.root / f"align_{split}.txt")
        self.stems = [
            s for s in base
            if (self.root / "JPEGImages" / _ir_stem_to_rgb_filename(s)).exists()
        ]

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        stem     = self.stems[idx]
        img      = Image.open(self.root / "JPEGImages" / _ir_stem_to_rgb_filename(stem)).convert("RGB")
        ann_path = self.root / "Annotations" / f"{stem}.xml"
        boxes, labels = _objects_to_tensors(_parse_voc_xml(ann_path), self.min_area)
        boxes_tv = tv_tensors.BoundingBoxes(
            boxes, format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(img.height, img.width),
        )
        img_t, target = self.transform(img, {"boxes": boxes_tv, "labels": labels})
        target["boxes"] = torch.as_tensor(target["boxes"], dtype=torch.float32)
        target["stem"]  = stem
        return img_t, target


class FLIRIRDataset(Dataset):
    """Unlabeled real IR target — training. Returns image [0,1] 3ch.

    v2: có thể truyền `stems` để dùng phần disjoint (tách khỏi RGB source).
    """

    def __init__(self, root: str, split: str = "train",
                 transform: Optional[Callable] = None, img_size: int = 640,
                 stems: Optional[List[str]] = None) -> None:
        self.root      = Path(root)
        self.transform = transform or get_ir_weak_transform(img_size)
        img_dir        = self.root / "JPEGImages"
        base = stems if stems is not None else _read_split_file(self.root / f"align_{split}.txt")
        self.ir_paths  = [p for s in base if (p := _find_image(img_dir, s)) is not None]

    def __len__(self) -> int:
        return len(self.ir_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.ir_paths[idx])
        return self.transform(img)


class FLIRIRValDataset(Dataset):
    """Labeled real IR — evaluation only. Returns (image [0,1] 3ch, target).

    Đã loại ảnh không có nhãn (giống paper: 'removed images without labels').
    """

    def __init__(self, root: str, split: str = "validation",
                 transform: Optional[Callable] = None, min_area: float = 16.0,
                 img_size: int = 640) -> None:
        self.root      = Path(root)
        self.transform = transform or get_ir_val_transform(img_size)
        self.min_area  = min_area
        split_file     = self.root / f"align_{split}.txt"
        self.img_dir   = self.root / "JPEGImages"
        self.stems     = [s for s in _read_split_file(split_file)
                          if _find_image(self.img_dir, s) is not None
                          and (self.root / "Annotations" / f"{s}.xml").exists()]

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        stem     = self.stems[idx]
        img      = Image.open(_find_image(self.img_dir, stem))
        ann_path = self.root / "Annotations" / f"{stem}.xml"
        boxes, labels = _objects_to_tensors(_parse_voc_xml(ann_path), self.min_area)
        boxes_tv = tv_tensors.BoundingBoxes(
            boxes, format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(img.height, img.width),
        )
        img_t, target = self.transform(img, {"boxes": boxes_tv, "labels": labels})
        target["boxes"] = torch.as_tensor(target["boxes"], dtype=torch.float32)
        target["stem"]  = stem
        return img_t, target


def rgb_collate(batch: List) -> Tuple[List, List]:
    return [b[0] for b in batch], [b[1] for b in batch]


def ir_collate(batch: List) -> List:
    return list(batch)


def ir_val_collate(batch: List) -> Tuple[List, List]:
    return [b[0] for b in batch], [b[1] for b in batch]


# ─────────────────────────────────────────────────────────────────────────────
# EMA + pseudo-label filtering
# ─────────────────────────────────────────────────────────────────────────────

def update_ema(teacher: torch.nn.Module, student: torch.nn.Module,
               alpha: float = 0.9996) -> None:
    """teacher ← α·teacher + (1-α)·student (Mean Teacher EMA)."""
    for t_p, s_p in zip(teacher.parameters(), student.parameters()):
        t_p.data.mul_(alpha).add_(s_p.data, alpha=1.0 - alpha)


@torch.no_grad()
def update_ema_effective(teacher: torch.nn.Module, student: torch.nn.Module,
                        alpha: float = 0.9996) -> None:
    """EMA teacher ← α·teacher + (1-α)·student_EFFECTIVE (khớp theo TÊN tham số).

    Method 1 tái tham số hoá tham số của student (base+σ(a)·Δ) nên
    student.parameters() lệch cấu trúc với teacher → không thể zip trực tiếp. Ở đây
    với mỗi tham số của teacher, lấy giá trị HIỆU DỤNG tương ứng của student.

    getattr() trả thẳng giá trị hiệu dụng nếu attr đã được parametrize (base+σ(a)·Δ)
    và trả giá trị thường nếu không → dùng chung một đường cho cả `weight` lẫn `bias`.
    """
    for name, t_p in teacher.named_parameters():
        mod_name, _, attr = name.rpartition(".")
        s_mod = student.get_submodule(mod_name)
        s_val = getattr(s_mod, attr)
        t_p.data.mul_(alpha).add_(s_val.detach(), alpha=1.0 - alpha)


def filter_pseudo_labels(preds: List[Dict], threshold: float = 0.5) -> List[Dict]:
    return [
        {
            "boxes":  p["boxes"] [p["scores"] >= threshold],
            "labels": p["labels"][p["scores"] >= threshold],
        }
        for p in preds
    ]


# ─────────────────────────────────────────────────────────────────────────────
# DENSE DISTILLATION — port GIỐNG HỆT D3T (repo gốc: experiment/flir_rgb2thermal).
#
# torchvision FCOS mặc định chỉ trả loss (train) / detection (eval). Ở đây ta "tap"
# thẳng backbone + head để lấy output DÀY ĐẶC theo từng vị trí feature-map:
#   cls_logits      (N, HWA, K)
#   bbox_regression (N, HWA, 4)   — ltrb đã relu, tương ứng deltas của D3T
#   bbox_ctrness    (N, HWA, 1)   — nhánh centerness, đóng vai "iou/quality" của D3T
# và decode ra box xyxy (cho loss UHL).  Vì strong-aug là photometric-only, lưới
# vị trí của teacher (weak) và student (strong) TRÙNG KHỚP → distill hợp lệ.
# ─────────────────────────────────────────────────────────────────────────────

def dense_forward(model: torch.nn.Module, images: List[torch.Tensor]):
    """Trả (cls_logits, bbox_reg, bbox_ctrness, boxes_xyxy, anchors) dày đặc.

    boxes_xyxy, anchors: (N, HWA, 4).  Dùng cho student (có grad) & teacher (no_grad).
    """
    from collections import OrderedDict
    images_t, _ = model.transform(images, None)
    feats = model.backbone(images_t.tensors)
    if isinstance(feats, torch.Tensor):
        feats = OrderedDict([("0", feats)])
    feats = list(feats.values())
    ho = model.head(feats)
    anchors = model.anchor_generator(images_t, feats)   # list[N] mỗi phần tử (HWA,4)
    anchors = torch.stack(anchors)                       # (N, HWA, 4)
    boxes = model.box_coder.decode(ho["bbox_regression"], anchors)  # (N, HWA, 4)
    return ho["cls_logits"], ho["bbox_regression"], ho["bbox_ctrness"], boxes, anchors


def QFLv2(pred_sigmoid, teacher_sigmoid, weight=None, beta=2.0, reduction="sum"):
    """Quality Focal Loss v2 — port đúng D3T. weight chỉ dùng làm mask pos/valid."""
    pt = pred_sigmoid
    zerolabel = pt.new_zeros(pt.shape)
    loss = F.binary_cross_entropy(pred_sigmoid, zerolabel, reduction="none") * pt.pow(beta)
    pos = weight > 0
    pt = teacher_sigmoid[pos] - pred_sigmoid[pos]
    loss[pos] = F.binary_cross_entropy(
        pred_sigmoid[pos], teacher_sigmoid[pos], reduction="none") * pt.pow(beta)
    valid = weight >= 0
    if reduction == "mean":
        loss = loss[valid].mean()
    elif reduction == "sum":
        loss = loss[valid].sum()
    return loss


def vfl_loss(pred, target, alpha=0.75, gamma=2.0):
    """Varifocal Loss (HMfocalLoss, weight_type='iou') — port đúng D3T, reduction=mean."""
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    focal_weight = target * (target > 0.0).float() + \
        alpha * (pred_sigmoid - target).abs().pow(gamma) * (target <= 0.0).float()
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none") * focal_weight
    return loss.mean()


def get_distill_loss(student_out, teacher_out, num_classes: int,
                     ratio: float, hm_alpha: float, hm_beta: float,
                     un_alpha: float, w_logits: float, w_deltas: float,
                     w_quality: float, w_uhl: float) -> torch.Tensor:
    """Tổng 4 loss distillation của D3T (đã nhân trọng số). Trả 1 scalar."""
    s_logits, s_deltas, s_ctr, s_boxes, _        = student_out
    t_logits, t_deltas, t_ctr, t_boxes, _        = teacher_out
    N = s_logits.shape[0]

    s_logits_f = s_logits.reshape(-1, num_classes)
    t_logits_f = t_logits.reshape(-1, num_classes)
    s_ctr_f    = s_ctr.reshape(-1, 1)
    t_ctr_f    = t_ctr.reshape(-1, 1)
    s_boxes_f  = s_boxes.reshape(-1, 4)
    t_boxes_f  = t_boxes.reshape(-1, 4)

    # ── Region selection theo harmony measure (chỉ dùng teacher, no grad) ──────
    with torch.no_grad():
        count_num = max(1, int(t_logits_f.size(0) * ratio))
        t_probs   = t_logits_f.sigmoid()
        cls_prob  = t_probs.max(1)[0]
        iou       = t_ctr_f.sigmoid().squeeze(1)
        hm        = cls_prob.pow(hm_alpha) * iou.pow(hm_beta)
        sorted_vals, sorted_inds = torch.topk(hm, hm.numel())
        mask = torch.zeros_like(hm)
        mask[sorted_inds[:count_num]] = 1.0
        fg_num = sorted_vals[:count_num].sum().clamp(min=1.0)
        b_mask = mask > 0.0

    # ── Uncertainty weighting (detach) ────────────────────────────────────────
    cls_prob    = t_probs.max(1)[0]
    iou         = t_ctr_f.sigmoid().squeeze(1)
    hm          = cls_prob.pow(hm_alpha) * iou.pow(hm_beta)
    loss_weight = torch.exp(-((1.0 - hm) / un_alpha)).detach()

    # ── 1) logits: QFLv2 (dense, chuẩn hoá bằng fg_num) ────────────────────────
    loss_logits = QFLv2(s_logits_f.sigmoid(), t_probs,
                        weight=loss_weight, reduction="sum") / fg_num

    # ── 2) deltas: GIoU giữa box decode student↔teacher (dense, có trọng số) ───
    giou = generalized_box_iou_loss(s_boxes_f, t_boxes_f, reduction="none", eps=1e-7)
    loss_deltas = (giou * loss_weight).mean()

    # ── 3) quality: BCE centerness student↔teacher (dense, có trọng số) ────────
    loss_quality = F.binary_cross_entropy(
        s_ctr_f.sigmoid(), t_ctr_f.sigmoid(),
        weight=loss_weight.unsqueeze(1), reduction="mean")

    # ── 4) UHL: VFL với target = IoU với box lân cận, trên top-RATIO vùng ──────
    n_b_mask   = b_mask.view(N, -1)
    n_s_logits = s_logits                          # (N, HWA, K)
    t_probs_n  = t_probs.view(N, -1, num_classes)
    logits_list, targets_list = [], []
    for idx in range(N):
        m = n_b_mask[idx]
        if m.sum() <= 1:
            continue
        pos_boxes = s_boxes[idx][m].detach()
        ious = box_iou(pos_boxes, pos_boxes)
        ious.fill_diagonal_(0)
        ious_max, _ = ious.max(dim=1)
        loss_mask = ious_max > 0
        cls_iou_targets = torch.zeros_like(n_s_logits[idx][m])
        pseudo_labels = t_probs_n[idx].max(1)[1][m]
        cls_iou_targets[loss_mask, pseudo_labels[loss_mask]] = ious_max[loss_mask]
        logits_list.append(n_s_logits[idx][m][loss_mask])
        targets_list.append(cls_iou_targets[loss_mask])

    if logits_list and sum(t.shape[0] for t in logits_list) > 0:
        loss_uhl = vfl_loss(torch.cat(logits_list), torch.cat(targets_list))
    else:
        loss_uhl = s_logits.sum() * 0.0

    return (w_logits * loss_logits + w_deltas * loss_deltas
            + w_quality * loss_quality + w_uhl * loss_uhl)


def distill(student_out, teacher_out, cfg: Dict) -> torch.Tensor:
    """Wrapper gọn: gọi get_distill_loss với cấu hình trong dict cfg."""
    return get_distill_loss(
        student_out, teacher_out, num_classes=NUM_CLASSES,
        ratio=cfg["ratio"], hm_alpha=cfg["hm_alpha"], hm_beta=cfg["hm_beta"],
        un_alpha=cfg["un_alpha"], w_logits=cfg["w_logits"], w_deltas=cfg["w_deltas"],
        w_quality=cfg["w_quality"], w_uhl=cfg["w_uhl"])


# ─────────────────────────────────────────────────────────────────────────────
# PHƯƠNG PHÁP 1 — DIRECT DIFFERENTIABLE GATE
#   W_eff = W_rgb(frozen) + σ(a_group) · ΔW      (chèn qua torch.nn.utils.parametrize)
# 8 nhóm gate; mỗi conv thuộc đúng 1 nhóm theo tên tham số.
# ─────────────────────────────────────────────────────────────────────────────

GROUP_NAMES = ["stem", "C2", "C3", "C4", "C5", "FPN", "cls_head", "reg_head"]
N_GROUPS = len(GROUP_NAMES)


def group_id_for_name(name: str) -> int:
    """Ánh xạ tên tham số (torchvision FCOS ResNet50-FPN) → id nhóm gate 0..7.

    Trả -1 nếu không thuộc nhóm nào cần gate (sẽ để nguyên, không parametrize).
    """
    if "backbone.body" in name:
        if "layer1" in name:  return 1   # C2
        if "layer2" in name:  return 2   # C3
        if "layer3" in name:  return 3   # C4
        if "layer4" in name:  return 4   # C5
        return 0                          # conv1/bn1 stem (mọi thứ còn lại của body)
    if "backbone.fpn" in name:            return 5
    if "head.classification_head" in name: return 6
    if "head.regression_head" in name:     return 7
    return -1


class GateBank(nn.Module):
    """Giữ N_GROUPS gate logit a_l (init = gate_init). g_l = σ(a_l)."""

    def __init__(self, n_groups: int = N_GROUPS, gate_init: float = -2.0) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.full((n_groups,), float(gate_init)))

    def gates(self) -> torch.Tensor:
        return torch.sigmoid(self.logits)


class DirectGate(nn.Module):
    """Parametrization: nhận base (đóng băng) → trả base + σ(a_group)·ΔW.

    Tham chiếu GateBank qua list bọc ([bank]) để KHÔNG bị nn.Module đăng ký lại
    logits (tránh nhân đôi tham số trong optimizer).
    """

    def __init__(self, base_shape, bank: GateBank, group_id: int,
                 delta_std: float = 1e-4, device=None) -> None:
        super().__init__()
        self._bank_ref = [bank]           # bọc list → không auto-register submodule
        self.group_id = int(group_id)
        # tạo ΔW thẳng trên device của conv (register_parametrization sẽ eval forward
        # ngay lúc đăng ký → base ở cuda, delta phải cùng device, tránh lỗi mismatch)
        if delta_std > 0:
            self.delta = nn.Parameter(torch.randn(base_shape, device=device) * delta_std)
        else:
            self.delta = nn.Parameter(torch.zeros(base_shape, device=device))

    def forward(self, base: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self._bank_ref[0].logits[self.group_id])
        return base + g * self.delta


def install_direct_gates(model: nn.Module, gate_init: float = -2.0,
                         delta_std: float = 1e-4, train_others: bool = False,
                         gate_bias: str = "off"):
    """Chèn parametrization X_eff = base + σ(a)·ΔX vào tham số của student.

    LUÔN gate `weight` của mọi Conv2d thuộc 8 nhóm. `gate_bias` mở rộng thêm:
      "off"  : chỉ conv weight — hành vi CŨ, không đổi gì (mặc định).
      "conv" : gate thêm `bias` của Conv2d (FPN + 3 conv cuối head).
      "all"  : gate thêm cả affine (weight/bias) của GroupNorm trong head.

    Bias/affine dùng CHUNG gate a_l của nhóm → vẫn đúng N_GROUPS nhóm, GateBank và
    L_gate/L_Δ/L_Fisher không phải đổi.

    FrozenBatchNorm2d KHÔNG BAO GIỜ được gate ở bất kỳ chế độ nào: weight/bias/
    running_* của nó là BUFFER mang thống kê ImageNet, giữ nguyên cố định. Vòng lặp
    dưới chỉ xét Conv2d và GroupNorm nên BN không thể lọt vào.

    - Đóng băng mọi base (`parametrizations.<attr>.original`) và (mặc định) phần còn lại.
    - Trả (GateBank đã gắn làm model.gate_bank, tổng số tensor được gate).
    """
    # tạo bank trên đúng device của model (logits dùng trong forward parametrization,
    # được eval ngay lúc register → phải cùng device với base weight)
    device = next(model.parameters()).device
    bank = GateBank(N_GROUPS, gate_init).to(device)
    model.add_module("gate_bank", bank)

    # thu thập (module, attr, group_id) TRƯỚC khi register (tránh đổi tên khi lặp)
    targets = []
    for mod_name, module in model.named_modules():
        gid = group_id_for_name(mod_name)
        if gid < 0:
            continue
        if isinstance(module, nn.Conv2d):
            targets.append((module, "weight", gid))
            if gate_bias in ("conv", "all") and module.bias is not None:
                targets.append((module, "bias", gid))
        elif gate_bias == "all" and isinstance(module, nn.GroupNorm):
            for attr in ("weight", "bias"):
                if getattr(module, attr, None) is not None:
                    targets.append((module, attr, gid))
        # mọi module khác (FrozenBatchNorm2d, …) bị bỏ qua — KHÔNG gate

    for module, attr, gid in targets:
        t = getattr(module, attr)
        P.register_parametrization(
            module, attr, DirectGate(t.shape, bank, gid, delta_std, device=t.device))
        # base (original) đóng băng
        getattr(module.parametrizations, attr).original.requires_grad_(False)

    # Đóng băng mọi tham số KHÔNG phải delta / gate trừ khi train_others
    for name, prm in model.named_parameters():
        is_delta = ".parametrizations." in name and name.endswith(".0.delta")
        is_gate  = name.startswith("gate_bank.")
        if is_delta or is_gate:
            prm.requires_grad_(True)
        elif not train_others:
            prm.requires_grad_(False)
    return bank, len(targets)


def count_gated(model: nn.Module) -> Dict[str, int]:
    """Đếm số tensor được gate theo loại — dùng để in log cho rõ."""
    out = {"conv.weight": 0, "conv.bias": 0, "norm.affine": 0}
    mods = dict(model.named_modules())
    for name, _ in model.named_parameters():
        if not (".parametrizations." in name and name.endswith(".0.delta")):
            continue
        owner, _, tail = name.rpartition(".parametrizations.")
        attr = tail.split(".")[0]
        mod = mods.get(owner)
        if isinstance(mod, nn.Conv2d):
            out["conv.weight" if attr == "weight" else "conv.bias"] += 1
        else:
            out["norm.affine"] += 1
    return out


def gate_reg_loss(bank: GateBank) -> torch.Tensor:
    """L_gate = mean_l σ(a_l) — phạt bật gate."""
    return bank.gates().mean()


def delta_reg_loss(model: nn.Module) -> torch.Tensor:
    """L_Δ = Σ_l ‖ΔW_l‖² trên mọi delta đã cài."""
    total = None
    for name, prm in model.named_parameters():
        if name.endswith(".delta"):
            t = (prm ** 2).sum()
            total = t if total is None else total + t
    return total if total is not None else torch.zeros((), device=next(model.parameters()).device)


def fisher_reg_loss(model: nn.Module, bank: GateBank,
                    fisher: Dict[str, torch.Tensor]) -> torch.Tensor:
    """L_Fisher = Σ_i F_i·(σ(a)·ΔW_i)²  (W_eff-W_rgb = σ(a)·ΔW theo đúng công thức).

    fisher: dict tên_delta → tensor F cùng shape ΔW (ước lượng trước train).
    """
    total = None
    named = dict(model.named_parameters())
    for name, prm in named.items():
        if not name.endswith(".delta"):
            continue
        F_i = fisher.get(name)
        if F_i is None:
            continue
        # tìm group id của delta này qua parametrization module
        # tên dạng: <mod>.parametrizations.<attr>.0.delta   (attr = weight | bias)
        try:
            owner, _, tail = name.rpartition(".parametrizations.")
            attr = tail.split(".")[0]
            sub  = model.get_submodule(owner)
            gid  = getattr(sub.parametrizations, attr)[0].group_id
        except Exception:
            gid = 0
        g = torch.sigmoid(bank.logits[gid])
        eff_delta = g * prm
        t = (F_i * eff_delta ** 2).sum()
        total = t if total is None else total + t
    return total if total is not None else torch.zeros((), device=bank.logits.device)


@torch.enable_grad()
def estimate_fisher(model, rgb_loader, device, n_batches: int = 20) -> Dict[str, torch.Tensor]:
    """Ước lượng Fisher chéo cho từng ΔW = E[(∂L_sup/∂ΔW)²] trên vài batch RGB source.

    Vì W_eff = base + σ(a)·ΔW, gradient theo ΔW tỉ lệ gradient theo W_eff (nhân σ(a)),
    nên bình phương gradient ΔW là proxy Fisher hợp lệ cho tầm quan trọng với RGB.
    """
    was_training = model.training
    model.train()
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()
              if n.endswith(".delta")}
    gen = infinite(rgb_loader)
    cnt = 0
    for _ in range(n_batches):
        images, targets = next(gen)
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items() if k != "stem"} for t in targets]
        model.zero_grad(set_to_none=True)
        loss = sum(model(images, targets).values())
        loss.backward()
        for n, p in model.named_parameters():
            if n.endswith(".delta") and p.grad is not None:
                fisher[n] += p.grad.detach() ** 2
        cnt += 1
    for n in fisher:
        fisher[n] /= max(1, cnt)
    model.zero_grad(set_to_none=True)
    if not was_training:
        model.eval()
    return fisher


# ─────────────────────────────────────────────────────────────────────────────
# Bridge domain scheduler — tỉ lệ (rgb : ir_gan : ir_real) đổi theo phase.
# ir_gan đóng vai trò cầu nối, xuất hiện ở giai đoạn chuyển tiếp rồi mờ dần.
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_RATIO_SCHEDULE = "7:3:0,5:4:1,4:3:3,3:2:5,2:1:7,2:0:8"


def parse_ratio_schedule(s: str) -> List[Tuple[int, int, int]]:
    phases = []
    for chunk in s.split(","):
        a, b, c = (int(x) for x in chunk.strip().split(":"))
        phases.append((a, b, c))
    return phases


class BlockDomainScheduler:
    """
    Lập lịch domain theo CHU KỲ NGẮN trong từng phase (giống cấu trúc D3T).

    Chia total_iters thành len(phases) giai đoạn đều nhau. Trong MỖI phase, các
    iter được chia thành nhiều CHU KỲ độ dài `cycle_iters` (mặc định 200). MỖI
    chu kỳ lại được chia theo tỉ lệ (rgb : ir_gan : ir_real) của phase thành BA
    KHỐI LIỀN, train tuần tự HẾT rgb → HẾT ir_gan → HẾT ir_real TRONG chu kỳ đó,
    rồi lặp sang chu kỳ kế tiếp. Nhờ vậy khối ir_real không bao giờ dài quá một
    chu kỳ → luôn quay lại neo rgb sau mỗi `cycle_iters` iter (chống trôi/collapse).

    Chu kỳ cuối phase có thể ngắn hơn `cycle_iters` (phần dư) — vẫn chia theo tỉ lệ.
    Stateless: get_domain(it) là hàm thuần của it (không giữ trạng thái nội bộ).
    """

    DOMAINS = ("rgb", "ir_gan", "ir_real")

    def __init__(self, phases: List[Tuple[int, int, int]], total_iters: int,
                 cycle_iters: int = 200) -> None:
        self.phases      = phases
        self.total_iters = total_iters
        self.n_phases    = len(phases)
        self.cycle_iters = max(1, cycle_iters)
        # ranh giới iter của từng phase (chia đều): phase p = [bounds[p], bounds[p+1])
        self.phase_bounds = [p * total_iters // self.n_phases
                             for p in range(self.n_phases + 1)]
        # số iter mỗi khối (rgb, ir_gan, ir_real) cho MỘT chu kỳ đầy đủ của từng phase
        self.cycle_blocks = []
        for p in range(self.n_phases):
            L    = self.phase_bounds[p + 1] - self.phase_bounds[p]
            full = min(self.cycle_iters, L) if L > 0 else 0
            self.cycle_blocks.append(self._blocks_for_phase(full, phases[p]))

    @staticmethod
    def _blocks_for_phase(length: int, ratio: Tuple[int, int, int]) -> List[int]:
        """Chia `length` iter theo `ratio` (3 domain) thành 3 khối liền.

        Dùng largest-remainder: chia sàn rồi rải phần dư cho các domain có
        ratio>0 theo phần thập phân lớn nhất. Domain ratio=0 luôn nhận 0 iter.
        """
        total = sum(ratio)
        if length <= 0 or total == 0:
            return [max(0, length), 0, 0]
        raw    = [length * r / total for r in ratio]
        counts = [int(x) for x in raw]                    # chia sàn
        rem    = length - sum(counts)                     # phần dư
        order  = sorted((i for i in range(3) if ratio[i] > 0),
                        key=lambda i: raw[i] - counts[i], reverse=True)
        for j in range(rem):
            counts[order[j % len(order)]] += 1
        return counts

    def phase_for_iter(self, it: int) -> int:
        for p in range(self.n_phases):
            if it < self.phase_bounds[p + 1]:
                return p
        return self.n_phases - 1

    def ratio_for_iter(self, it: int) -> Tuple[int, int, int]:
        return self.phases[self.phase_for_iter(it)]

    def _cycle_len_at(self, p: int, offset: int) -> Tuple[int, int]:
        """Trả (offset_trong_chu_kỳ, độ_dài_chu_kỳ) cho vị trí offset trong phase p."""
        L           = self.phase_bounds[p + 1] - self.phase_bounds[p]
        cycle_start = (offset // self.cycle_iters) * self.cycle_iters
        cycle_len   = min(self.cycle_iters, L - cycle_start)   # chu kỳ cuối có thể ngắn
        return offset - cycle_start, cycle_len

    def blocks_for_iter(self, it: int) -> List[int]:
        p              = self.phase_for_iter(it)
        offset         = it - self.phase_bounds[p]
        _, cycle_len   = self._cycle_len_at(p, offset)
        return self._blocks_for_phase(cycle_len, self.phases[p])

    def get_domain(self, it: int) -> str:
        p              = self.phase_for_iter(it)
        offset         = it - self.phase_bounds[p]              # vị trí trong phase
        off_in_cyc, cl = self._cycle_len_at(p, offset)          # vị trí trong chu kỳ
        n_rgb, n_gan, _ = self._blocks_for_phase(cl, self.phases[p])
        if off_in_cyc < n_rgb:
            return "rgb"
        if off_in_cyc < n_rgb + n_gan:
            return "ir_gan"
        return "ir_real"


# ─────────────────────────────────────────────────────────────────────────────
# GAN: RGB [0,1] → fake-IR [0,1] 3ch (CycleGAN G_A, tanh output [-1,1])
# ─────────────────────────────────────────────────────────────────────────────

def rgb_to_gan_ir(gan: torch.nn.Module, images: List[torch.Tensor]) -> List[torch.Tensor]:
    out = []
    for img in images:
        gan_in  = (img * 2.0 - 1.0).unsqueeze(0)   # [0,1] → [-1,1]
        gan_out = gan(gan_in).squeeze(0)            # (1,H,W) [-1,1]
        gan_out = (gan_out + 1.0) / 2.0            # → [0,1]
        gan_out = gan_out.repeat(3, 1, 1)          # → (3,H,W)
        out.append(gan_out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Model builder — min_size=max_size=img_size để khớp lúc pretrain teacher.
# ─────────────────────────────────────────────────────────────────────────────

def build_fcos(img_size: int) -> torch.nn.Module:
    return fcos_resnet50_fpn(num_classes=NUM_CLASSES,
                             min_size=img_size, max_size=img_size)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation (ảnh [0,1], FCOS tự normalize)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_map(model, dataloader, device) -> Dict:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    model.eval()
    # extended_summary=True → có tensor `precision` để trích per-class @0.5.
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True,
                                  extended_summary=True)
    with torch.no_grad():
        for images, targets in dataloader:
            imgs        = [img.to(device) for img in images]
            targets_dev = [{k: v.to(device) for k, v in t.items() if k != "stem"}
                           for t in targets]
            preds = model(imgs)
            metric.update(preds, targets_dev)
    result = metric.compute()
    metrics = {
        "map":    result["map"].item(),
        "map_50": result["map_50"].item(),
        "map_75": result["map_75"].item(),
    }
    classes = result.get("classes")
    if result.get("map_per_class") is not None and classes is not None:
        per_class = result["map_per_class"].tolist()
        classes_l = classes.tolist()
        if not isinstance(per_class, list):
            per_class = [per_class]
        if not isinstance(classes_l, list):
            classes_l = [classes_l]
        # (1) per-class @[0.50:0.95]
        metrics["map_per_class"] = {
            FLIR_CLASSES[c]: per_class[i]
            for i, c in enumerate(classes_l) if 0 <= c < NUM_CLASSES
        }
        # (2) per-class @0.50 — trích từ precision[T,R,K,A,M]
        #     T=0 → IoU 0.50 ; A=0 → area 'all' ; M=-1 → maxDet lớn nhất.
        prec = result.get("precision")
        if prec is not None:
            try:
                p50 = prec[0, :, :, 0, -1]           # [R, K]  (recall × class)
                pc50 = {}
                for i, c in enumerate(classes_l):
                    if not (0 <= c < NUM_CLASSES) or i >= p50.shape[1]:
                        continue
                    col   = p50[:, i]
                    valid = col > -1                  # -1 = không có dữ liệu
                    pc50[FLIR_CLASSES[c]] = (col[valid].mean().item()
                                             if bool(valid.any()) else float("nan"))
                metrics["map_per_class_50"] = pc50
            except Exception:
                pass
    return metrics


def print_eval_results(step: int, m: Dict, model_name: str = "student") -> None:
    print(f"\n{'='*60}")
    print(f"Eval @ iter {step} — real IR val set  ({model_name})")
    print(f"  mAP (0.50:0.95) : {m['map']:.4f}")
    print(f"  mAP @ 0.50      : {m['map_50']:.4f}")
    print(f"  mAP @ 0.75      : {m['map_75']:.4f}")
    pc   = m.get("map_per_class", {})
    pc50 = m.get("map_per_class_50", {})
    if pc:
        print(f"  {'per-class':12s}      : {'@[.50:.95]':>11s} | {'@0.50':>7s}")
        for cls, v in pc.items():
            v50 = pc50.get(cls, float("nan"))
            print(f"  {cls:12s}      : {v:>11.4f} | {v50:>7.4f}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Visualization — vẽ GT (xanh) + prediction (đỏ) trên bộ ảnh IR val cố định.
# ─────────────────────────────────────────────────────────────────────────────

def prepare_viz_samples(val_dataset, n: int = 8) -> List[tuple]:
    n = min(n, len(val_dataset))
    return [val_dataset[i] for i in range(n)]


def save_predictions(model, viz_samples, it, output_dir, device,
                     conf_threshold: float = 0.4, upscale: int = 2) -> None:
    from PIL import ImageDraw, Image as PILImage
    was_training = model.training
    model.eval()
    for i, (img, target) in enumerate(viz_samples):
        sample_dir = os.path.join(output_dir, "viz", f"sample_{i:02d}")
        os.makedirs(sample_dir, exist_ok=True)

        with torch.no_grad():
            pred = model([img.to(device)])[0]      # ảnh [0,1]; FCOS tự normalize

        img_np  = (img.cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
        H, W    = img_np.shape[0], img_np.shape[1]
        img_pil = PILImage.fromarray(img_np).resize((W * upscale, H * upscale), PILImage.BILINEAR)
        draw    = ImageDraw.Draw(img_pil)
        s       = upscale

        # GT — xanh lá
        for box, lbl in zip(target["boxes"].numpy(), target["labels"].numpy()):
            x1, y1, x2, y2 = [c * s for c in box]
            cls = FLIR_CLASSES[int(lbl)]
            draw.rectangle([x1, y1, x2, y2], outline=(0, 220, 0), width=2)
            draw.rectangle([x1, y1 - 14, x1 + len(cls) * 7 + 4, y1], fill=(0, 220, 0))
            draw.text((x1 + 2, y1 - 14), cls, fill=(0, 0, 0))

        # Prediction — đỏ
        for box, score, lbl in zip(pred["boxes"].cpu().numpy(),
                                   pred["scores"].cpu().numpy(),
                                   pred["labels"].cpu().numpy()):
            if score < conf_threshold:
                continue
            x1, y1, x2, y2 = [c * s for c in box]
            cls  = FLIR_CLASSES[int(lbl)] if int(lbl) < NUM_CLASSES else "?"
            text = f"{cls} {score:.2f}"
            draw.rectangle([x1, y1, x2, y2], outline=(255, 60, 60), width=2)
            draw.rectangle([x1, y2, x1 + len(text) * 7 + 4, y2 + 14], fill=(255, 60, 60))
            draw.text((x1 + 2, y2), text, fill=(255, 255, 255))

        draw.rectangle([0, 0, 130, 16], fill=(20, 20, 20))
        draw.text((4, 2), f"iter {it}", fill=(255, 220, 0))
        img_pil.save(os.path.join(sample_dir, f"iter_{it:06d}.jpg"), quality=92)
    if was_training:
        model.train()


# ─────────────────────────────────────────────────────────────────────────────
# λ schedule (D3T-style): ramp tuyến tính 0 → lambda_max trong ramp_iters.
# ─────────────────────────────────────────────────────────────────────────────

def lambda_for_iter(it: int, lambda_max: float, ramp_iters: int) -> float:
    if ramp_iters <= 0:
        return lambda_max
    return lambda_max * min(1.0, it / ramp_iters)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def infinite(loader):
    """Sinh batch vô hạn (cycle qua loader) — phù hợp train theo iteration."""
    while True:
        for batch in loader:
            yield batch


def train(t_model_rgb, t_model_ir, s_model, gan,
          rgb_loader, ir_loader, val_loader,
          optimizer, scheduler, domain_scheduler,
          total_iters, eval_every, log_every, device, output_dir,
          pseudo_conf, lambda_max, lambda_ramp_iters,
          w_match, w_cross, ema_alpha, grad_clip,
          distill_cfg, gate_cfg, gate_bank, fisher=None,
          viz_samples=None, viz_conf=0.4):

    t_model_rgb.eval()
    t_model_ir.eval()
    s_model.train()

    lam_gate   = gate_cfg["lambda_gate"]
    lam_delta  = gate_cfg["lambda_delta"]
    lam_fisher = gate_cfg["lambda_fisher"]

    strong_rgb_aug = get_rgb_strong_transform()
    strong_ir_aug  = get_ir_strong_transform()

    rgb_gen = infinite(rgb_loader)
    ir_gen  = infinite(ir_loader)

    best_map     = 0.0
    running_loss = 0.0

    for it in range(1, total_iters + 1):
        domain = domain_scheduler.get_domain(it - 1)
        lam    = lambda_for_iter(it, lambda_max, lambda_ramp_iters)

        # ── rgb / ir_gan: có GT → supervised + unsupervised ───────────────────
        if domain in ("rgb", "ir_gan"):
            images, targets = next(rgb_gen)
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items() if k != "stem"}
                       for t in targets]

            if domain == "ir_gan":
                with torch.no_grad():
                    images = rgb_to_gan_ir(gan, images)
                strong_aug       = strong_ir_aug
                t_match, t_cross = t_model_ir, t_model_rgb   # ảnh IR-like
                ema_teacher      = t_model_ir
            else:  # rgb
                strong_aug       = strong_rgb_aug
                t_match, t_cross = t_model_rgb, t_model_ir   # ảnh RGB
                ema_teacher      = t_model_rgb

            # Supervised loss (ảnh weak, GT thật)
            loss_sup = sum(s_model(images, targets).values())

            # Dense distillation: student(strong) mô phỏng teacher(weak) — KHÔNG NMS
            s_images = [strong_aug(img) for img in images]
            s_out = dense_forward(s_model, s_images)
            with torch.no_grad():
                t_match_out = dense_forward(t_match, images)
                t_cross_out = dense_forward(t_cross, images)
            loss_uns_match = distill(s_out, t_match_out, distill_cfg)
            loss_uns_cross = distill(s_out, t_cross_out, distill_cfg)

            loss_uns   = w_match * loss_uns_match + w_cross * loss_uns_cross
            total_loss = loss_sup + lam * loss_uns

        # ── ir_real: không GT → chỉ unsupervised (distillation) ───────────────
        else:  # ir_real
            batch  = next(ir_gen)
            images = [img.to(device) for img in batch]

            t_match, t_cross = t_model_ir, t_model_rgb
            ema_teacher      = t_model_ir

            s_images = [strong_ir_aug(img) for img in images]
            s_out = dense_forward(s_model, s_images)
            with torch.no_grad():
                t_match_out = dense_forward(t_match, images)
                t_cross_out = dense_forward(t_cross, images)
            loss_uns_match = distill(s_out, t_match_out, distill_cfg)
            loss_uns_cross = distill(s_out, t_cross_out, distill_cfg)

            loss_uns   = w_match * loss_uns_match + w_cross * loss_uns_cross
            total_loss = lam * loss_uns

        # ── Regularizer PHƯƠNG PHÁP 1: L_gate + L_Δ (+ L_Fisher nếu bật) ──────
        l_gate  = gate_reg_loss(gate_bank)
        l_delta = delta_reg_loss(s_model)
        total_loss = total_loss + lam_gate * l_gate + lam_delta * l_delta
        if lam_fisher > 0 and fisher is not None:
            total_loss = total_loss + lam_fisher * fisher_reg_loss(s_model, gate_bank, fisher)

        optimizer.zero_grad()
        total_loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in s_model.parameters() if p.requires_grad], grad_clip)
        optimizer.step()
        scheduler.step()                              # cosine theo iteration
        update_ema_effective(ema_teacher, s_model, alpha=ema_alpha)  # EMA theo W_eff

        running_loss += total_loss.item()

        if it % log_every == 0:
            r      = domain_scheduler.ratio_for_iter(it - 1)
            cur_lr = optimizer.param_groups[0]["lr"]
            gates  = gate_bank.gates().detach().cpu().tolist()
            gstr   = " ".join(f"{n}={g:.2f}" for n, g in zip(GROUP_NAMES, gates))
            print(f"Iter [{it}/{total_iters}] ratio={r[0]}:{r[1]}:{r[2]} "
                  f"domain={domain:8s} λ={lam:.2f} lr={cur_lr:.2e} "
                  f"loss(avg{log_every})={running_loss/log_every:.4f}")
            print(f"    gates: {gstr}")
            running_loss = 0.0

        # ── Eval + checkpoint định kỳ theo iteration ──────────────────────────
        if it % eval_every == 0 or it == total_iters:
            metrics = evaluate_map(s_model, val_loader, device)
            print_eval_results(it, metrics)
            if viz_samples:
                save_predictions(s_model, viz_samples, it, output_dir, device, viz_conf)
            s_model.train()
            if metrics["map"] > best_map:
                best_map = metrics["map"]
                ckpt = os.path.join(output_dir, f"best_iter{it:06d}_map{best_map:.4f}.pth")
                torch.save(s_model.state_dict(), ckpt)
                print(f"New best mAP {best_map:.4f} → {ckpt}")

    last = os.path.join(output_dir, f"last_iter{total_iters:06d}.pth")
    torch.save(s_model.state_dict(), last)
    print(f"Last model saved → {last}")
    print(f"\nTraining complete. Best mAP: {best_map:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="D3T-style RGB→IR domain adaptation (v2: disjoint data như paper)")

    p.add_argument("--data-root",           required=True)
    p.add_argument("--gan-weights",         required=True)
    p.add_argument("--teacher-rgb-weights", required=True,
                   help="Checkpoint teacher RGB. CŨNG dùng để init student.")
    p.add_argument("--teacher-ir-weights",  required=True,
                   help="Checkpoint teacher IR (pretrain trên GAN-IR).")
    p.add_argument("--cyclegan-repo", default="/kaggle/working/pytorch-CycleGAN-and-pix2pix")
    p.add_argument("--output-dir",    default="/kaggle/working/d3t_v2")

    p.add_argument("--total-iters",  type=int,   default=20000,
                   help="Tổng số iteration huấn luyện (train theo iter giống D3T).")
    p.add_argument("--eval-every",   type=int,   default=1000,
                   help="Eval trên IR val + lưu best mỗi N iteration.")
    p.add_argument("--log-every",    type=int,   default=100)
    p.add_argument("--batch-size",   type=int,   default=8)
    p.add_argument("--img-size",     type=int,   default=640,
                   help="Cạnh ảnh (vuông). Khớp với lúc pretrain teacher (default 640).")
    p.add_argument("--num-workers",  type=int,   default=2)

    # Optimizer: SGD + cosine/constant + clip
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--lr-schedule",  type=str,   default="cosine",
                   choices=["cosine", "constant"],
                   help="cosine (mặc định) = CosineAnnealingLR giảm LR về 0 theo total-iters. "
                        "constant = giữ LR = --lr cố định suốt quá trình train (giống D3T).")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip",    type=float, default=10.0)

    # === v2: disjoint data split (đúng paper) ===
    p.add_argument("--disjoint-split", dest="disjoint_split", action="store_true", default=True,
                   help="(mặc định) Tách RGB source và real-IR target thành 2 nửa "
                        "KHÔNG giao nhau của align_train (chống overfit như paper).")
    p.add_argument("--no-disjoint-split", dest="disjoint_split", action="store_false",
                   help="Tắt disjoint: rgb & ir_real dùng chung align_train (cặp khớp — KHÔNG giống paper).")
    p.add_argument("--source-frac", type=float, default=0.5,
                   help="Tỉ lệ ảnh align_train dành cho RGB source; phần còn lại là real-IR target. "
                        "Paper: 0.5 (2064 source / 2064 target).")
    p.add_argument("--split-seed",  type=int,   default=-1,
                   help="Cách tách disjoint. Mặc định -1 = lấy 'first N' theo thứ tự gốc ĐÚNG paper "
                        "('first 2,064 RGB / a separate set of 2,064 thermal'). >=0 = xáo trộn xác định.")

    # Domain ratio schedule
    p.add_argument("--ratio-schedule", type=str, default=DEFAULT_RATIO_SCHEDULE,
                   help="Tỉ lệ (rgb:ir_gan:ir_real) theo phase, phân tách bằng dấu phẩy. "
                        f"Default: {DEFAULT_RATIO_SCHEDULE}")
    p.add_argument("--cycle-iters", type=int, default=200,
                   help="Độ dài chu kỳ (iter) trong mỗi phase (giống RGB_IR_CHANGE_TOTAL "
                        "của D3T). Mỗi chu kỳ chia theo tỉ lệ phase thành rgb→ir_gan→ir_real. "
                        "Nhỏ = neo rgb dày hơn (ir_real block ngắn hơn). Default: 200.")

    # Pseudo-label / λ
    p.add_argument("--pseudo-conf",        type=float, default=0.5)
    p.add_argument("--lambda-max",        type=float, default=1.0)
    p.add_argument("--lambda-ramp-iters", type=int,   default=6000,
                   help="Số iteration ramp λ 0→lambda_max (0 = λ cố định ngay từ đầu).")

    # Trọng số 2 teacher trong unsupervised loss
    p.add_argument("--w-match", type=float, default=1.0,
                   help="Trọng số teacher CÙNG domain với ảnh (cao).")
    p.add_argument("--w-cross", type=float, default=0.3,
                   help="Trọng số teacher CHÉO domain (thấp).")

    p.add_argument("--ema-alpha", type=float, default=0.9996)

    # === Dense distillation (giống D3T) ===
    p.add_argument("--distill-ratio", type=float, default=0.01,
                   help="Tỉ lệ vùng top theo harmony measure (D3T RATIO=0.01).")
    p.add_argument("--hm-alpha", type=float, default=0.5,
                   help="Số mũ cls_prob trong hm = cls_prob^α·iou^β (D3T HM.ALPHA=0.5).")
    p.add_argument("--hm-beta",  type=float, default=0.5,
                   help="Số mũ iou trong harmony measure (D3T HM.BETA=0.5).")
    p.add_argument("--un-alpha", type=float, default=1.0,
                   help="UN_REGULAR_ALPHA cho uncertainty weighting (D3T=1.0).")
    p.add_argument("--dw-logits",  type=float, default=4.0, help="Trọng số loss logits (D3T=4).")
    p.add_argument("--dw-deltas",  type=float, default=1.0, help="Trọng số loss deltas (D3T=1).")
    p.add_argument("--dw-quality", type=float, default=1.0, help="Trọng số loss quality (D3T=1).")
    p.add_argument("--dw-uhl",     type=float, default=1.0, help="Trọng số loss UHL/VFL (D3T=1).")

    # === PHƯƠNG PHÁP 1: Direct Differentiable Gate ===
    p.add_argument("--gate-init",  type=float, default=-2.0,
                   help="Giá trị khởi tạo gate logit a_l (a=-2 → g≈0.12, model bắt đầu gần source).")
    p.add_argument("--delta-std",  type=float, default=1e-4,
                   help="Std khởi tạo ΔW ~ N(0, std). 0 = khởi tạo ΔW=0.")
    p.add_argument("--gate-lr",    type=float, default=1e-2,
                   help="LR riêng cho gate logits (thường lớn hơn LR trọng số để gate học kịp).")
    p.add_argument("--gate-bias", choices=["off", "conv", "all"], default="off",
                   help="Ap delta cho BIAS nua hay khong. off (mac dinh) = chi conv "
                        "weight, giu nguyen hanh vi cu. conv = gate them bias cua "
                        "Conv2d. all = gate them ca affine cua GroupNorm trong head. "
                        "MOI CHE DO deu KHONG dung toi FrozenBatchNorm2d (he so BN "
                        "la buffer thong ke ImageNet, giu nguyen).")
    p.add_argument("--train-others", action="store_true", default=False,
                   help="Cho phép train cả bias/BN (mặc định đóng băng — mọi adaptation qua gate·ΔW).")
    p.add_argument("--lambda-gate",  type=float, default=0.01,
                   help="λ_g cho L_gate=mean σ(a) (phạt bật gate → thưa hoá adaptation).")
    p.add_argument("--lambda-delta", type=float, default=1e-4,
                   help="λ_Δ cho L_Δ=Σ‖ΔW‖² (chống nhập nhằng scale g·ΔW).")
    p.add_argument("--lambda-fisher", type=float, default=0.0,
                   help="λ_F cho L_Fisher=Σ F·(σ(a)ΔW)² (0 = tắt; >0 cần --fisher-batches).")
    p.add_argument("--fisher-batches", type=int, default=20,
                   help="Số batch RGB source dùng ước lượng Fisher (khi λ_fisher>0).")

    # Visualization
    p.add_argument("--viz-samples", type=int,   default=8,
                   help="Số ảnh IR val cố định để vẽ dự đoán mỗi lần eval (0 = tắt).")
    p.add_argument("--viz-conf",    type=float, default=0.4,
                   help="Ngưỡng confidence khi vẽ box prediction.")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # CycleGAN repo
    if not os.path.isdir(args.cyclegan_repo):
        print(f"Cloning CycleGAN repo → {args.cyclegan_repo}")
        os.system("git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git "
                  f"{args.cyclegan_repo}")
    sys.path.insert(0, args.cyclegan_repo)
    from models import networks  # noqa: E402

    for label, path in [("--data-root", args.data_root),
                        ("--gan-weights", args.gan_weights),
                        ("--teacher-rgb-weights", args.teacher_rgb_weights),
                        ("--teacher-ir-weights", args.teacher_ir_weights)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label}: '{path}' không tồn tại")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Device      : {device}")
    print(f"Image size  : {args.img_size}x{args.img_size}")
    print(f"Output dir  : {args.output_dir}")

    # ── Datasets & loaders ──────────────────────────────────────────────────
    # v2: tách RGB source và real-IR target thành 2 nửa disjoint (như paper).
    if args.disjoint_split:
        all_stems = _read_split_file(Path(args.data_root) / "align_train.txt")
        src_stems, tgt_stems = _split_disjoint(all_stems, args.source_frac, args.split_seed)
        rgb_ds = FLIRRGBDataset(args.data_root, stems=src_stems, img_size=args.img_size)
        ir_ds  = FLIRIRDataset(args.data_root,  stems=tgt_stems, img_size=args.img_size)
        mode = ("first-N" if args.split_seed < 0 else f"shuffled(seed={args.split_seed})")
        print(f"DISJOINT split [{mode}]: RGB source={len(src_stems)} | real-IR target={len(tgt_stems)} "
              f"(không trùng ảnh — đúng paper mục 4.1)")
    else:
        rgb_ds = FLIRRGBDataset(args.data_root, split="train", img_size=args.img_size)
        ir_ds  = FLIRIRDataset(args.data_root,  split="train", img_size=args.img_size)
        print("WARNING: KHÔNG disjoint — rgb & real-IR dùng chung align_train (cặp khớp, KHÔNG giống paper).")

    val_ds = FLIRIRValDataset(args.data_root, split="validation", img_size=args.img_size)
    print(f"RGB train   : {len(rgb_ds)} | IR train : {len(ir_ds)} | IR val (test) : {len(val_ds)}")

    rgb_loader = torch.utils.data.DataLoader(
        rgb_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=rgb_collate, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    ir_loader = torch.utils.data.DataLoader(
        ir_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=ir_collate, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=ir_val_collate, num_workers=args.num_workers, pin_memory=True)

    # ── GAN (frozen) ────────────────────────────────────────────────────────
    gan = networks.define_G(input_nc=3, output_nc=1, ngf=64,
                            netG="resnet_9blocks", norm="instance")
    gan.load_state_dict(torch.load(args.gan_weights, map_location="cpu"))
    gan.eval().to(device)
    for prm in gan.parameters():
        prm.requires_grad = False

    # ── Teachers (frozen, pretrained) ────────────────────────────────────────
    t_model_rgb = build_fcos(args.img_size)
    t_model_ir  = build_fcos(args.img_size)
    t_model_rgb.load_state_dict(torch.load(args.teacher_rgb_weights, map_location="cpu"))
    t_model_ir.load_state_dict(torch.load(args.teacher_ir_weights,   map_location="cpu"))
    t_model_rgb.to(device)
    t_model_ir.to(device)
    for prm in list(t_model_rgb.parameters()) + list(t_model_ir.parameters()):
        prm.requires_grad = False
    print("Teachers loaded (frozen).")

    # ── Student = ĐÚNG weight teacher RGB ─────────────────────────────────────
    s_model = build_fcos(args.img_size)
    s_model.load_state_dict(torch.load(args.teacher_rgb_weights, map_location="cpu"))
    s_model.to(device)
    print("Student initialized from teacher RGB weights.")

    # ── PHƯƠNG PHÁP 1: cài Direct Gate (W_eff = W_rgb + σ(a)·ΔW) ───────────────
    gate_bank, n_gated = install_direct_gates(
        s_model, gate_init=args.gate_init, delta_std=args.delta_std,
        train_others=args.train_others, gate_bias=args.gate_bias)
    gate_bank.to(device)
    n_delta = sum(p.numel() for n, p in s_model.named_parameters()
                  if n.endswith(".delta") and p.requires_grad)
    brk = count_gated(s_model)
    print(f"Direct gates : {n_gated} tensor được gate | {N_GROUPS} nhóm gate "
          f"({', '.join(GROUP_NAMES)})")
    print(f"  gate-bias={args.gate_bias} → conv.weight={brk['conv.weight']} "
          f"conv.bias={brk['conv.bias']} norm.affine={brk['norm.affine']} "
          f"| FrozenBN: KHÔNG gate (giữ nguyên)")
    print(f"  init a={args.gate_init} → g≈{torch.sigmoid(torch.tensor(args.gate_init)).item():.3f} "
          f"| ΔW std={args.delta_std} | ΔW params={n_delta/1e6:.2f}M | "
          f"train_others={args.train_others}")
    print(f"  λ_gate={args.lambda_gate} λ_delta={args.lambda_delta} λ_fisher={args.lambda_fisher}")

    # ── Optimizer / scheduler ────────────────────────────────────────────────
    #   Tách 2 nhóm: gate logits (LR riêng, không weight-decay) và phần còn lại (ΔW…).
    gate_params  = [p for n, p in s_model.named_parameters()
                    if p.requires_grad and n.startswith("gate_bank.")]
    other_params = [p for n, p in s_model.named_parameters()
                    if p.requires_grad and not n.startswith("gate_bank.")]
    optimizer = torch.optim.SGD(
        [{"params": other_params, "lr": args.lr, "weight_decay": args.weight_decay},
         {"params": gate_params,  "lr": args.gate_lr, "weight_decay": 0.0}],
        lr=args.lr, momentum=0.9)
    if args.lr_schedule == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda it: 1.0)
        print(f"LR schedule : CONSTANT | lr={args.lr:.2e} (giữ cố định suốt train)")
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_iters)
        print(f"LR schedule : COSINE | lr={args.lr:.2e} → 0 theo {args.total_iters} iter")

    phases           = parse_ratio_schedule(args.ratio_schedule)
    domain_scheduler = BlockDomainScheduler(phases, args.total_iters,
                                            cycle_iters=args.cycle_iters)
    print(f"Ratio schedule (rgb:ir_gan:ir_real) per phase: {phases}")
    print(f"Domain scheduler: BLOCK theo CHU KỲ {args.cycle_iters} iter "
          f"(mỗi chu kỳ: rgb → ir_gan → ir_real theo tỉ lệ phase)")
    for p in range(domain_scheduler.n_phases):
        s, e = domain_scheduler.phase_bounds[p], domain_scheduler.phase_bounds[p + 1]
        n_rgb, n_gan, n_real = domain_scheduler.cycle_blocks[p]
        n_cyc = -(-(e - s) // domain_scheduler.cycle_iters)   # số chu kỳ (làm tròn lên)
        print(f"  Phase {p} [{s:>5d}-{e:<5d}] ratio={phases[p]} → "
              f"{n_cyc} chu kỳ × (rgb:{n_rgb} | ir_gan:{n_gan} | ir_real:{n_real})")
    print(f"Total iters : {args.total_iters} | eval every {args.eval_every}")

    viz_samples = prepare_viz_samples(val_ds, args.viz_samples) if args.viz_samples > 0 else None
    if viz_samples:
        print(f"Viz samples : {len(viz_samples)} ảnh IR val cố định")

    distill_cfg = dict(
        ratio=args.distill_ratio, hm_alpha=args.hm_alpha, hm_beta=args.hm_beta,
        un_alpha=args.un_alpha, w_logits=args.dw_logits, w_deltas=args.dw_deltas,
        w_quality=args.dw_quality, w_uhl=args.dw_uhl)
    print(f"Distillation: DENSE giống D3T | ratio={args.distill_ratio} "
          f"hm=({args.hm_alpha},{args.hm_beta}) "
          f"weights(logits/deltas/quality/uhl)="
          f"{args.dw_logits}/{args.dw_deltas}/{args.dw_quality}/{args.dw_uhl}")

    gate_cfg = dict(lambda_gate=args.lambda_gate, lambda_delta=args.lambda_delta,
                    lambda_fisher=args.lambda_fisher)

    # ── Fisher (tùy chọn) — ước lượng trên vài batch RGB source trước khi train ──
    fisher = None
    if args.lambda_fisher > 0:
        print(f"Estimating Fisher trên {args.fisher_batches} batch RGB source...")
        fisher = estimate_fisher(s_model, rgb_loader, device, args.fisher_batches)
        s_model.train()
        print("Fisher estimated.")

    train(t_model_rgb, t_model_ir, s_model, gan,
          rgb_loader, ir_loader, val_loader,
          optimizer, scheduler, domain_scheduler,
          total_iters=args.total_iters, eval_every=args.eval_every,
          log_every=args.log_every, device=device, output_dir=args.output_dir,
          pseudo_conf=args.pseudo_conf, lambda_max=args.lambda_max,
          lambda_ramp_iters=args.lambda_ramp_iters,
          w_match=args.w_match, w_cross=args.w_cross,
          ema_alpha=args.ema_alpha, grad_clip=args.grad_clip,
          distill_cfg=distill_cfg, gate_cfg=gate_cfg, gate_bank=gate_bank,
          fisher=fisher, viz_samples=viz_samples, viz_conf=args.viz_conf)


if __name__ == "__main__":
    main()
