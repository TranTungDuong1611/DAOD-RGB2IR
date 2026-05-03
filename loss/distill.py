import torch.nn.functional as F
import torch


def QFLv2(pred_sigmoid, teacher_sigmoid, weight=None, beta=2.0, reduction="mean"):
    """
    Use to calculate the loss logits. 
    With the input is the probability of Student and pseudo from teacher, and the weight. 
    Output is a scalar.
    """
    pt = pred_sigmoid
    zerolabel = pt.new_zeros(pt.shape)
    loss = F.binary_cross_entropy(
        pred_sigmoid, zerolabel, reduction='none') * pt.pow(beta)
    pos = weight > 0

    # positive goes to bbox quality
    pt = teacher_sigmoid[pos] - pred_sigmoid[pos]
    loss[pos] = F.binary_cross_entropy(
        pred_sigmoid[pos], teacher_sigmoid[pos], reduction='none') * pt.pow(beta)

    valid = weight >= 0
    if reduction == "mean":
        loss = loss[valid].mean()
    elif reduction == "sum":
        loss = loss[valid].sum()
    return loss

def giou_loss_ltrb(pred, target, weight=None, reduction="mean"):
    """
    GIoU loss cho format LTRB. Input are the prediction of students in LTRB format
    and the prediction of teachers in LTRB format. Shape [N, 4]
    pred/target shape: [N, 4]
    """
    pred_l, pred_t, pred_r, pred_b = pred.unbind(dim=-1)
    tgt_l, tgt_t, tgt_r, tgt_b = target.unbind(dim=-1)

    # Area(Width = L+R, Height = T+B)
    pred_area = (pred_l + pred_r).clamp(min=0) * (pred_t + pred_b).clamp(min=0)
    target_area = (tgt_l + tgt_r).clamp(min=0) * (tgt_t + tgt_b).clamp(min=0)

    # Intersection 
    inter_w = (torch.min(pred_l, tgt_l) + torch.min(pred_r, tgt_r)).clamp(min=0)
    inter_h = (torch.min(pred_t, tgt_t) + torch.min(pred_b, tgt_b)).clamp(min=0)
    inter_area = inter_w * inter_h

    # Union 
    union = pred_area + target_area - inter_area
    iou = inter_area / union.clamp(min=1e-7)

    # Enclosing box 
    enc_w = (torch.max(pred_l, tgt_l) + torch.max(pred_r, tgt_r)).clamp(min=0)
    enc_h = (torch.max(pred_t, tgt_t) + torch.max(pred_b, tgt_b)).clamp(min=0)
    enc_area = enc_w * enc_h
    
    giou = iou - (enc_area - union) / enc_area.clamp(min=1e-7)
    loss = 1.0 - giou

    if weight is not None:
        loss = loss * weight
        
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss