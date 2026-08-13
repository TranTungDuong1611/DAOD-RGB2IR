import torch
from config import Phase
from config import LossConfig


def compute_combined_loss(
    student,                
    distill_adapter,        
    student_images,         
    global_step: int,       
    phase: Phase,            
    loss_cfg: LossConfig,   
    targets=None,           
    teacher_rgb_images=None,
    teacher_ir_images=None  
):
    loss_dict = {}
    total_loss = torch.tensor(0.0, device=student_images.device)

    sup_phase_w, dist_phase_w = loss_cfg.get_phase_weights(phase)

    # 1. Calculate Supervised Loss (if we have GT)
    if targets is not None and sup_phase_w > 0.0:
        sup_components = student(student_images, targets)
        
        for k, v in sup_components.items():
            weighted_v = v * sup_phase_w
            loss_dict[f"sup_{k}"] = weighted_v
            total_loss += weighted_v.squeeze()

    # 2. Calculate Distillation Loss (if using distilled labels)
    if dist_phase_w > 0.0 and teacher_rgb_images is not None and teacher_ir_images is not None:
        kd_components = distill_adapter.distill_step(
            student_images, 
            teacher_rgb_images, 
            teacher_ir_images, 
            global_step
        )
        
        for k, v in kd_components.items():
            if "logits" in k:
                comp_w = loss_cfg.weight_logits
            elif "deltas" in k:
                comp_w = loss_cfg.weight_deltas
            elif "quality" in k:
                comp_w = loss_cfg.weight_quality
            else:
                comp_w = 1.0
            
            weighted_v = v * comp_w * dist_phase_w
            loss_dict[k] = weighted_v
            total_loss += weighted_v.squeeze()

    return total_loss, loss_dict