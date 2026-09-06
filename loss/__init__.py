"""Loss package exports.

The new criterion is imported eagerly. Legacy symbols are resolved lazily so
the new FCOS path does not import the old orchestration code, while explicit
legacy callers remain usable until those files are retired.
"""

from .d3t_criterion import D3TLossCriterion

__all__ = [
    "D3TLossCriterion",
    "compute_combined_loss",
    "VarifocalLoss",
    "HMfocalLoss",
    "varifocal_loss",
    "hmfocal_loss",
    "QFLv2",
    "giou_loss_ltrb",
]


def __getattr__(name):
    if name == "compute_combined_loss":
        from .orchestrator import compute_combined_loss

        return compute_combined_loss
    if name in {"VarifocalLoss", "HMfocalLoss", "varifocal_loss", "hmfocal_loss"}:
        from .hmfocal import HMfocalLoss, VarifocalLoss, hmfocal_loss, varifocal_loss

        return {
            "VarifocalLoss": VarifocalLoss,
            "HMfocalLoss": HMfocalLoss,
            "varifocal_loss": varifocal_loss,
            "hmfocal_loss": hmfocal_loss,
        }[name]
    if name in {"QFLv2", "giou_loss_ltrb"}:
        from .distill import QFLv2, giou_loss_ltrb

        return {"QFLv2": QFLv2, "giou_loss_ltrb": giou_loss_ltrb}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
