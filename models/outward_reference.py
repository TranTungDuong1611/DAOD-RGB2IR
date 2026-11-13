"""BiRefNet OutRef heads, adapted to spatially gate the FCOS residual branch.

Reference: ZhengPeng7/BiRefNet, models/birefnet.py, gdt_convs/pred/attn.
The shared gradient feature and two separate heads follow that implementation.
Our neutral 2*sigmoid gain differs from its suppressive sigmoid gate.
"""

from torch import nn


class OutwardReference(nn.Module):
    def __init__(self, channels, hidden_channels=16, batch_norm=True):
        super().__init__()
        self.gdt_convs = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        self.gdt_convs_pred = nn.Conv2d(hidden_channels, 1, 1)
        self.gdt_convs_attn = nn.Conv2d(hidden_channels, 1, 1)
        nn.init.zeros_(self.gdt_convs_attn.weight)
        nn.init.zeros_(self.gdt_convs_attn.bias)

    def forward(self, feature):
        gradient_feature = self.gdt_convs(feature)
        edge_logits = self.gdt_convs_pred(gradient_feature) if self.training else None
        attention = 2 * self.gdt_convs_attn(gradient_feature).sigmoid()
        return attention, edge_logits
