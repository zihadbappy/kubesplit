"""Small CNN split into a client bottom and a server top.

The cut sits after the second conv block, so each sample crosses the network
as a 64x8x8 float32 activation (16 KiB). In decoupled mode the client also
owns an auxiliary head, giving it a local loss that never needs gradients
from the server (Decoupled Greedy Learning).
"""

import torch
from torch import nn

NUM_CLASSES = 10
CUT_SHAPE = (64, 8, 8)  # activation shape per sample at the cut


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class ClientBottom(nn.Module):
    """3x32x32 image -> 64x8x8 activation."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(conv_block(3, 32), conv_block(32, 64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class AuxHead(nn.Module):
    """Client-side auxiliary classifier used only in decoupled mode."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        c = CUT_SHAPE[0]
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(2),
            nn.Flatten(),
            nn.Linear(c * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class ServerTop(nn.Module):
    """64x8x8 activation -> class logits."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(conv_block(64, 128), conv_block(128, 256))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, num_classes)
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(h))
