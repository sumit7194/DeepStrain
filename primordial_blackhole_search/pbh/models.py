"""Two detector architectures, built for a controlled comparison.

SpectrogramCNN  — the field-standard baseline: treat the (128, 256)
                  spectrogram as an image, convolve globally.
ChunkTransformer — the long-signal hypothesis: encode 16 s chunks with a
                  shared CNN, then let a transformer integrate evidence
                  ACROSS chunks. A faint minutes-long track deposits a little
                  energy in many chunks; attention can sum coherent structure
                  that global pooling dilutes.

Both ~1-2 M params so the comparison is about inductive bias, not capacity.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import config as C


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.SiLU(),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.SiLU(),
    )


class SpectrogramCNN(nn.Module):
    """(B, 1, 128, 256) -> logit"""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _block(1, 32),    # 64 x 128
            _block(32, 64),   # 32 x 64
            _block(64, 128),  # 16 x 32
            _block(128, 256), # 8 x 16
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ChunkTransformer(nn.Module):
    """(B, 1, 128, 256) -> reshape to 16 chunks of 16 time-bins -> shared CNN
    encoder per chunk -> transformer across chunks -> logit."""

    N_CHUNKS = 16
    D = 128

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            _block(1, 32),   # (32, 64, 8)
            _block(32, 64),  # (64, 32, 4)
            _block(64, 128), # (128, 16, 2)
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),    # 128
        )
        self.pos = nn.Parameter(torch.randn(1, self.N_CHUNKS + 1, self.D) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, self.D) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=self.D, nhead=4, dim_feedforward=256, dropout=0.1,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=4)
        self.head = nn.Linear(self.D, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        # (B, 1, 128, 256) -> 16 chunks of (1, 128, 16)
        chunks = x.unfold(3, 16, 16)            # (B, 1, 128, 16, 16)
        chunks = chunks.permute(0, 3, 1, 2, 4)  # (B, 16, 1, 128, 16)
        z = self.encoder(chunks.reshape(b * self.N_CHUNKS, 1, C.N_FREQ_BINS, 16))
        z = z.reshape(b, self.N_CHUNKS, self.D)
        z = torch.cat([self.cls.expand(b, -1, -1), z], dim=1) + self.pos
        z = self.transformer(z)
        return self.head(z[:, 0]).squeeze(-1)


class _ResBlock1d(nn.Module):
    """Strided 1-D residual block on SIGNED features (phase is preserved —
    no early magnitude/|.|^2 — so coherence builds through receptive-field depth)."""

    def __init__(self, cin: int, cout: int, k: int = 16, stride: int = 4) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, k, stride=stride, padding=k // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(cout)
        self.act = nn.SiLU()
        self.down = nn.Conv1d(cin, cout, 1, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.down(x)
        y = self.act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        m = min(y.shape[-1], r.shape[-1])
        return self.act(y[..., :m] + r[..., :m])


class SemiCoherentNet(nn.Module):
    """Stage-1 learned semi-coherent detector on 64-s WHITENED STRAIN.

    64-s window -> 8 chunks of 8 s -> shared 1-D ResNet encoder -> per-chunk
    score rho_i -> combiner over {rho_i} + a consistency feature (spread of the
    rho_i, the learned analog of stage-0's chi^2 glitch veto) -> logit.
    """

    N_CHUNKS = 8

    def __init__(self) -> None:
        super().__init__()
        chans = [1, 32, 64, 128, 128, 128, 128]  # /4 per block: 32768 -> 8
        self.encoder = nn.Sequential(
            *[_ResBlock1d(chans[i], chans[i + 1]) for i in range(len(chans) - 1)],
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        d = chans[-1]
        self.chunk_score = nn.Linear(d, 1)
        self.combiner = nn.Sequential(
            nn.Linear(d + self.N_CHUNKS + 4, 128),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, length = x.shape
        lc = length // self.N_CHUNKS
        chunks = (
            x[..., : lc * self.N_CHUNKS]
            .reshape(b, 1, self.N_CHUNKS, lc)
            .permute(0, 2, 1, 3)
            .reshape(b * self.N_CHUNKS, 1, lc)
        )
        z = self.encoder(chunks).reshape(b, self.N_CHUNKS, -1)  # (B, 8, d)
        rho = self.chunk_score(z).squeeze(-1)                   # (B, 8)
        r2 = rho**2
        feats = torch.cat(
            [rho, r2.sum(1, keepdim=True), r2.mean(1, keepdim=True),
             r2.std(1, keepdim=True), r2.amax(1, keepdim=True), z.mean(1)],
            dim=1,
        )
        return self.combiner(feats).squeeze(-1)


def make_model(name: str) -> nn.Module:
    return {
        "cnn": SpectrogramCNN,
        "transformer": ChunkTransformer,
        "semicoherent": SemiCoherentNet,
    }[name]()
