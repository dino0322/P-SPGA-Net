# p_module.py
# -*- coding: utf-8 -*-
"""
P_Module: Patch Embedding -> Positional Encoding -> Positional Encoding Block (variants)
          -> Scalar Weighting (gated residual/skip) -> (optional) GAP -> (optional) classifier.

Designed to sit after vision backbone that outputs a feature map (B, C, H, W).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)

PosBlockMode = Literal[
    "dense",
    "dense_conv1d_3",
    "dense_conv1d_5",
    "maxpool_dense",
]
def make_gn(ch: int) -> nn.GroupNorm:
    for g in (32, 16, 8, 4, 2, 1):
        if ch % g ==0:
            return nn.GroupNorm(num_groups=g, num_channels=ch)
    return nn.GroupNorm(num_groups=1, num_channels=ch)

class PositionalEncodingBlock(nn.Module):
    """
    Operates on sequence (B, L, D) and returns (B, L, D).
    Implements:
      - dense
      - dense + conv1d(k=3)
      - dense + conv1d(k=5)
      - maxpool1d + dense
    Uses a learnable scalar alpha for residual scaling: out = x + alpha * f(x)
    """

    def __init__(self, dim: int, mode: PosBlockMode = "dense_conv1d_3", alpha_init: float = 1e-3):
        super().__init__()
        self.dim = dim
        self.mode = mode

        # Dense exists in all modes here (even maxpool_dense uses dense2)
        self.dense = nn.Linear(dim, dim)

        if mode in ("dense_conv1d_3", "dense_conv1d_5"):
            k = 3 if mode.endswith("_3") else 5
            self.conv1d = nn.Conv1d(dim, dim, kernel_size=k, padding=k // 2)

        if mode == "maxpool_dense":
            self.pool1d = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
            self.dense2 = nn.Linear(dim, dim)

        # Learnable residual scale (LayerScale-style)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        # Nonlinearities (kept simple + stable)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        """
        residual = x

        if self.mode == "dense":
            y = self.act(self.dense(x))  # (B, L, D)
            return residual + self.alpha * y

        if self.mode in ("dense_conv1d_3", "dense_conv1d_5"):
            y = self.act(self.dense(x))  # (B, L, D)
            y = y.permute(0, 2, 1)       # (B, D, L)
            y = self.act(self.conv1d(y)) # (B, D, L)
            y = y.permute(0, 2, 1)       # (B, L, D)
            return residual + self.alpha * y

        if self.mode == "maxpool_dense":
            # pool over token dimension (L)
            y = x.permute(0, 2, 1)       # (B, D, L)
            y = self.pool1d(y)           # (B, D, L)
            y = y.permute(0, 2, 1)       # (B, L, D)
            y = self.act(self.dense2(y)) # (B, L, D)
            return residual + self.alpha * y

        raise ValueError(f"Unknown PositionalEncodingBlock mode: {self.mode}")

def add_noise_safe(tensor, noise_std_ratio=0.6):
    std = torch.std(tensor)
    noise_std = std * noise_std_ratio
    noise = torch.randn_like(tensor) * noise_std
    return tensor + noise
class P_Module(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_depth_point: bool = True,
        use_multi_scale: bool = True,
        use_se: bool = True,
        use_skip: bool = True,
        noise_std_ratio=0.6,
        pos_block_mode: PosBlockMode = "dense_conv1d_3",
        alpha_init: float = 1e-3,
        num_classes= None,
        apply_softmax: bool = False,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.use_depth_point = use_depth_point
        self.use_multi_scale = use_multi_scale
        self.use_se = use_se
        self.use_skip = use_skip
        self.noise_std_ratio = float(noise_std_ratio)

        self.pos_block_mode = pos_block_mode
        self.num_classes = num_classes
        self.apply_softmax = apply_softmax

        # ---------------- Patch Embedding ----------------
        # Input: (B, in_channels, H, W) -> (B, out_channels, H, W)
        if self.use_depth_point:
            self.depthwise = nn.Conv2d(
                in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False
            )
            self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.proj_norm = make_gn(out_channels)
        else:
            self.simple_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.proj_norm = make_gn(out_channels)

        # Multi-activation triplet
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.gelu = nn.GELU()

        # ---------------- Positional Encoding (2D) ----------------
        # "positional encoding" is implemented as convolutional position-dependent mix
        if self.use_multi_scale:
            self.dil_conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False)
            self.dil_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2, bias=False)
            self.dil_conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=4, dilation=4, bias=False)
            self.fuse = nn.Conv2d(out_channels * 3, out_channels, kernel_size=3, padding=1, bias=False)
            self.fuse_norm = make_gn(out_channels)
        else:
            self.fuse = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
            self.fuse_norm = make_gn(out_channels)

        # ---------------- Positional Encoding Block (token-wise variants) ----------------
        # Convert (B, D, H, W) <-> (B, L, D), apply sequence block, then reshape back.
        self.pos_block = PositionalEncodingBlock(out_channels, mode=pos_block_mode, alpha_init=alpha_init)

        # ---------------- Squeeze-Excitation (optional) ----------------
        # SE runs on channel dimension of (B, C, H, W).
        if self.use_se:
            se_in = out_channels
            se_hidden = max(8, out_channels // 4)
            self.se_fc1 = nn.Linear(se_in, se_hidden)
            self.se_fc2 = nn.Linear(se_hidden, se_in)

        # ---------------- Scalar Weighting / Skip ----------------
        # 1) A learnable scalar gate for the entire module output (stabilizes training; like LayerScale at module-level)
        self.gamma = nn.Parameter(torch.tensor(1.0))
        # 2) skip projection to align channels
        if self.use_skip:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.skip_norm = make_gn(out_channels)

        # ----------------(x) classification head ----------------
        self.dropout = nn.Dropout(p=float(dropout_p)) if dropout_p > 0 else nn.Identity()
        if num_classes is not None:
            self.classifier = nn.Linear(out_channels, num_classes)
    
     
 
    def _patch_embed(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_depth_point:
            z = self.pointwise(self.depthwise(x))
            z = self.proj_norm(z)
            # multiplicative
            y = self.gelu(z) * self.elu(z) * self.relu(z)
            return y
        y = self.relu(self.proj_norm(self.simple_proj(x)))
        return y

    def _positional_encode_2d(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, D, H, W)
        Returns: (B, D, H, W)
        """
        if self.use_multi_scale:
            d1 = self.gelu(self.dil_conv1(x))
            d2 = self.elu(self.dil_conv2(x))
            d4 = self.relu(self.dil_conv4(x))
            # concat along channel
            cat = torch.cat([d1, d2, d4], dim=1)  # (B, 3D, H, W)
            y = self.fuse_norm(self.fuse(cat))    # (B, D, H, W)
            return self.elu(y)
        y = self.fuse_norm(self.fuse(x))
        return self.elu(y)

    def _positional_block_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, D, H, W) -> tokens (B, L, D) -> apply pos_block -> back to (B, D, H, W)
        """
        B, D, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, L, D) with L=H*W
        tokens = self.pos_block(tokens)        # (B, L, D)
        x2 = tokens.transpose(1, 2).view(B, D, H, W)
        return x2

    def _se(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, D, H, W)
        """
        if not self.use_se:
            return x
        B, D, _, _ = x.shape
        s = F.adaptive_avg_pool2d(x, 1).view(B, D)  # (B, D)
        s = self.relu(self.se_fc1(s))
        s = torch.sigmoid(self.se_fc2(s))          # (B, D), sigmoid for SE
        s = s.view(B, D, 1, 1)
        return x * s

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        
        x = add_noise_safe(x, self.noise_std_ratio)

        # 1) Patch Embedding / projection
        pe = self._patch_embed(x)                # (B, D, H, W)

        # 2) Positional Encoding (2D multi-scale)
        pe = self._positional_encode_2d(pe)      # (B, D, H, W)

        # 3) Positional Encoding Block (token-wise variants)
        pe = self._positional_block_tokens(pe)   # (B, D, H, W)

        # 4) Optional SE attention
        pe = self._se(pe)                        # (B, D, H, W)

        # 5) Scalar weighting + optional skip
        y = self.gamma * pe
        if self.use_skip:
            skip = self.skip_norm(self.skip(x))
            y = y + skip

        # If no classifier head, return feature map
        if self.num_classes is None:
            return y

        # 6) GAP + classifier
        pooled = F.adaptive_avg_pool2d(y, 1).flatten(1)  # (B, D)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)                 # (B, num_classes)

        if self.apply_softmax:
            probs = torch.softmax(logits, dim=1)
            return (probs, y) if return_features else probs
        return (logits, y) if return_features else logits


if __name__ == "__main__":
    # quick sanity check
    B, C, H, W = 2, 1280, 7, 7
    x = torch.randn(B, C, H, W)

    m = P_Module(
        in_channels=C,
        out_channels=512,
        use_depth_point=True,
        use_multi_scale=True,
        use_se=True,
        use_skip=True,
        noise_std_ratio=0.6,
        pos_block_mode="dense_conv1d_3",
        num_classes=2,
        apply_softmax=False,
    )

    out = m(x)
    print("Output shape:", out.shape)
