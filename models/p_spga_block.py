import math
import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, channels, kernel_size, dilation=1):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=padding, dilation=dilation, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class MultiScaleBranch(nn.Module):
    def __init__(self, channels, dilation=3):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.branch2 = DepthwiseSeparableConv(channels, kernel_size=3)
        self.branch3 = DepthwiseSeparableConv(channels, kernel_size=5)
        self.branch4 = DepthwiseSeparableConv(channels, kernel_size=3, dilation=dilation)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 4, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x):
        out = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)
        return x + self.fuse(out)


class ChunkMultiScaleBranch(nn.Module):
    def __init__(self, channels, dilation=3, expansion=2):
        super().__init__()
        if channels % 4 != 0:
            raise ValueError(f'channels must be divisible by 4 for chunk branch, got {channels}')
        chunk_channels = channels // 4
        hidden = channels * expansion
        self.branch_i = nn.Identity()
        self.branch_s = DepthwiseSeparableConv(chunk_channels, kernel_size=3)
        self.branch_m = DepthwiseSeparableConv(chunk_channels, kernel_size=5)
        self.branch_l = DepthwiseSeparableConv(chunk_channels, kernel_size=3, dilation=dilation)
        self.norm = nn.BatchNorm2d(channels)
        self.channel_mixer = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        x_i, x_s, x_m, x_l = torch.chunk(x, chunks=4, dim=1)
        mixed = torch.cat([
            self.branch_i(x_i),
            self.branch_s(x_s),
            self.branch_m(x_m),
            self.branch_l(x_l),
        ], dim=1)
        return self.channel_mixer(self.norm(mixed))


class StatisticalChannelGate(nn.Module):
    def __init__(self, channels, stat_type='mean_std_range', reduction=16):
        super().__init__()
        if stat_type not in {'mean_std', 'mean_std_range'}:
            raise ValueError(f'Unsupported stat_type: {stat_type}')
        self.stat_type = stat_type
        num_stats = 2 if stat_type == 'mean_std' else 3
        hidden = max(4, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channels * num_stats, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        mean = x.mean(dim=(2, 3))
        std = x.std(dim=(2, 3), unbiased=False)
        stats = [mean, std]
        if self.stat_type == 'mean_std_range':
            stats.append(x.amax(dim=(2, 3)) - x.amin(dim=(2, 3)))
        gate = self.mlp(torch.cat(stats, dim=1)).view(x.size(0), x.size(1), 1, 1)
        return x * gate, gate


class StatisticalSpatialPrior(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        spatial_mean = x.mean(dim=1, keepdim=True)
        spatial_std = x.std(dim=1, keepdim=True, unbiased=False)
        spatial_max = x.max(dim=1, keepdim=True)[0]
        return self.conv(torch.cat([spatial_mean, spatial_std, spatial_max], dim=1))


class LearnableSpatialRefinement(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.GELU(),
            nn.Conv2d(8, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        spatial_mean = x.mean(dim=1, keepdim=True)
        spatial_std = x.std(dim=1, keepdim=True, unbiased=False)
        spatial_max = x.max(dim=1, keepdim=True)[0]
        return self.spatial(torch.cat([spatial_mean, spatial_std, spatial_max], dim=1))


class AlphaScheduler:
    def __init__(self, alpha_max=0.8, alpha_min=0.2, total_epochs=50):
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.total_epochs = max(1, total_epochs)

    def get_alpha(self, epoch=None):
        if epoch is None:
            return 0.5
        epoch = min(max(0, int(epoch)), self.total_epochs)
        return self.alpha_min + 0.5 * (self.alpha_max - self.alpha_min) * (1.0 + math.cos(math.pi * epoch / self.total_epochs))


class PSPGABlock(nn.Module):
    """Plug-in P-SPGA block for existing CNN feature maps [B, C, H, W]."""

    def __init__(
        self,
        channels,
        dilation=3,
        stat_type='mean_std_range',
        alpha_max=0.8,
        alpha_min=0.2,
        total_epochs=50,
        use_branch=True,
        branch_type='full',
        use_stat_gate=True,
        use_prior_map=True,
        use_learnable_attention=True,
        use_progressive_fusion=True,
        residual_scale=1.0,
    ):
        super().__init__()
        self.use_stat_gate = use_stat_gate
        self.use_prior_map = use_prior_map
        self.use_learnable_attention = use_learnable_attention
        self.use_progressive_fusion = use_progressive_fusion
        self.residual_scale = residual_scale
        self.scheduler = AlphaScheduler(alpha_max=alpha_max, alpha_min=alpha_min, total_epochs=total_epochs)
        if not use_branch:
            self.branch = nn.Identity()
        elif branch_type == 'full':
            self.branch = MultiScaleBranch(channels, dilation=dilation)
        elif branch_type == 'chunk':
            self.branch = ChunkMultiScaleBranch(channels, dilation=dilation)
        else:
            raise ValueError(f'Unsupported branch_type: {branch_type}')
        self.channel_gate = StatisticalChannelGate(channels, stat_type=stat_type) if use_stat_gate else None
        self.prior_map = StatisticalSpatialPrior() if use_prior_map else None
        self.learnable_attention = LearnableSpatialRefinement() if use_learnable_attention else None

    def forward(self, x, epoch=None, return_attention=False):
        f1 = self.branch(x)
        channel_gate = None
        if self.channel_gate is not None:
            f2, channel_gate = self.channel_gate(f1)
        else:
            f2 = f1

        ps = self.prior_map(f2) if self.prior_map is not None else None
        al = self.learnable_attention(f2) if self.learnable_attention is not None else None
        alpha = self.scheduler.get_alpha(epoch)

        if self.use_progressive_fusion and ps is not None and al is not None:
            attention = alpha * ps + (1.0 - alpha) * al
        elif ps is not None:
            attention = ps
        elif al is not None:
            attention = al
        else:
            attention = None

        out = x + self.residual_scale * f2 * attention if attention is not None else f2
        if return_attention:
            return out, {
                'channel_gate': channel_gate,
                'statistical_prior': ps,
                'learnable_attention': al,
                'fused_attention': attention,
                'alpha': alpha,
            }
        return out