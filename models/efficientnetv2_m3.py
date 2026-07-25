
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.efficientnetv2_P1 import get_efficientnet_v2
from models.p_spga_block import PSPGABlock


class LatentAttentionPooling(nn.Module):
    def __init__(self, in_channels, num_heads=8, num_latents=1, dropout=0.1, max_h=32, max_w=32):
        super().__init__()
        if in_channels % num_heads != 0:
            raise ValueError(f'in_channels={in_channels} must be divisible by num_heads={num_heads}')
        self.latent_query = nn.Parameter(torch.randn(1, num_latents, in_channels))
        nn.init.xavier_uniform_(self.latent_query)
        self.row_embed = nn.Parameter(torch.randn(max_h, in_channels // 2))
        self.col_embed = nn.Parameter(torch.randn(max_w, in_channels // 2))
        nn.init.normal_(self.row_embed, std=0.02)
        nn.init.normal_(self.col_embed, std=0.02)
        self.mha = nn.MultiheadAttention(in_channels, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ln_k = nn.LayerNorm(in_channels)
        self.ln_q = nn.LayerNorm(in_channels)
        self.ln_out = nn.LayerNorm(in_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, c, h, w = x.shape
        if h > self.row_embed.size(0) or w > self.col_embed.size(0):
            raise ValueError(f'feature map {h}x{w} exceeds positional capacity')
        row = self.row_embed[:h].unsqueeze(1).repeat(1, w, 1)
        col = self.col_embed[:w].unsqueeze(0).repeat(h, 1, 1)
        pos = torch.cat([row, col], dim=-1).view(1, h * w, c).repeat(b, 1, 1)
        tokens = x.flatten(2).transpose(1, 2) + pos
        tokens = self.ln_k(tokens)
        query = self.ln_q(self.latent_query.expand(b, -1, -1))
        pooled, _ = self.mha(query=query, key=tokens, value=tokens)
        pooled = self.ln_out(self.dropout(pooled))
        return F.gelu(pooled.squeeze(1))


class EfficientNetV2PSPGA_AttnPool(nn.Module):
    """M3: EfficientNetV2 + A5-style P-SPGA + latent attention pooling head."""

    def __init__(self, num_labels=4, hidden_dim=512, model_variant='s', dilation=3,
                 stat_type='mean_std_range', num_heads=8, attn_dropout=0.1):
        super().__init__()
        configs = {
            's': {'model_name': 'efficientnet_v2_s', 'dropout': 0.2, 'stochastic_depth': 0.2},
            'm': {'model_name': 'efficientnet_v2_m', 'dropout': 0.3, 'stochastic_depth': 0.3},
            'l': {'model_name': 'efficientnet_v2_l', 'dropout': 0.4, 'stochastic_depth': 0.4},
            'xl': {'model_name': 'efficientnet_v2_xl', 'dropout': 0.4, 'stochastic_depth': 0.5},
        }
        config = configs.get(model_variant, configs['s'])
        self.backbone = get_efficientnet_v2(
            model_name=config['model_name'],
            pretrained=False,
            nclass=0,
            dropout=config['dropout'],
            stochastic_depth=config['stochastic_depth'],
        )
        self.backbone.head.classifier = nn.Identity()
        self.pspga_block = PSPGABlock(
            channels=self.backbone.final_stage_channel,
            dilation=dilation,
            stat_type=stat_type,
            use_branch=True,
            branch_type='chunk',
            use_stat_gate=True,
            use_prior_map=True,
            use_learnable_attention=True,
            use_progressive_fusion=True,
            residual_scale=0.1,
        )
        num_features = 1280
        self.attn_pool = LatentAttentionPooling(num_features, num_heads=num_heads, dropout=attn_dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Linear(num_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, images, labels=None, epoch=None):
        x = self.backbone.stem(images)
        x = self.backbone.blocks(x)
        x = self.pspga_block(x, epoch=epoch)
        x = self.backbone.head.bottleneck(x)
        features = self.attn_pool(x)
        logits = self.classifier(features)
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, logits
        return logits
