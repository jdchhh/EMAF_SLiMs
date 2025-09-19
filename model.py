import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


class EnhancedAttention(nn.Module):
    def __init__(self, input_dim, num_heads=1):
        super().__init__()
        self.to_q = nn.Linear(input_dim, input_dim)
        self.to_k = nn.Linear(input_dim, input_dim)
        self.to_v = nn.Linear(input_dim, input_dim)

        self.scale = input_dim ** -0.5
        self.attn_proj = nn.Linear(input_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        B, L, D = x.shape
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, L, L)
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # (B, L, D)
        out = self.attn_proj(out) + x
        return self.norm(out)


class ProteinClassifier(nn.Module):
    def __init__(self, protbert_dim=128, pssm_dim=20, phy_dim=108,
                 mlp_hidden=64, num_classes=2):
        super().__init__()

        self.pssm_att = EnhancedAttention(pssm_dim)
        self.phy_att = EnhancedAttention(phy_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=protbert_dim + pssm_dim + phy_dim,
            num_heads=4,
            batch_first=False
        )

        self.mlp = nn.Sequential(
            nn.Linear(protbert_dim + pssm_dim + phy_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(mlp_hidden, num_classes)
        )

    def forward(self, protbert, pssm, phy):
        pssm_enhanced = self.pssm_att(pssm)  # (B, L, 20)
        phy_enhanced = self.phy_att(phy)  # (B, L, 108)
        combined = torch.cat([
            protbert,  # (B, L, 128)
            pssm_enhanced,  # (B, L, 20)
            phy_enhanced  # (B, L, 108)
        ], dim=-1)  # (B, L, 256)

        attn_output, _ = self.cross_attn(
            query=combined.transpose(0, 1),
            key=combined.transpose(0, 1),
            value=combined.transpose(0, 1)
        )

        return self.mlp(attn_output.mean(dim=0))  # (B, num_classes)
