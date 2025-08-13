import torch
import torch.nn as nn
import torch.nn.functional as F


class MILClassifier(nn.Module):
    def __init__(
        self,
        base_model,
        feature_dim,
        num_classes,
        num_patches,
        attention_dim=128,
        nhead=4,
        num_layers=2,
    ):
        super().__init__()
        self.base_model = base_model
        self.num_patches = num_patches
        self.feature_dim = feature_dim
        self.attention_dim = attention_dim
        self.reduced_dim = feature_dim // 4

        self.tokenizer = nn.Sequential(
            nn.Conv2d(
                feature_dim, self.reduced_dim, kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(self.reduced_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.reduced_dim, attention_dim), nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.reduced_dim, attention_dim), nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(attention_dim, 1)

        self.global_processor = nn.Sequential(
            nn.Linear(self.reduced_dim, self.reduced_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

        self.fusion_layer = nn.Sequential(
            nn.Linear(self.reduced_dim + self.reduced_dim // 2, self.reduced_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.reduced_dim, self.reduced_dim // 2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.reduced_dim // 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        batch_size, num_patches, C, H, W = x.size()
        x_patches = x.view(-1, C, H, W)
        feat_patches = self.base_model(x_patches)
        if feat_patches.dim() == 2:
            feat_patches = feat_patches.view(
                batch_size * num_patches, self.feature_dim, 1, 1
            )
        tokens_patches = self.tokenizer(feat_patches)
        tokens_patches = tokens_patches.view(batch_size, num_patches, self.reduced_dim)

        A_V = self.attention_V(tokens_patches)
        A_U = self.attention_U(tokens_patches)
        A = self.attention_weights(A_V * A_U)
        A = torch.transpose(A, 2, 1)
        A = nn.functional.softmax(A, dim=2)
        M = torch.bmm(A, tokens_patches)
        M = M.view(batch_size, -1)

        overlap_ratio = 0.2
        patch_height = H
        step = int(patch_height * (1 - overlap_ratio))
        full_height = step * (num_patches - 1) + patch_height
        full_img = torch.zeros(batch_size, C, full_height, W, device=x.device)
        count = torch.zeros(batch_size, 1, full_height, W, device=x.device)
        for i in range(num_patches):
            start_h = i * step
            end_h = start_h + patch_height
            full_img[:, :, start_h:end_h, :] += x[:, i]
            count[:, :, start_h:end_h, :] += 1
        full_img = full_img / count.clamp(min=1.0)
        global_patch_resized = nn.functional.interpolate(
            full_img, size=(H, W), mode="bilinear", align_corners=False
        )
        feat_global = self.base_model(global_patch_resized)
        if feat_global.dim() == 2:
            feat_global = feat_global.view(batch_size, self.feature_dim, 1, 1)
        tokens_global = self.tokenizer(feat_global)
        tokens_global = tokens_global.view(batch_size, self.reduced_dim)
        processed_global = self.global_processor(tokens_global)
        fused_features = torch.cat([M, processed_global], dim=1)
        fused_features = self.fusion_layer(fused_features)
        logits = self.classifier(fused_features)
        return logits


class MILClassifierV2(nn.Module):
    def __init__(
        self,
        base_model,
        feature_dim,
        num_classes,
        attn_hidden=256,
        attn_dropout=0.1,
        head_dropout=0.1,
    ):
        super().__init__()
        self.base_model = base_model
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        self.attn_V = nn.Linear(feature_dim, attn_hidden)
        self.attn_U = nn.Linear(feature_dim, attn_hidden)
        self.attn_w = nn.Linear(attn_hidden, 1)
        self.attn_drop = nn.Dropout(attn_dropout)

        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(head_dropout),
            nn.Linear(feature_dim, num_classes),
        )

        self._init_weights()
        self.last_attn_weights = None

    def _init_weights(self):
        for m in [self.attn_V, self.attn_U, self.attn_w]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def _encode_patches(self, x):
        B, N, C, H, W = x.shape
        x = x.contiguous().view(B * N, C, H, W)
        feats = self.base_model(x)
        if feats.dim() == 4:
            feats = F.adaptive_avg_pool2d(feats, 1).squeeze(-1).squeeze(-1)
        feats = feats.view(B, N, self.feature_dim)
        return feats

    def forward(self, x, mask=None, temperature=1.0):
        feats = self._encode_patches(x)
        V = torch.tanh(self.attn_V(feats))
        U = torch.sigmoid(self.attn_U(feats))
        scores = self.attn_w(self.attn_drop(V * U)).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        attn_weights = torch.softmax(scores / max(temperature, 1e-6), dim=1)
        pooled = torch.bmm(attn_weights.unsqueeze(1), feats).squeeze(1)
        logits = self.head(pooled)
        self.last_attn_weights = attn_weights.detach()
        return logits


class _GatedAttnPool(nn.Module):
    def __init__(self, d_model: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.V = nn.Linear(d_model, hidden)
        self.U = nn.Linear(d_model, hidden)
        self.w = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.constant_(self.V.bias, 0.0)
        nn.init.xavier_uniform_(self.U.weight)
        nn.init.constant_(self.U.bias, 0.0)
        nn.init.xavier_uniform_(self.w.weight)
        nn.init.constant_(self.w.bias, 0.0)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor = None, temperature: float = 1.0
    ):
        v = torch.tanh(self.V(x))
        u = torch.sigmoid(self.U(x))
        scores = self.w(self.drop(v * u)).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores / max(temperature, 1e-6), dim=1)
        pooled = torch.bmm(attn.unsqueeze(1), x).squeeze(1)
        return pooled, attn


class MILClassifierV3(nn.Module):
    def __init__(
        self,
        base_model_local: nn.Module,
        base_model_global: nn.Module,
        local_dim: int,
        global_dim: int,
        num_classes: int = 1,
        fusion: str = "concat",
        attn_hidden: int = 256,
        attn_dropout: float = 0.1,
        head_dropout: float = 0.1,
        global_size: tuple = (448, 448),
    ):
        super().__init__()
        assert fusion in ("concat", "gated"), "fusion must be 'concat' or 'gated'"
        self.base_model_local = base_model_local
        self.base_model_global = base_model_global
        self.local_dim = local_dim
        self.global_dim = global_dim
        self.num_classes = num_classes
        self.fusion = fusion
        self.global_size = global_size

        self.mil_pool = _GatedAttnPool(
            d_model=local_dim, hidden=attn_hidden, dropout=attn_dropout
        )

        if fusion == "concat":
            fused_dim = local_dim + global_dim
            self.head = nn.Sequential(
                nn.LayerNorm(fused_dim),
                nn.Dropout(head_dropout),
                nn.Linear(fused_dim, num_classes),
            )
        else:
            self.global_to_local = nn.Linear(global_dim, local_dim)
            self.gate = nn.Sequential(
                nn.Linear(local_dim + local_dim, local_dim),
                nn.ReLU(inplace=True),
                nn.Linear(local_dim, 1),
                nn.Sigmoid(),
            )
            self.head = nn.Sequential(
                nn.LayerNorm(local_dim),
                nn.Dropout(head_dropout),
                nn.Linear(local_dim, num_classes),
            )

        self.last_attn_weights = None
        self.last_global_feat = None
        self.last_local_feat = None

        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
        if fusion == "gated":
            nn.init.xavier_uniform_(self.global_to_local.weight)
            nn.init.constant_(self.global_to_local.bias, 0.0)

    def _encode_patches(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = x.shape
        x_ = x.contiguous().view(B * N, C, H, W)
        feats = self.base_model_local(x_)
        if feats.dim() == 4:
            feats = F.adaptive_avg_pool2d(feats, 1).squeeze(-1).squeeze(-1)
        feats = feats.view(B, N, -1)
        return feats

    def _encode_global(self, global_img: torch.Tensor) -> torch.Tensor:
        feats_g = self.base_model_global(global_img)
        if feats_g.dim() == 4:
            feats_g = F.adaptive_avg_pool2d(feats_g, 1).squeeze(-1).squeeze(-1)
        return feats_g

    def forward(self, x_patches: torch.Tensor, mask: torch.Tensor = None):
        # x_patches: (B, N+1, C, H, W)
        # Use first N patches for MIL, last patch as global image
        B, N_plus_1, C, H, W = x_patches.shape
        N = N_plus_1 - 1
        x_local = x_patches[:, :N]  # (B, N, C, H, W)
        x_global = x_patches[:, N]  # (B, C, H, W)

        feats_l = self._encode_patches(x_local)
        pooled_l, attn = self.mil_pool(feats_l, mask=mask, temperature=1.0)
        feats_g = self._encode_global(x_global)

        if self.fusion == "concat":
            fused = torch.cat([pooled_l, feats_g], dim=1)
        else:
            g_proj = self.global_to_local(feats_g)
            gate = self.gate(torch.cat([pooled_l, g_proj], dim=1))
            fused = gate * pooled_l + (1.0 - gate) * g_proj
        logits = self.head(fused)
        self.last_attn_weights = attn.detach()
        self.last_global_feat = feats_g.detach()
        self.last_local_feat = pooled_l.detach()
        return logits
