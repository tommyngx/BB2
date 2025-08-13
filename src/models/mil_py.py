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
        # x: (batch_size, num_patches+1, C, H, W)
        batch_size, num_patches_plus1, C, H, W = x.size()
        num_patches = num_patches_plus1 - 1
        x_local = x[:, :num_patches]  # (B, N, C, H, W)
        x_global = x[:, num_patches]  # (B, C, H, W)

        # Local patch features
        x_patches = x_local.contiguous().view(-1, C, H, W)
        feat_patches = self.base_model(x_patches)
        if feat_patches.dim() == 2:
            feat_patches = feat_patches.view(
                batch_size * num_patches, self.feature_dim, 1, 1
            )
        tokens_patches = self.tokenizer(feat_patches)
        tokens_patches = tokens_patches.view(batch_size, num_patches, self.reduced_dim)

        # Attention pooling
        A_V = self.attention_V(tokens_patches)
        A_U = self.attention_U(tokens_patches)
        A = self.attention_weights(A_V * A_U)
        A = torch.transpose(A, 2, 1)
        A = nn.functional.softmax(A, dim=2)
        M = torch.bmm(A, tokens_patches)
        M = M.view(batch_size, -1)

        # Global image features
        global_patch_resized = x_global  # (B, C, H, W)
        feat_global = self.base_model(global_patch_resized)
        if feat_global.dim() == 2:
            feat_global = feat_global.view(batch_size, self.feature_dim, 1, 1)
        tokens_global = self.tokenizer(feat_global)
        tokens_global = tokens_global.view(batch_size, self.reduced_dim)
        processed_global = self.global_processor(tokens_global)

        # Fusion and classification
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
        # x: (B, N, C, H, W)
        B, N, C, H, W = x.shape
        x = x.contiguous().view(B * N, C, H, W)
        feats = self.base_model(x)
        if feats.dim() == 4:
            feats = F.adaptive_avg_pool2d(feats, 1).squeeze(-1).squeeze(-1)
        feats = feats.view(B, N, self.feature_dim)
        return feats

    def forward(self, x, mask=None, temperature=1.0):
        # x: (B, N+1, C, H, W)
        B, N_plus1, C, H, W = x.shape
        N = N_plus1 - 1
        x_local = x[:, :N]  # (B, N, C, H, W)
        feats = self._encode_patches(x_local)
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


class MILClassifierV4(nn.Module):
    """
    MILClassifierV4:
    - Attention pooling cho local patch (MIL)
    - Global feature làm query trong cross-attention, ALL local patches làm key/value
    - Đầu ra fused qua head classifier
    """

    def __init__(
        self,
        base_model_local: nn.Module,
        base_model_global: nn.Module,
        local_dim: int,
        global_dim: int,
        num_classes: int = 1,
        attn_hidden: int = 256,
        attn_dropout: float = 0.1,
        head_dropout: float = 0.1,
        cross_attn_heads: int = 4,
    ):
        super().__init__()
        self.base_model_local = base_model_local
        self.base_model_global = base_model_global
        self.local_dim = local_dim
        self.global_dim = global_dim
        self.num_classes = num_classes

        # MIL attention pooling cho local patch
        self.mil_pool = _GatedAttnPool(
            d_model=local_dim, hidden=attn_hidden, dropout=attn_dropout
        )

        # Cross-attention: global feature làm query, ALL local patches làm key/value
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=local_dim, num_heads=cross_attn_heads, batch_first=True
        )
        self.global_proj = nn.Linear(global_dim, local_dim)

        # Head classifier
        self.head = nn.Sequential(
            nn.LayerNorm(local_dim),
            nn.Dropout(head_dropout),
            nn.Linear(local_dim, num_classes),
        )

        # Init weights
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
        nn.init.xavier_uniform_(self.global_proj.weight)
        nn.init.constant_(self.global_proj.bias, 0.0)

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
        B, N_plus_1, C, H, W = x_patches.shape
        N = N_plus_1 - 1
        x_local = x_patches[:, :N]  # (B, N, C, H, W)
        x_global = x_patches[:, N]  # (B, C, H, W)

        feats_l = self._encode_patches(x_local)  # (B, N, local_dim)
        pooled_l, attn = self.mil_pool(
            feats_l, mask=mask, temperature=1.0
        )  # (B, local_dim)
        feats_g = self._encode_global(x_global)  # (B, global_dim)
        feats_g_proj = self.global_proj(feats_g).unsqueeze(1)  # (B, 1, local_dim)

        # Cross-attention: global query, ALL local patches as key/value
        cross_attn_out, _ = self.cross_attn(
            query=feats_g_proj, key=feats_l, value=feats_l
        )  # (B, 1, local_dim)
        fused = cross_attn_out.squeeze(1)  # (B, local_dim)

        logits = self.head(fused)
        return logits


class MILClassifierV5(nn.Module):
    """
    MILClassifierV5: Advanced MIL classifier with:
    - Dual-path processing (local MIL + global)
    - Cross-attention between global and local features
    - Residual connections and layer normalization
    - Adaptive fusion mechanism
    """

    def __init__(
        self,
        base_model_local: nn.Module,
        base_model_global: nn.Module,
        local_dim: int,
        global_dim: int,
        num_classes: int = 1,
        attn_hidden: int = 256,
        attn_dropout: float = 0.1,
        head_dropout: float = 0.1,
        cross_attn_heads: int = 8,
        fusion_method: str = "adaptive",
    ):
        super().__init__()
        self.base_model_local = base_model_local
        self.base_model_global = base_model_global
        self.local_dim = local_dim
        self.global_dim = global_dim
        self.num_classes = num_classes
        self.fusion_method = fusion_method

        # MIL attention pooling
        self.mil_pool = _GatedAttnPool(
            d_model=local_dim, hidden=attn_hidden, dropout=attn_dropout
        )

        # Project global features to local dimension
        self.global_proj = nn.Sequential(
            nn.Linear(global_dim, local_dim),
            nn.LayerNorm(local_dim),
            nn.ReLU(inplace=True),
        )

        # Cross-attention layers
        self.cross_attn_local = nn.MultiheadAttention(
            embed_dim=local_dim, num_heads=cross_attn_heads, batch_first=True
        )
        self.cross_attn_global = nn.MultiheadAttention(
            embed_dim=local_dim, num_heads=cross_attn_heads, batch_first=True
        )

        # Layer norms for residual connections
        self.ln_local = nn.LayerNorm(local_dim)
        self.ln_global = nn.LayerNorm(local_dim)

        # Fusion mechanism
        if fusion_method == "adaptive":
            self.fusion_gate = nn.Sequential(
                nn.Linear(local_dim * 2, local_dim),
                nn.ReLU(inplace=True),
                nn.Linear(local_dim, 2),
                nn.Softmax(dim=-1),
            )
        elif fusion_method == "concat":
            self.fusion_proj = nn.Linear(local_dim * 2, local_dim)

        # Enhanced head with residual connection
        self.head = nn.Sequential(
            nn.LayerNorm(local_dim),
            nn.Linear(local_dim, local_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(local_dim // 2, num_classes),
        )

        self._init_weights()
        self.last_attn_weights = None
        self.last_fusion_weights = None

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

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
        B, N_plus_1, C, H, W = x_patches.shape
        N = N_plus_1 - 1
        x_local = x_patches[:, :N]
        x_global = x_patches[:, N]

        # Encode patches and global image
        feats_l = self._encode_patches(x_local)  # (B, N, local_dim)
        pooled_l, attn_weights = self.mil_pool(feats_l, mask=mask)  # (B, local_dim)

        feats_g = self._encode_global(x_global)  # (B, global_dim)
        feats_g_proj = self.global_proj(feats_g)  # (B, local_dim)

        # Cross-attention
        # Local-to-global attention
        local_enhanced, _ = self.cross_attn_local(
            query=pooled_l.unsqueeze(1),
            key=feats_g_proj.unsqueeze(1),
            value=feats_g_proj.unsqueeze(1),
        )
        local_enhanced = self.ln_local(pooled_l + local_enhanced.squeeze(1))

        # Global-to-local attention
        global_enhanced, _ = self.cross_attn_global(
            query=feats_g_proj.unsqueeze(1), key=feats_l, value=feats_l
        )
        global_enhanced = self.ln_global(feats_g_proj + global_enhanced.squeeze(1))

        # Fusion
        if self.fusion_method == "adaptive":
            fusion_input = torch.cat([local_enhanced, global_enhanced], dim=-1)
            fusion_weights = self.fusion_gate(fusion_input)  # (B, 2)
            fused = (
                fusion_weights[:, 0:1] * local_enhanced
                + fusion_weights[:, 1:2] * global_enhanced
            )
            self.last_fusion_weights = fusion_weights.detach()
        elif self.fusion_method == "concat":
            fused = self.fusion_proj(
                torch.cat([local_enhanced, global_enhanced], dim=-1)
            )
        else:  # average
            fused = (local_enhanced + global_enhanced) / 2

        # Classification
        logits = self.head(fused)
        self.last_attn_weights = attn_weights.detach()

        return logits
