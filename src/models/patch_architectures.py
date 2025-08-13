import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchResNet(nn.Module):
    def __init__(self, base_model, feature_dim, num_classes, num_patches):
        super().__init__()
        self.base_model = base_model
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * num_patches, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
        self.num_patches = num_patches

    def forward(self, x):
        batch_size, num_patches, C, H, W = x.size()
        x = x.view(-1, C, H, W)
        x = self.base_model(x)
        x = x.view(batch_size, num_patches * x.size(-1))
        x = self.classifier(x)
        return x


class PatchTransformerClassifier(nn.Module):
    def __init__(
        self, base_model, feature_dim, num_classes, num_patches, nhead=8, num_layers=1
    ):
        super().__init__()
        self.base_model = base_model
        self.num_patches = num_patches
        self.feature_dim = feature_dim
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, feature_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            dim_feedforward=feature_dim,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * num_patches, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        batch_size, num_patches, C, H, W = x.size()
        x = x.view(-1, C, H, W)
        x = self.base_model(x)
        x = x.view(batch_size, num_patches, self.feature_dim)
        x = x + self.pos_embed
        x = self.transformer_encoder(x)
        x = x.contiguous().view(batch_size, -1)
        x = self.classifier(x)
        return x


class TokenMixerClassifier(nn.Module):
    def __init__(
        self, base_model, feature_dim, num_classes, num_patches, nhead=4, num_layers=1
    ):
        super().__init__()
        self.base_model = base_model
        self.feature_dim = feature_dim
        self.num_patches = num_patches
        self.tokenizer = nn.Sequential(
            nn.Conv2d(
                feature_dim, feature_dim // 4, kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(feature_dim // 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.reduced_dim = feature_dim // 4
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, self.reduced_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.reduced_dim,
            nhead=nhead,
            dim_feedforward=self.reduced_dim,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.reduced_dim * num_patches, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        batch_size, num_patches, C, H, W = x.size()
        x = x.view(-1, C, H, W)
        x = self.base_model(x)
        if x.dim() == 2:
            x = x.view(batch_size * num_patches, self.feature_dim, 1, 1)
        x = self.tokenizer(x)
        x = x.view(batch_size, num_patches, self.reduced_dim)
        x = x + self.pos_embed
        x = self.transformer_encoder(x)
        x = x.contiguous().view(batch_size, -1)
        x = self.classifier(x)
        return x


class PatchGlobalLocalClassifier(nn.Module):
    def __init__(self, base_model, feature_dim, num_classes, num_patches):
        super().__init__()
        self.base_model = base_model
        self.num_patches = num_patches
        self.feature_dim = feature_dim
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, feature_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=8,
            dim_feedforward=feature_dim,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * (num_patches + 1), 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        batch_size, num_patches, C, H, W = x.size()
        x_patches = x.view(-1, C, H, W)
        feat_patches = self.base_model(x_patches)
        feat_patches = feat_patches.view(batch_size, num_patches, self.feature_dim)
        # reconstruct global image
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
        feat_global = feat_global.unsqueeze(1)
        feats = torch.cat([feat_patches, feat_global], dim=1)
        feats = feats + self.pos_embed
        feats = self.transformer_encoder(feats)
        feats = feats.contiguous().view(batch_size, -1)
        out = self.classifier(feats)
        return out


class PatchGlobalLocalTokenMixerClassifier(nn.Module):
    def __init__(
        self, base_model, feature_dim, num_classes, num_patches, nhead=4, num_layers=2
    ):
        super().__init__()
        self.base_model = base_model
        self.num_patches = num_patches
        self.feature_dim = feature_dim
        self.reduced_dim = feature_dim // 4
        self.tokenizer = nn.Sequential(
            nn.Conv2d(
                feature_dim, self.reduced_dim, kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(self.reduced_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, self.reduced_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.reduced_dim,
            nhead=nhead,
            dim_feedforward=self.reduced_dim,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.reduced_dim * (num_patches + 1), 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
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
        tokens_global = tokens_global.view(batch_size, 1, self.reduced_dim)
        tokens = torch.cat([tokens_patches, tokens_global], dim=1)
        tokens = tokens + self.pos_embed
        tokens = self.transformer_encoder(tokens)
        tokens = tokens.contiguous().view(batch_size, -1)
        out = self.classifier(tokens)
        return out


# MILClassifier, MILClassifierV2, MILClassifierV3 có thể copy từ project/patch/models.py nếu cần
# ...existing code for MILClassifier, MILClassifierV2, MILClassifierV3...
