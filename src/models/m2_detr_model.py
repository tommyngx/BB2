import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .m2_model import (
    ResNetFeatureWrapper,
    TimmFeatureWrapper,
    DinoFeatureWrapper,
    SpatialAttention,
)
from .backbone import get_resnet_backbone, get_timm_backbone, get_dino_backbone


class PositionEmbeddingSine(nn.Module):
    """
    Sine-cosine positional encoding (from DETR)
    Helps model understand spatial locations
    """

    def __init__(self, num_pos_feats=128, temperature=10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            pos: [B, C, H, W]
        """
        B, C, H, W = x.shape

        # Create coordinate grids
        y_embed = (
            torch.arange(H, dtype=torch.float32, device=x.device)
            .unsqueeze(1)
            .repeat(1, W)
        )
        x_embed = (
            torch.arange(W, dtype=torch.float32, device=x.device)
            .unsqueeze(0)
            .repeat(H, 1)
        )

        # Normalize to [0, 1]
        y_embed = y_embed / H
        x_embed = x_embed / W

        # Create sine-cosine embeddings
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t

        pos_x = torch.stack(
            [pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()], dim=3
        ).flatten(2)
        pos_y = torch.stack(
            [pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()], dim=3
        ).flatten(2)

        pos = torch.cat([pos_y, pos_x], dim=2).permute(2, 0, 1)  # [C, H, W]
        pos = pos.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, C, H, W]

        return pos


class EfficientDeformableAttention(nn.Module):
    """
    Efficient Deformable Attention with reduced complexity
    """

    def __init__(self, embed_dim, num_heads=8, num_points=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = embed_dim // num_heads

        # Learnable offset and attention
        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_points)

        # Value projection
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize offsets to cover nearby regions
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 2
        )
        for i in range(self.num_points):
            grid_init_i = grid_init.clone()
            grid_init_i *= (i + 1) / self.num_points  # Gradually increase offset
            if i == 0:
                offset_init = grid_init_i
            else:
                offset_init = torch.cat([offset_init, grid_init_i], dim=1)
        self.sampling_offsets.bias.data = offset_init.view(-1)

        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)

    def forward(self, query, reference_points, value, spatial_shapes):
        """
        Args:
            query: [B, N_q, C]
            reference_points: [B, N_q, 2] in [0, 1]
            value: [B, H*W, C]
            spatial_shapes: (H, W)
        Returns:
            output: [B, N_q, C]
        """
        B, N_q, C = query.shape
        H, W = spatial_shapes

        # Generate offsets and weights
        offsets = self.sampling_offsets(query).view(
            B, N_q, self.num_heads, self.num_points, 2
        )
        attn_weights = self.attention_weights(query).view(
            B, N_q, self.num_heads, self.num_points
        )
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Normalize offsets
        offsets = offsets / torch.tensor([W, H], device=query.device) * 0.1

        # Sampling locations
        sampling_locations = reference_points[:, :, None, None, :] + offsets
        sampling_locations = sampling_locations.clamp(0, 1)

        # Reshape value to 2D spatial
        value_spatial = self.value_proj(value).view(
            B, H, W, self.num_heads, self.head_dim
        )
        value_spatial = value_spatial.permute(
            0, 3, 1, 2, 4
        )  # [B, num_heads, H, W, head_dim]

        # Sample features using grid_sample
        sampled_features = []
        for head in range(self.num_heads):
            # Prepare grid for grid_sample: [B, N_q, num_points, 2]
            grid = sampling_locations[:, :, head, :, :].clone()
            grid[..., 0] = grid[..., 0] * 2 - 1  # Normalize to [-1, 1]
            grid[..., 1] = grid[..., 1] * 2 - 1

            # Sample: [B, head_dim, H, W] -> [B, head_dim, N_q, num_points]
            value_head = value_spatial[:, head].permute(
                0, 3, 1, 2
            )  # [B, head_dim, H, W]
            sampled = F.grid_sample(
                value_head, grid, mode="bilinear", align_corners=False
            )
            sampled = sampled.permute(0, 2, 3, 1)  # [B, N_q, num_points, head_dim]

            # Weighted sum
            weighted = (sampled * attn_weights[:, :, head, :, None]).sum(
                dim=2
            )  # [B, N_q, head_dim]
            sampled_features.append(weighted)

        # Concatenate heads
        output = torch.cat(sampled_features, dim=-1)  # [B, N_q, C]
        output = self.output_proj(output)

        return output


class EfficientDETRDecoder(nn.Module):
    """
    Efficient DETR Decoder:
    - Reduced layers (2 instead of 6)
    - Auxiliary losses for faster convergence
    - Iterative bbox refinement
    """

    def __init__(self, feature_dim, num_queries=5, num_heads=4, num_layers=2):
        super().__init__()
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.feature_dim = feature_dim

        # Learnable queries and reference points
        self.query_embed = nn.Embedding(num_queries, feature_dim)
        self.ref_point_head = nn.Linear(feature_dim, 2)

        # Positional encoding
        self.pos_encoder = PositionEmbeddingSine(feature_dim // 2)

        # Decoder layers
        self.decoder_layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "self_attn": nn.MultiheadAttention(
                            feature_dim, num_heads, dropout=0.1, batch_first=True
                        ),
                        "cross_attn": EfficientDeformableAttention(
                            feature_dim, num_heads, num_points=4
                        ),
                        "ffn": nn.Sequential(
                            nn.Linear(feature_dim, feature_dim * 2),
                            nn.ReLU(),
                            nn.Dropout(0.1),
                            nn.Linear(feature_dim * 2, feature_dim),
                        ),
                        "norm1": nn.LayerNorm(feature_dim),
                        "norm2": nn.LayerNorm(feature_dim),
                        "norm3": nn.LayerNorm(feature_dim),
                    }
                )
                for _ in range(num_layers)
            ]
        )

        # Prediction heads (shared across layers for auxiliary losses)
        self.bbox_head = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feature_dim, feature_dim),
                    nn.ReLU(),
                    nn.Linear(feature_dim, 4),
                    nn.Sigmoid(),
                )
                for _ in range(num_layers)
            ]
        )

        self.obj_score_head = nn.ModuleList(
            [nn.Linear(feature_dim, 1) for _ in range(num_layers)]
        )

    def forward(self, feat_map):
        """
        Args:
            feat_map: [B, C, H, W]
        Returns:
            outputs: dict with intermediate predictions for auxiliary losses
        """
        B, C, H, W = feat_map.shape

        # Add positional encoding
        pos_embed = self.pos_encoder(feat_map)
        feat_map_with_pos = feat_map + pos_embed

        # Flatten: [B, H*W, C]
        feat_seq = feat_map_with_pos.flatten(2).permute(0, 2, 1)

        # Initialize queries
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # [B, N_q, C]

        # Initialize reference points from queries
        reference_points = self.ref_point_head(queries).sigmoid()  # [B, N_q, 2]

        # Store intermediate outputs for auxiliary losses
        all_bbox_preds = []
        all_obj_scores = []

        # Decoder layers
        for layer_idx, layer in enumerate(self.decoder_layers):
            # Self-attention
            queries_norm = layer["norm1"](queries)
            queries2, _ = layer["self_attn"](queries_norm, queries_norm, queries_norm)
            queries = queries + queries2

            # Cross-attention with deformable sampling
            queries_norm = layer["norm2"](queries)
            queries2 = layer["cross_attn"](
                queries_norm, reference_points, feat_seq, (H, W)
            )
            queries = queries + queries2

            # FFN
            queries_norm = layer["norm3"](queries)
            queries2 = layer["ffn"](queries_norm)
            queries = queries + queries2

            # Predict bbox and objectness
            bbox_pred = self.bbox_head[layer_idx](queries)
            obj_score = self.obj_score_head[layer_idx](queries)

            # Iterative refinement: update reference points
            if layer_idx < self.num_layers - 1:
                # Use predicted bbox center to update reference points
                new_ref_x = (bbox_pred[..., 0] + bbox_pred[..., 2]) / 2
                new_ref_y = (bbox_pred[..., 1] + bbox_pred[..., 3]) / 2
                reference_points = torch.stack([new_ref_x, new_ref_y], dim=-1).detach()

            all_bbox_preds.append(bbox_pred)
            all_obj_scores.append(obj_score)

        return {
            "pred_bboxes": all_bbox_preds[-1],  # Final prediction
            "obj_scores": all_obj_scores[-1],
            "aux_outputs": [  # Auxiliary outputs for training
                {"pred_bboxes": bbox, "obj_scores": score}
                for bbox, score in zip(all_bbox_preds[:-1], all_obj_scores[:-1])
            ],
        }


class M2DETRModel(nn.Module):
    """
    Efficient Multi-task DETR for Mammography
    - Classification + Detection
    - Optimized for few objects (max 3 lesions)
    """

    def __init__(self, backbone, feature_dim, num_classes=2, num_queries=5):
        super().__init__()
        self.backbone = backbone
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classification head
        self.classifier = nn.Linear(feature_dim, num_classes)

        # Spatial attention
        self.spatial_attention = SpatialAttention(feature_dim)

        # Efficient DETR decoder
        self.detr_decoder = EfficientDETRDecoder(
            feature_dim,
            num_queries=num_queries,
            num_heads=4,
            num_layers=2,  # Reduced from 3 to 2
        )

    def forward(self, x):
        # Backbone
        feat_map = self.backbone(x)  # [B, C, H, W]

        # Ensure 4D
        if feat_map.dim() == 2:
            feat_map = feat_map.unsqueeze(-1).unsqueeze(-1)
        elif feat_map.dim() != 4:
            raise ValueError(f"Unexpected backbone shape: {feat_map.shape}")

        # Classification branch
        cls_feat = self.global_pool(feat_map).flatten(1)
        cls_output = self.classifier(cls_feat)

        # Detection branch with attention
        attn_feat, attn_map = self.spatial_attention(feat_map)
        detr_outputs = self.detr_decoder(attn_feat)

        return {
            "cls_logits": cls_output,
            "pred_bboxes": detr_outputs["pred_bboxes"],
            "obj_scores": detr_outputs["obj_scores"],
            "aux_outputs": detr_outputs["aux_outputs"],  # For auxiliary losses
            "attn_map": attn_map,
        }


def get_m2_detr_model(model_type="resnet50", num_classes=2, num_queries=5):
    """Get efficient M2 DETR model (num_queries=5 for mammography)"""
    # Get backbone (reuse from m2_model.py)
    if model_type in ["resnet34", "resnet50", "resnet101", "resnext50", "resnet152"]:
        backbone, feature_dim = get_resnet_backbone(model_type)
        backbone = ResNetFeatureWrapper(backbone)
    elif model_type in [
        "resnest50",
        "convnextv2_tiny",
        "efficientnetv2s",
        "maxvit_tiny",
        "swinv2_tiny",
        "eva02_small",
    ]:
        backbone, feature_dim = get_timm_backbone(model_type)
        if hasattr(backbone, "forward_features"):
            backbone = TimmFeatureWrapper(backbone)
    elif model_type in ["dinov2_small", "dinov2_base", "dinov3_vit16small"]:
        backbone, feature_dim = get_dino_backbone(model_type)
        backbone = DinoFeatureWrapper(backbone)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model = M2DETRModel(backbone, feature_dim, num_classes, num_queries)
    return model
