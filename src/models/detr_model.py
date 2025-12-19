import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .m2_model import (
    ResNetFeatureWrapper,
    TimmFeatureWrapper,
    DinoFeatureWrapper,
)
from .backbone import get_resnet_backbone, get_timm_backbone, get_dino_backbone


class MultiScaleSpatialAttention(nn.Module):
    """
    Improved Multi-Scale Spatial Attention for mammography detection
    - Uses multi-scale features (different receptive fields)
    - Channel attention + spatial attention
    - Better for detecting lesions of different sizes
    """

    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.in_channels = in_channels

        # Channel Attention (squeeze-excitation style)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )

        # Multi-scale spatial features
        # Use dilated convs to capture different scales without increasing params much
        self.spatial_branch1 = nn.Conv2d(2, 1, kernel_size=3, padding=1, dilation=1)
        self.spatial_branch2 = nn.Conv2d(2, 1, kernel_size=3, padding=2, dilation=2)
        self.spatial_branch3 = nn.Conv2d(2, 1, kernel_size=3, padding=4, dilation=4)

        # Fusion layer for multi-scale spatial features
        self.spatial_fusion = nn.Conv2d(3, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W] - attended features
            attn_map: [B, 1, H, W] - final attention map
        """
        B, C, H, W = x.shape

        # 1. Channel Attention
        channel_weight = self.channel_attn(x)  # [B, C, 1, 1]
        x_channel = x * channel_weight  # [B, C, H, W]

        # 2. Multi-scale Spatial Attention
        # Aggregate spatial statistics
        avg_out = torch.mean(x_channel, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x_channel, dim=1, keepdim=True)  # [B, 1, H, W]
        spatial_input = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]

        # Multi-scale branches
        s1 = self.spatial_branch1(spatial_input)  # [B, 1, H, W] - local
        s2 = self.spatial_branch2(spatial_input)  # [B, 1, H, W] - medium
        s3 = self.spatial_branch3(spatial_input)  # [B, 1, H, W] - global

        # Fuse multi-scale features
        spatial_cat = torch.cat([s1, s2, s3], dim=1)  # [B, 3, H, W]
        spatial_attn = self.sigmoid(self.spatial_fusion(spatial_cat))  # [B, 1, H, W]

        # 3. Apply spatial attention
        out = x_channel * spatial_attn

        return out, spatial_attn


class PositionEmbeddingSine(nn.Module):
    """
    Sine-cosine positional encoding (from DETR)
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

    def __init__(self, embed_dim, num_heads=4, num_points=4):
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

    def forward(
        self,
        query,
        reference_points,
        value,
        spatial_shapes,
        return_attention_weights=False,
    ):
        """
        Args:
            query: [B, N_q, C]
            reference_points: [B, N_q, 2] in [0, 1]
            value: [B, H*W, C]
            spatial_shapes: (H, W)
            return_attention_weights: bool, whether to return attention maps
        Returns:
            output: [B, N_q, C]
            attn_maps: [B, H, W] global attention map (max-pooled over queries) if return_attention_weights else None
        """
        B, N_q, C = query.shape
        H, W = spatial_shapes
        device = query.device

        # Generate offsets and weights
        offsets = self.sampling_offsets(query).view(
            B, N_q, self.num_heads, self.num_points, 2
        )
        attn_weights = self.attention_weights(query).view(
            B, N_q, self.num_heads, self.num_points
        )
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Normalize offsets to relative scale
        offsets_x = offsets[..., 0] / float(W) * 0.1
        offsets_y = offsets[..., 1] / float(H) * 0.1
        offsets = torch.stack([offsets_x, offsets_y], dim=-1)

        # Sampling locations
        sampling_locations = reference_points[:, :, None, None, :] + offsets
        sampling_locations = sampling_locations.clamp(0, 1)

        # Project value and reshape for grid_sample
        value_projected = self.value_proj(value)  # [B, H*W, C]
        value_spatial = value_projected.view(B, H, W, self.num_heads, self.head_dim)
        value_spatial = value_spatial.permute(0, 3, 4, 1, 2)  # [B, num_heads, head_dim, H, W]

        # Prepare for grid_sample: reshape to [B*num_heads, head_dim, H, W]
        value_for_sample = value_spatial.reshape(B * self.num_heads, self.head_dim, H, W)

        # Prepare grid: [B, N_q, num_heads, num_points, 2] -> [B*num_heads, N_q*num_points, 1, 2]
        grid = sampling_locations.permute(0, 2, 1, 3, 4).reshape(
            B * self.num_heads, N_q * self.num_points, 1, 2
        )
        # Normalize to [-1, 1] for grid_sample
        grid = grid * 2 - 1

        # Sample features: [B*num_heads, head_dim, N_q*num_points, 1]
        sampled_feats = F.grid_sample(
            value_for_sample,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )
        
        # Reshape: [B, num_heads, head_dim, N_q, num_points]
        sampled_feats = sampled_feats.squeeze(-1).view(
            B, self.num_heads, self.head_dim, N_q, self.num_points
        ).permute(0, 3, 1, 4, 2)  # [B, N_q, num_heads, num_points, head_dim]

        # Apply attention weights and aggregate
        attn_weights_expanded = attn_weights.unsqueeze(-1)  # [B, N_q, num_heads, num_points, 1]
        output = (sampled_feats * attn_weights_expanded).sum(dim=3)  # [B, N_q, num_heads, head_dim]
        output = output.flatten(2)  # [B, N_q, C]
        output = self.output_proj(output)

        if return_attention_weights:
            # Create per-query attention maps
            per_query_maps = torch.zeros((B, N_q, H, W), device=device)
            
            # Accumulate attention weights at sampling locations
            for b in range(B):
                for q in range(N_q):
                    for h in range(self.num_heads):
                        for p in range(self.num_points):
                            # Get attention weight
                            weight = attn_weights[b, q, h, p].item()
                            
                            # Get sampling location in pixel coordinates
                            x_norm = sampling_locations[b, q, h, p, 0].item()
                            y_norm = sampling_locations[b, q, h, p, 1].item()
                            
                            x = int(x_norm * (W - 1))
                            y = int(y_norm * (H - 1))
                            
                            # Clamp to valid range
                            x = max(0, min(W - 1, x))
                            y = max(0, min(H - 1, y))
                            
                            # Accumulate
                            per_query_maps[b, q, y, x] += weight

            # Normalize per-query maps
            per_query_maps = per_query_maps / (self.num_heads * self.num_points + 1e-6)

            # Max pool over queries to get global attention map
            attn_maps = per_query_maps.max(dim=1).values  # [B, H, W]

            return output, attn_maps
        else:
            return output, None


class EfficientDeformableAttention_ori(nn.Module):
    """
    Efficient Deformable Attention with reduced complexity
    """

    def __init__(self, embed_dim, num_heads=4, num_points=4):
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

        # REMOVED: buffer approach doesn't work with DataParallel

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

    def forward(
        self,
        query,
        reference_points,
        value,
        spatial_shapes,
        return_attention_weights=False,
    ):
        """
        Args:
            query: [B, N_q, C]
            reference_points: [B, N_q, 2] in [0, 1]
            value: [B, H*W, C]
            spatial_shapes: (H, W)
            return_attention_weights: bool, whether to return attention maps
        Returns:
            output: [B, N_q, C]
            attn_maps: [B, H, W] global attention map (averaged over queries) if return_attention_weights else None
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

        # FIXED: Direct division to avoid creating tensor (multi-GPU safe)
        offsets_x = offsets[..., 0] / float(W) * 0.1
        offsets_y = offsets[..., 1] / float(H) * 0.1
        offsets = torch.stack([offsets_x, offsets_y], dim=-1)

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

        # UPDATED: Create global attention map (averaged over all queries)
        attn_maps_global = None
        if return_attention_weights:
            attn_maps_global = torch.zeros(B, H, W, device=query.device)

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

            # UPDATED: Accumulate to global map (all queries combined)
            if return_attention_weights:
                for b in range(B):
                    for q in range(N_q):
                        # Get sampling locations for this query
                        locs = sampling_locations[b, q, head]  # [num_points, 2]
                        weights = attn_weights[b, q, head]  # [num_points]

                        # Convert normalized coords to pixel coords
                        x_coords = (locs[:, 0] * (W - 1)).long().clamp(0, W - 1)
                        y_coords = (locs[:, 1] * (H - 1)).long().clamp(0, H - 1)

                        # Accumulate weights at sampling locations
                        for i in range(self.num_points):
                            attn_maps_global[b, y_coords[i], x_coords[i]] += weights[i]

        output = torch.cat(sampled_features, dim=-1)  # [B, N_q, C]
        output = self.output_proj(output)

        if return_attention_weights:
            # UPDATED: Normalize by total number of accumulations
            # Average over all queries and heads
            attn_maps_global = attn_maps_global / (
                N_q * self.num_heads * self.num_points + 1e-6
            )
            return output, attn_maps_global  # [B, H, W]
        else:
            return output


class EfficientDETRDecoder(nn.Module):
    """
    Efficient DETR Decoder with Denoising Queries:
    - Simple denoising queries from GT boxes
    - IoU-weighted objectness loss
    - Reduced layers for efficiency
    """

    def __init__(self, feature_dim, num_queries=5, num_heads=2, num_layers=1):
        super().__init__()
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.feature_dim = feature_dim

        # Learnable queries and reference points
        self.query_embed = nn.Embedding(num_queries, feature_dim)
        self.ref_point_head = nn.Linear(feature_dim, 2)

        # Box embedding for denoising queries
        self.box_embed = nn.Sequential(
            nn.Linear(4, feature_dim), nn.ReLU(), nn.Linear(feature_dim, feature_dim)
        )

        # Positional encoding
        self.pos_encoder = PositionEmbeddingSine(feature_dim // 2)

        # Decoder layers (reduced complexity)
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

        # Prediction heads (shared for efficiency)
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

    def generate_noisy_boxes(self, gt_boxes, noise_scale=0.2):
        """
        Generate noisy boxes from GT for denoising queries
        Args:
            gt_boxes: [B, N_gt, 4] normalized boxes
            noise_scale: noise level (default 0.2)
        Returns:
            noisy_boxes: [B, N_gt, 4]
        """
        noise = torch.randn_like(gt_boxes) * noise_scale
        noisy_boxes = (gt_boxes + noise).clamp(0, 1)
        return noisy_boxes

    def forward(self, feat_map, gt_boxes=None, return_attention_maps=False):
        """
        Args:
            feat_map: [B, C, H, W]
            gt_boxes: [B, N_gt, 4] during training, None during inference
            return_attention_maps: bool, whether to return cross-attention maps
        Returns:
            outputs: dict with predictions and global attention map
        """
        B, C, H, W = feat_map.shape

        # Add positional encoding
        pos_embed = self.pos_encoder(feat_map)
        feat_map_with_pos = feat_map + pos_embed
        feat_seq = feat_map_with_pos.flatten(2).permute(0, 2, 1)

        # Initialize learnable queries
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # [B, N_q, C]
        reference_points = self.ref_point_head(queries).sigmoid()  # [B, N_q, 2]

        # Generate denoising queries if training
        dn_queries = None
        dn_ref_points = None
        num_dn = 0

        if self.training and gt_boxes is not None:
            # Generate noisy boxes
            noisy_boxes = self.generate_noisy_boxes(gt_boxes)
            num_dn = noisy_boxes.shape[1]

            # Encode noisy boxes to queries
            dn_queries = self.box_embed(noisy_boxes)  # [B, N_gt, C]

            # Reference points from noisy box centers
            dn_ref_x = (noisy_boxes[..., 0] + noisy_boxes[..., 2]) / 2
            dn_ref_y = (noisy_boxes[..., 1] + noisy_boxes[..., 3]) / 2
            dn_ref_points = torch.stack([dn_ref_x, dn_ref_y], dim=-1)  # [B, N_gt, 2]

            # Concatenate denoising queries with learnable queries
            queries = torch.cat([dn_queries, queries], dim=1)  # [B, N_dn+N_q, C]
            reference_points = torch.cat([dn_ref_points, reference_points], dim=1)

        all_bbox_preds = []
        all_obj_scores = []
        cross_attn_maps = None

        # Decoder layers
        for layer_idx, layer in enumerate(self.decoder_layers):
            # Self-attention
            queries_norm = layer["norm1"](queries)
            queries2, _ = layer["self_attn"](queries_norm, queries_norm, queries_norm)
            queries = queries + queries2

            # Cross-attention with attention map extraction
            queries_norm = layer["norm2"](queries)

            # Get attention maps from last layer only
            return_attn = return_attention_maps and (layer_idx == self.num_layers - 1)

            if return_attn:
                queries2, attn_maps = layer["cross_attn"](
                    queries_norm,
                    reference_points,
                    feat_seq,
                    (H, W),
                    return_attention_weights=True,
                )
                cross_attn_maps = attn_maps  # [B, H, W] global map
            else:
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

            # Iterative refinement
            if layer_idx < self.num_layers - 1:
                new_ref_x = (bbox_pred[..., 0] + bbox_pred[..., 2]) / 2
                new_ref_y = (bbox_pred[..., 1] + bbox_pred[..., 3]) / 2
                reference_points = torch.stack([new_ref_x, new_ref_y], dim=-1).detach()

            all_bbox_preds.append(bbox_pred)
            all_obj_scores.append(obj_score)

        # Split denoising and matching outputs
        if num_dn > 0:
            dn_bbox_preds = [pred[:, :num_dn] for pred in all_bbox_preds]
            dn_obj_scores = [score[:, :num_dn] for score in all_obj_scores]

            match_bbox_preds = [pred[:, num_dn:] for pred in all_bbox_preds]
            match_obj_scores = [score[:, num_dn:] for score in all_obj_scores]

            return {
                "pred_bboxes": match_bbox_preds[-1],
                "obj_scores": match_obj_scores[-1],
                "aux_outputs": [
                    {"pred_bboxes": bbox, "obj_scores": score}
                    for bbox, score in zip(match_bbox_preds[:-1], match_obj_scores[:-1])
                ],
                "dn_outputs": [
                    {"pred_bboxes": bbox, "obj_scores": score, "gt_boxes": gt_boxes}
                    for bbox, score in zip(dn_bbox_preds, dn_obj_scores)
                ],
                "cross_attn_maps": cross_attn_maps,  # [B, H, W]
            }
        else:
            return {
                "pred_bboxes": all_bbox_preds[-1],
                "obj_scores": all_obj_scores[-1],
                "aux_outputs": [
                    {"pred_bboxes": bbox, "obj_scores": score}
                    for bbox, score in zip(all_bbox_preds[:-1], all_obj_scores[:-1])
                ],
                "dn_outputs": [],
                "cross_attn_maps": cross_attn_maps,  # [B, H, W]
            }


class M2DETRModel(nn.Module):
    """
    Efficient Multi-task DETR for Mammography
    """

    def __init__(
        self, backbone, feature_dim, num_classes=2, num_queries=5, reduced_dim=256
    ):
        super().__init__()
        self.backbone = backbone
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        if feature_dim > reduced_dim:
            self.feature_proj = nn.Conv2d(feature_dim, reduced_dim, kernel_size=1)
            feature_dim = reduced_dim
        else:
            self.feature_proj = None

        self.classifier = nn.Linear(feature_dim, num_classes)

        # UPDATED: Use improved multi-scale spatial attention
        self.spatial_attention = MultiScaleSpatialAttention(feature_dim, reduction=8)

        self.detr_decoder = EfficientDETRDecoder(
            feature_dim,
            num_queries=num_queries,
            num_heads=1,
            num_layers=1,
        )

    def forward(self, x, gt_boxes=None, return_attention_maps=False):
        """
        Args:
            x: input images
            gt_boxes: [B, N_gt, 4] ground truth boxes (training only)
            return_attention_maps: bool, whether to return decoder attention maps
        Returns:
            dict with cls_logits, pred_bboxes, obj_scores, attn_maps [B, H, W]
        """
        # Backbone
        feat_map = self.backbone(x)

        # Ensure 4D
        if feat_map.dim() == 2:
            feat_map = feat_map.unsqueeze(-1).unsqueeze(-1)
        elif feat_map.dim() != 4:
            raise ValueError(f"Unexpected backbone shape: {feat_map.shape}")

        # ADDED: Project to lower dimension
        if self.feature_proj is not None:
            feat_map = self.feature_proj(feat_map)

        cls_feat = self.global_pool(feat_map).flatten(1)
        cls_output = self.classifier(cls_feat)

        # Get spatial attention (keep for feature enhancement)
        attn_feat, spatial_attn_map = self.spatial_attention(feat_map)

        # Get decoder outputs with optional cross-attention maps
        detr_outputs = self.detr_decoder(attn_feat, gt_boxes, return_attention_maps)

        return {
            "cls_logits": cls_output,
            "pred_bboxes": detr_outputs["pred_bboxes"],
            "obj_scores": detr_outputs["obj_scores"],
            "aux_outputs": detr_outputs["aux_outputs"],
            "dn_outputs": detr_outputs.get("dn_outputs", []),
            "attn_maps": detr_outputs.get(
                "cross_attn_maps"
            ),  # UPDATED: [B, H, W] global attention map
            "spatial_attn": spatial_attn_map,  # [B, 1, H, W]
        }


def get_detr_model(model_type="resnet50", num_classes=2, num_queries=5):
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
        "maxvit_tiny512",
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

    model = M2DETRModel(
        backbone, feature_dim, num_classes, num_queries, reduced_dim=256
    )
    return model
