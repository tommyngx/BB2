import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                focal_loss = self.alpha * focal_loss
            else:
                alpha_t = self.alpha[targets]
                focal_loss = alpha_t * focal_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30.0):
        super().__init__()
        # Kiểm tra đầu vào
        assert len(cls_num_list) > 0, "cls_num_list must not be empty"
        assert all(n > 0 for n in cls_num_list), (
            "cls_num_list must contain positive values"
        )

        self.s = float(s)

        # Tính margin: m_j ∝ n_j^{-1/4}, chuẩn hóa để max(m_j) = max_m
        n = torch.tensor(cls_num_list, dtype=torch.float32)
        m_list = 1.0 / torch.sqrt(torch.sqrt(n))
        m_list = m_list * (
            float(max_m) / m_list.max().clamp_min(1e-6)
        )  # Tăng clamp_min
        self.register_buffer("m_list", m_list)

        # Xử lý weight
        if weight is not None:
            w = torch.as_tensor(weight, dtype=torch.float32)
            assert w.numel() == len(cls_num_list), (
                "weight must have same length as cls_num_list"
            )
            assert (w >= 0).all(), "weight must contain non-negative values"
            self.register_buffer("weight", w)
        else:
            self.weight = None

    def forward(self, logits, targets):
        # Kiểm tra đầu vào
        assert torch.is_tensor(logits) and torch.is_tensor(targets), (
            "logits and targets must be tensors"
        )
        B, C = logits.shape
        assert C == self.m_list.numel(), (
            "logits must have shape [batch_size, num_classes]"
        )
        assert targets.shape == (B,), "targets must have shape [batch_size]"
        if targets.dtype != torch.long:
            assert targets.dtype in (torch.int, torch.long), (
                "targets must be integer type"
            )
            targets = targets.long()
        assert targets.min() >= 0 and targets.max() < C, "targets out of range"

        # Margin cho từng mẫu, khớp dtype với logits
        batch_m = self.m_list[targets].to(dtype=logits.dtype)  # [B]

        # Trừ margin vào logit của lớp đúng
        logits_m = logits.clone()
        row_idx = torch.arange(B, device=logits.device)
        logits_m[row_idx, targets] -= batch_m

        # Scale và tính cross entropy
        logits_s = self.s * logits_m
        weight = (
            None if self.weight is None else self.weight.to(logits.device, logits.dtype)
        )
        return F.cross_entropy(logits_s, targets, weight=weight)
