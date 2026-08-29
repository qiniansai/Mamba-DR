import torch
import torch.nn as nn
import torch.nn.functional as F


class KnowledgeGuideLoss(nn.Module):
    """
    知识引导损失，支持基础病变concepts的知识对齐

    特性:
    1. 支持基础病变concepts的知识对齐（有病变/无病变两种嵌入）
    """

    def __init__(
        self,
        presence_embeds: torch.Tensor,
        absence_embeds: torch.Tensor,
        knowledge_embeds: torch.Tensor = None,
        lesion_names: list = None,
        eps: float = 1e-8,
        base_weight: float = 1.0,
        *args,
        **kwargs
    ) -> None:
        """
        Args:
            presence_embeds: 有病变的文本嵌入 [num_lesions, embed_dim]
            absence_embeds: 无病变的文本嵌入 [num_lesions, embed_dim]
            knowledge_embeds: 基础病变的知识嵌入 [num_lesions, embed_dim] (已废弃，保留兼容性)
            lesion_names: 基础病变名称列表
            base_weight: 基础病变concepts的损失权重
        """
        super().__init__(*args, **kwargs)

        self.loss = nn.CrossEntropyLoss(reduction="none")

        self.num_lesions = presence_embeds.shape[0]

        if knowledge_embeds is not None:
            self.knowledge_embeds_T = knowledge_embeds.T
        else:
            dummy_embeds = torch.zeros(self.num_lesions, presence_embeds.shape[1])
            self.knowledge_embeds_T = dummy_embeds.T

        self.presence_embeds_T = presence_embeds.T
        self.absence_embeds_T = absence_embeds.T

        self.lesion_names = lesion_names or []
        self.base_weight = base_weight
        self.eps = eps

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算知识引导损失

        Args:
            inputs: 基础病变concept的特征 [B, num_lesions, proj_dim]
            targets: 基础病变concept的标签 [B, num_lesions]

        Returns:
            total_loss: 加权后的总损失
        """
        total_loss = 0.0

        # 基础病变concepts的知识对齐损失
        if inputs is not None and targets is not None:
            base_loss = self._compute_alignment_loss(
                inputs, targets, self.knowledge_embeds_T
            )
            total_loss += self.base_weight * base_loss

        return total_loss

    def _compute_alignment_loss(self, inputs: torch.Tensor, targets: torch.Tensor,
                                knowledge_embeds_T: torch.Tensor) -> torch.Tensor:
        """
        计算对齐损失（支持有/无病变两种嵌入）

        Args:
            inputs: concept特征 [B, num_concepts, proj_dim]
            targets: concept标签 [B, num_concepts] (0=无病变, 1=有病变)
            knowledge_embeds_T: 知识嵌入转置 [proj_dim, num_knowledge] (已废弃)

        Returns:
            alignment_loss: 标量损失值
        """
        B, num_concepts, proj_dim = inputs.shape

        if self.presence_embeds_T is not None and self.absence_embeds_T is not None:
            presence_embeds_T = self.presence_embeds_T.to(inputs.device)
            absence_embeds_T = self.absence_embeds_T.to(inputs.device)

            total_loss = 0.0
            valid_count = 0

            for lesion_idx in range(min(num_concepts, self.num_lesions)):
                lesion_feat = inputs[:, lesion_idx, :]
                lesion_label = targets[:, lesion_idx]

                presence_embed = presence_embeds_T[:, lesion_idx]
                absence_embed = absence_embeds_T[:, lesion_idx]

                presence_sim = torch.matmul(lesion_feat, presence_embed)
                absence_sim = torch.matmul(lesion_feat, absence_embed)

                logits = torch.stack([absence_sim, presence_sim], dim=1)

                loss = F.cross_entropy(logits, lesion_label.long(), reduction='none')

                total_loss += loss.mean()
                valid_count += 1

            return total_loss / valid_count if valid_count > 0 else torch.tensor(0.0, device=inputs.device)
        else:
            scores = torch.matmul(inputs, knowledge_embeds_T.to(inputs.device))

            num_knowledge = knowledge_embeds_T.shape[1]
            gt = torch.arange(min(num_concepts, num_knowledge), dtype=torch.long, device=inputs.device)
            gt = gt.unsqueeze(0).expand(B, -1)

            valid_concepts = min(num_concepts, num_knowledge)
            scores_valid = scores[:, :valid_concepts, :]

            alignment_loss = self.loss(
                scores_valid.reshape(-1, num_knowledge),
                gt.reshape(-1)
            ).reshape(B, valid_concepts)

            targets_valid = targets[:, :valid_concepts]
            weighted_loss = alignment_loss * targets_valid

            per_sample_loss = weighted_loss.sum(dim=-1) / (targets_valid.sum(dim=-1) + self.eps)

            return per_sample_loss.mean()