"""
分层病变关注机制

根据DR分级标准，每个等级关注特定病变组合：
- Grade 0: 排除所有病变
- Grade 1: 仅关注MA（微动脉瘤）
- Grade 2: 关注MA+HE+EX（轻中度病变）
- Grade 3: 关注HE+SE+大量病变（重度非增殖特征）
- Grade 4: 关注所有病变（新生血管，增殖性特征）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LevelSpecificConceptAttention(nn.Module):
    """
    层级特定的概念关注模块

    为每个DR等级学习一个病变关注掩码，使分类头专注于该等级相关的病变。
    """

    def __init__(
        self,
        num_classes: int = 5,
        num_concepts: int = 4,
        embed_dim: int = 640,
        num_heads: int = 8,
        lesion_names: list = None,
        level_to_lesion_map: dict = None,
        use_learnable_mask: bool = True,
        mask_temperature: float = 0.1,
    ):
        """
        Args:
            num_classes: DR等级数（默认5）
            num_concepts: concept token数量（包括基础病变和extra concepts）
            embed_dim: 特征维度
            num_heads: Cross Attention头数
            lesion_names: 病变名称列表，用于构建映射
            level_to_lesion_map: 自定义的等级到病变映射
            use_learnable_mask: 是否使用可学习的关注掩码
            mask_temperature: 掩码温度参数（控制softmax的sharpness）
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_concepts = num_concepts
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_learnable_mask = use_learnable_mask
        self.mask_temperature = mask_temperature

        self.level_to_lesion_map = self._build_level_to_lesion_map(
            lesion_names, level_to_lesion_map
        )

        self.dr_level_queries = nn.Parameter(
            torch.randn(num_classes - 1, embed_dim)
        )
        nn.init.normal_(self.dr_level_queries, std=0.02)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(0.1)
        )

        # The ICDR map is a prior, not an after-the-fact visualization.  Use
        # its log-weights to initialize the trainable attention logits and
        # retain a fixed copy for an explicit regularizer.
        hard_prior = self._build_hard_mask(num_classes - 1, num_concepts)
        icdr_prior = torch.log(hard_prior)
        self.register_buffer('icdr_prior', icdr_prior)

        if use_learnable_mask:
            self.level_concept_mask = nn.Parameter(icdr_prior.clone())
        else:
            self.register_buffer(
                'level_concept_mask',
                hard_prior
            )

        self.concept_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.Tanh(),
            nn.Linear(embed_dim // 4, 1),
        )

    def _build_level_to_lesion_map(self, lesion_names, custom_map):
        """构建DR等级到病变索引的映射

        根据实际的lesion_names顺序动态构建映射，确保映射正确。
        """
        if custom_map is not None:
            return custom_map

        if lesion_names is None:
            return {
                0: [],
                1: [2],
                2: [0, 1, 2],
                3: [0, 1, 2, 3],
                4: [0, 1, 2, 3],
            }

        name_to_idx = {name: idx for idx, name in enumerate(lesion_names)}

        level_map = {
            0: [],
            1: [name_to_idx.get("MA", None)] if "MA" in name_to_idx else [],
            2: [
                name_to_idx.get("MA", None),
                name_to_idx.get("HE", None),
                name_to_idx.get("EX", None),
            ],
            3: [
                name_to_idx.get("MA", None),
                name_to_idx.get("HE", None),
                name_to_idx.get("EX", None),
                name_to_idx.get("SE", None),
            ],
            4: [
                name_to_idx.get("MA", None),
                name_to_idx.get("HE", None),
                name_to_idx.get("EX", None),
                name_to_idx.get("SE", None),
            ],
        }

        for level in level_map:
            level_map[level] = [idx for idx in level_map[level] if idx is not None]

        return level_map

    def _build_hard_mask(self, num_levels, num_concepts):
        """构建硬掩码（基于医学先验）"""
        mask = torch.zeros(num_levels, num_concepts)

        for level in range(num_levels):
            dr_grade = level + 1
            lesion_indices = self.level_to_lesion_map.get(dr_grade, [])
            for idx in lesion_indices:
                if idx < num_concepts:
                    mask[level, idx] = 1.0

        mask = mask + 0.1

        return mask

    def forward(
        self,
        concept_tokens: torch.Tensor,
    ):
        """
        Args:
            concept_tokens: [B, num_concepts, embed_dim]

        Returns:
            level_features: [B, num_classes-1, embed_dim]
            attn_weights: [B, num_classes-1, num_concepts]
            level_concept_masks: [B, num_classes-1, num_concepts]
        """
        B = concept_tokens.shape[0]

        gate_weights = self.concept_gate(concept_tokens)
        gate_weights = torch.sigmoid(gate_weights)

        gated_concepts = concept_tokens * gate_weights

        level_masks = self.level_concept_mask.unsqueeze(0).expand(B, -1, -1)

        if self.use_learnable_mask:
            level_masks = F.softmax(level_masks / self.mask_temperature, dim=-1)
        else:
            level_masks = level_masks / (level_masks.sum(dim=-1, keepdim=True) + 1e-8)

        masked_concepts = torch.einsum('blc,bce->ble', level_masks, gated_concepts)

        queries = self.dr_level_queries.unsqueeze(0).expand(B, -1, -1)

        attn_output, attn_weights = self.cross_attn(
            query=queries,
            key=gated_concepts,
            value=gated_concepts,
        )

        combined_features = attn_output + 0.5 * masked_concepts

        combined_features = self.norm1(queries + combined_features)

        ffn_output = self.ffn(combined_features)
        level_features = self.norm2(combined_features + ffn_output)

        return level_features, attn_weights, level_masks

    def mask_prior_loss(self) -> torch.Tensor:
        """KL penalty that keeps learned level masks close to the ICDR prior."""
        if not self.use_learnable_mask:
            return self.icdr_prior.new_zeros(())

        learned_log_probs = F.log_softmax(
            self.level_concept_mask / self.mask_temperature, dim=-1
        )
        prior_probs = F.softmax(self.icdr_prior / self.mask_temperature, dim=-1)
        return F.kl_div(learned_log_probs, prior_probs, reduction='batchmean')

    def get_level_concept_importance(self) -> dict:
        """
        获取每个DR等级对各个concept的关注度（用于可视化）

        Returns:
            importance_dict: {level_name: {concept_name: importance_score}}
        """
        if self.use_learnable_mask:
            masks = F.softmax(self.level_concept_mask / self.mask_temperature, dim=-1)
        else:
            masks = self.level_concept_mask

        masks = masks.detach().cpu().numpy()

        level_names = [f"≥{i+1}级" for i in range(self.num_classes - 1)]

        importance_dict = {}
        for i, level_name in enumerate(level_names):
            importance_dict[level_name] = {
                f"concept_{j}": float(masks[i, j])
                for j in range(self.num_concepts)
            }

        return importance_dict
