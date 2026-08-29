from enum import Enum
from .mambavision_concept import (
    MambaVisionConcept,
    mambavision_tiny_concept,
    mambavision_tiny2_concept,
    mambavision_small_concept,
    mambavision_base_concept,
    mambavision_base_21k_concept,
    mambavision_large_concept,
    mambavision_large_21k_concept,
    mambavision_large2_concept,
    mambavision_large2_512_21k_concept,
    mambavision_large3_256_21k_concept,
    mambavision_large3_512_21k_concept,
    BACKBONE_CONFIGS,
    SVMIntervention,
)


class EncoderType(Enum):
    MAMBAVISION_TINY_CONCEPT = "mambavision_tiny_concept"
    MAMBAVISION_TINY2_CONCEPT = "mambavision_tiny2_concept"
    MAMBAVISION_SMALL_CONCEPT = "mambavision_small_concept"
    MAMBAVISION_BASE_CONCEPT = "mambavision_base_concept"
    MAMBAVISION_BASE_21K_CONCEPT = "mambavision_base_21k_concept"
    MAMBAVISION_LARGE_CONCEPT = "mambavision_large_concept"
    MAMBAVISION_LARGE_21K_CONCEPT = "mambavision_large_21k_concept"
    MAMBAVISION_LARGE2_CONCEPT = "mambavision_large2_concept"
    MAMBAVISION_LARGE2_512_21K_CONCEPT = "mambavision_large2_512_21k_concept"
    MAMBAVISION_LARGE3_256_21K_CONCEPT = "mambavision_large3_256_21k_concept"
    MAMBAVISION_LARGE3_512_21K_CONCEPT = "mambavision_large3_512_21k_concept"

    @property
    def load_func(self):
        return {
            EncoderType.MAMBAVISION_TINY_CONCEPT: mambavision_tiny_concept,
            EncoderType.MAMBAVISION_TINY2_CONCEPT: mambavision_tiny2_concept,
            EncoderType.MAMBAVISION_SMALL_CONCEPT: mambavision_small_concept,
            EncoderType.MAMBAVISION_BASE_CONCEPT: mambavision_base_concept,
            EncoderType.MAMBAVISION_BASE_21K_CONCEPT: mambavision_base_21k_concept,
            EncoderType.MAMBAVISION_LARGE_CONCEPT: mambavision_large_concept,
            EncoderType.MAMBAVISION_LARGE_21K_CONCEPT: mambavision_large_21k_concept,
            EncoderType.MAMBAVISION_LARGE2_CONCEPT: mambavision_large2_concept,
            EncoderType.MAMBAVISION_LARGE2_512_21K_CONCEPT: mambavision_large2_512_21k_concept,
            EncoderType.MAMBAVISION_LARGE3_256_21K_CONCEPT: mambavision_large3_256_21k_concept,
            EncoderType.MAMBAVISION_LARGE3_512_21K_CONCEPT: mambavision_large3_512_21k_concept,
        }[self]


def load_encoder(
    name: str,
    num_classes: int,
    num_lesions: int,
    img_size: int,
    pretrained: bool = True,
    use_ordinal_head: bool = True,
    ordinal_num_heads: int = 8,
    use_level_specific_attention: bool = True,
    lesion_names: list = None,
    intervention_strength: float = 1.0,
    intervention_temperature: float = 2.0,
) -> MambaVisionConcept:
    """
    加载编码器

    Args:
        name: 编码器名称
        num_classes: DR类别数量
        num_lesions: 病变数量
        img_size: 图像大小
        pretrained: 是否使用预训练权重
        use_ordinal_head: 是否使用序数分类头
        ordinal_num_heads: Cross Attention头数
        use_level_specific_attention: 是否使用分层病变关注机制
        lesion_names: 病变名称列表

    Returns:
        编码器实例
    """
    try:
        model = EncoderType(name).load_func(
            num_lesions=num_lesions,
            num_classes=num_classes,
            img_size=img_size,
            pretrained=pretrained,
            use_ordinal_head=use_ordinal_head,
            ordinal_num_heads=ordinal_num_heads,
            use_level_specific_attention=use_level_specific_attention,
            lesion_names=lesion_names,
            intervention_strength=intervention_strength,
            intervention_temperature=intervention_temperature,
        )
        return model
    except ValueError as e:
        raise ValueError(f"Unsupported model: {name}. Available options: {[et.value for et in EncoderType]}") from e
