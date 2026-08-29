"""
Text Encoder Module

A standalone text encoder module based on Bio_ClinicalBERT
for medical image-text contrastive learning.

Usage:
    from text_encoder import TextModel, TextEncoderWithPrompts
    
    # Basic usage
    text_model = TextModel(proj_dim=512)
    embeddings = text_model.encode_text(["diabetic retinopathy", "glaucoma"])
    
    # With prompt templates for zero-shot classification
    encoder = TextEncoderWithPrompts(caption="A fundus photograph of [CLS]")
    class_embeds_dict, class_embeds = encoder.compute_class_embeddings(
        categories=["normal", "glaucoma", "diabetic retinopathy"]
    )
"""

from .text_encoder import TextModel, TextEncoderWithPrompts, ProjectionLayer
from .domain_knowledge import definitions, abbreviations, lesion_abbreviations

__all__ = [
    'TextModel',
    'TextEncoderWithPrompts', 
    'ProjectionLayer',
    'definitions',
    'abbreviations',
    'lesion_abbreviations'
]
