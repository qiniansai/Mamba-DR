"""
Text Encoder Module
Based on Bio_ClinicalBERT for medical image-text contrastive learning.
"""

import os
import torch
from transformers import AutoModel, AutoTokenizer, logging

logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"


class ProjectionLayer(torch.nn.Module):
    def __init__(self, layer, projection=True, norm=True):
        super().__init__()

        self.apply_projection = projection
        self.norm_modality = bool(projection * norm)
        self.norm_projection = norm
        self.projection = layer

    def forward(self, x):

        if self.norm_modality:
            x = x / x.norm(dim=-1, keepdim=True)

        if self.apply_projection:
            x = self.projection(x)
            if self.norm_projection:
                x = x / x.norm(dim=-1, keepdim=True)

        return x


class TextModel(torch.nn.Module):
    """
    Text encoder based on pretrained BERT model.
    
    Args:
        bert_type: HuggingFace model identifier (default: 'emilyalsentzer/Bio_ClinicalBERT')
        proj_dim: Output projection dimension (default: 512)
        proj_bias: Whether to use bias in projection layer (default: False)
        projection: Whether to apply projection (default: True)
        norm: Whether to normalize features (default: True)
    """
    
    def __init__(self, bert_type='emilyalsentzer/Bio_ClinicalBERT', proj_dim=512, proj_bias=False, 
                 projection=True, norm=True):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(bert_type)
        self.tokenizer.model_max_length = 77

        self.model = AutoModel.from_pretrained(bert_type, output_hidden_states=True)

        self.projection_head_text = ProjectionLayer(
            layer=torch.nn.Linear(768, proj_dim, bias=proj_bias),
            projection=projection, 
            norm=norm
        )

    def tokenize(self, prompts_list):
        """Tokenize a list of text prompts."""
        text_tokens = self.tokenizer(prompts_list, truncation=True, padding=True, return_tensors='pt')
        return text_tokens

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through text encoder.
        
        Args:
            input_ids: Token IDs from tokenizer
            attention_mask: Attention mask from tokenizer
            
        Returns:
            Projected text embeddings of shape (batch_size, proj_dim)
        """
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        last_hidden_states = torch.stack([
            output['hidden_states'][1], 
            output['hidden_states'][2],
            output['hidden_states'][-1]
        ])
        embed = last_hidden_states.permute(1, 0, 2, 3).mean(2).mean(1)

        embed = self.projection_head_text(embed)
        return embed
    
    def encode_text(self, prompts_list, device='cuda'):
        """
        Convenient method to encode text prompts.
        
        Args:
            prompts_list: List of text strings to encode
            device: Device to run on ('cuda' or 'cpu')
            
        Returns:
            Text embeddings tensor
        """
        self.eval()
        with torch.no_grad():
            text_tokens = self.tokenize(prompts_list)
            input_ids = text_tokens["input_ids"].to(device)
            attention_mask = text_tokens["attention_mask"].to(device)
            embeddings = self.forward(input_ids, attention_mask)
        return embeddings


class TextEncoderWithPrompts(torch.nn.Module):
    """
    Text encoder with prompt template support for zero-shot classification.
    
    Args:
        bert_type: HuggingFace model identifier
        proj_dim: Output projection dimension
        caption: Prompt template with [CLS] placeholder (default: "A fundus photograph of [CLS]")
        **kwargs: Additional arguments passed to TextModel
    """
    
    def __init__(self, bert_type='emilyalsentzer/Bio_ClinicalBERT', proj_dim=512, 
                 caption="A fundus photograph of [CLS]", **kwargs):
        super().__init__()
        
        self.text_model = TextModel(bert_type=bert_type, proj_dim=proj_dim, **kwargs)
        self.caption = caption
        
    def compute_class_embeddings(self, categories, domain_knowledge=None, device='cuda'):
        """
        Compute text embeddings for a list of class categories.
        
        Args:
            categories: List of class names/labels
            domain_knowledge: Optional dict mapping class names to expert descriptions
            device: Device to run on
            
        Returns:
            dict: {class_name: embedding}
            tensor: Stacked embeddings tensor
        """
        self.eval()
        text_embeds_dict = {}
        
        for class_name in categories:
            if domain_knowledge and class_name in domain_knowledge:
                descriptions = domain_knowledge[class_name]
                if class_name not in descriptions:
                    descriptions.append(class_name)
            else:
                descriptions = [class_name]
            
            with torch.no_grad():
                prompts = [self.caption.replace("[CLS]", desc) for desc in descriptions]
                text_tokens = self.text_model.tokenize(prompts)
                input_ids = text_tokens["input_ids"].to(device).to(torch.long)
                attention_mask = text_tokens["attention_mask"].to(device).to(torch.long)
                
                text_embeds = self.text_model(input_ids, attention_mask)
            
            text_embeds_dict[class_name] = text_embeds.mean(0).unsqueeze(0)
        
        text_embeds = torch.cat(list(text_embeds_dict.values()))
        
        return text_embeds_dict, text_embeds
    
    def forward(self, input_ids, attention_mask):
        return self.text_model(input_ids, attention_mask)
    
    def tokenize(self, prompts_list):
        return self.text_model.tokenize(prompts_list)
    
    def compute_presence_absence_embeddings(self, categories, domain_knowledge=None, device='cuda'):
        """
        Compute text embeddings for presence/absence of each lesion type.
        
        For each lesion, generates two embeddings:
        - "has lesion" embedding (e.g., "has microaneurysms")
        - "no lesion" embedding (e.g., "no microaneurysms")
        
        Args:
            categories: List of lesion names
            domain_knowledge: Optional dict mapping lesion names to expert descriptions
            device: Device to run on
            
        Returns:
            presence_embeds: Tensor of "has lesion" embeddings [num_lesions, proj_dim]
            absence_embeds: Tensor of "no lesion" embeddings [num_lesions, proj_dim]
        """
        self.eval()
        presence_embeds_list = []
        absence_embeds_list = []
        
        for lesion_name in categories:
            with torch.no_grad():
                presence_prompts = []
                absence_prompts = []
                
                if domain_knowledge and lesion_name in domain_knowledge:
                    descriptions = domain_knowledge[lesion_name]
                    if lesion_name not in descriptions:
                        descriptions.append(lesion_name)
                    
                    for desc in descriptions:
                        presence_prompts.append(self.caption.replace("[CLS]", f"has {desc}"))
                        absence_prompts.append(self.caption.replace("[CLS]", f"no {desc}"))
                else:
                    presence_prompts.append(self.caption.replace("[CLS]", f"has {lesion_name}"))
                    absence_prompts.append(self.caption.replace("[CLS]", f"no {lesion_name}"))
                
                presence_tokens = self.text_model.tokenize(presence_prompts)
                presence_input_ids = presence_tokens["input_ids"].to(device).to(torch.long)
                presence_attention_mask = presence_tokens["attention_mask"].to(device).to(torch.long)
                presence_embeds = self.text_model(presence_input_ids, presence_attention_mask)
                
                absence_tokens = self.text_model.tokenize(absence_prompts)
                absence_input_ids = absence_tokens["input_ids"].to(device).to(torch.long)
                absence_attention_mask = absence_tokens["attention_mask"].to(device).to(torch.long)
                absence_embeds = self.text_model(absence_input_ids, absence_attention_mask)
            
            presence_embeds_list.append(presence_embeds.mean(0).unsqueeze(0))
            absence_embeds_list.append(absence_embeds.mean(0).unsqueeze(0))
        
        presence_embeds = torch.cat(presence_embeds_list)
        absence_embeds = torch.cat(absence_embeds_list)
        
        return presence_embeds, absence_embeds
