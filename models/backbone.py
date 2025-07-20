import torch.nn as nn
from torchvision import models

# You might need to install timm for more models like EfficientNet
# import timm

def get_backbone(name: str, pretrained: bool = True):
    """Returns a pretrained backbone model."""
    if name == 'resnet50':
        return models.resnet50(pretrained=pretrained)
    elif name == 'efficientnet-b4':
        # Use timm for efficientnet
        # return timm.create_model('efficientnet_b4', pretrained=pretrained)
        print("Using a placeholder for EfficientNet. Please install 'timm' for the actual model.")
        return models.resnet18(pretrained=pretrained) # Placeholder
    elif name == 'vit':
        # Use timm for ViT
        # return timm.create_model('vit_base_patch16_224', pretrained=pretrained)
        print("Using a placeholder for ViT. Please install 'timm' for the actual model.")
        return models.resnet18(pretrained=pretrained) # Placeholder
    else:
        raise ValueError(f"Backbone '{name}' not supported.")