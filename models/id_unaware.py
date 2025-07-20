import torch
import torch.nn as nn
from torch.autograd import Function

class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class IDUnawareModel(nn.Module):
    def __init__(self, backbone, num_classes=1, num_identities=1000):
        super().__init__()
        self.backbone = backbone
        # Assuming backbone outputs a flat feature vector
        feature_dim = backbone.fc.in_features
        self.backbone.fc = nn.Identity() # Remove original classifier

        self.classifier = nn.Linear(feature_dim, num_classes)
        self.identity_classifier = nn.Linear(feature_dim, num_identities)

    def forward(self, x, alpha=1.0):
        features = self.backbone(x)
        
        # Classification branch
        cls_output = self.classifier(features)

        # Identity branch with GRL
        reversed_features = GradientReversalLayer.apply(features, alpha)
        id_output = self.identity_classifier(reversed_features)

        return cls_output, id_output, features