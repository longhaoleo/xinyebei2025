import torch
import torch.nn as nn

class DualStreamModel(nn.Module):
    def __init__(self, backbone_spatial, backbone_freq, num_classes=1):
        super().__init__()
        self.backbone_spatial = backbone_spatial
        self.backbone_freq = backbone_freq

        # Assuming both backbones output flat feature vectors
        feature_dim_spatial = backbone_spatial.fc.in_features
        feature_dim_freq = backbone_freq.fc.in_features
        self.backbone_spatial.fc = nn.Identity()
        self.backbone_freq.fc = nn.Identity()

        self.fusion_dim = feature_dim_spatial + feature_dim_freq
        self.classifier = nn.Linear(self.fusion_dim, num_classes)

    def forward(self, x_spatial, x_freq):
        feat_spatial = self.backbone_spatial(x_spatial)
        feat_freq = self.backbone_freq(x_freq)

        # Fusion (e.g., concatenation)
        fused_features = torch.cat((feat_spatial, feat_freq), dim=1)

        output = self.classifier(fused_features)
        return output

def get_freq_domain_input(x):
    """Converts a batch of images to its frequency domain representation."""
    # Example: using FFT
    x_fft = torch.fft.fft2(x, dim=(-2, -1))
    x_fft_shifted = torch.fft.fftshift(x_fft, dim=(-2, -1))
    return torch.abs(x_fft_shifted)