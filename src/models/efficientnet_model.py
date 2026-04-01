import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class ModifiedEfficientNetB0(nn.Module):
    def __init__(self, num_outputs=1, dropout_rate=0.5):
        super(ModifiedEfficientNetB0, self).__init__()
        self.pretrained_model = EfficientNet.from_pretrained('efficientnet-b0')
        self.pretrained_model._fc = nn.Sequential(
            nn.Linear(self.pretrained_model._fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_outputs)
        )
        self.gradients = None
        self.activations = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.pretrained_model.extract_features(x)
        if x.requires_grad:
            x.register_hook(self.activations_hook)
        self.activations = x
        x = self.pretrained_model._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.pretrained_model._fc(x)
        return x

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self):
        return self.activations