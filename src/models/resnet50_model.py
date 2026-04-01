import torch
import torch.nn as nn
from torchvision import models

class ModifiedResNet50(nn.Module):
    def __init__(self, num_outputs=1, dropout_rate=0.5):
        super(ModifiedResNet50, self).__init__()
        self.pretrained_model = models.resnet50(pretrained=True)
        self.features_conv = nn.Sequential(*list(self.pretrained_model.children())[:-2])
        self.gradients = None
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        num_ftrs = self.pretrained_model.fc.in_features
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_outputs)
        )

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.features_conv(x)
        if x.requires_grad:
            x.register_hook(self.activations_hook)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.features_conv(x)