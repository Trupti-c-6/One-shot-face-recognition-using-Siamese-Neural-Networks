import torch
import torch.nn as nn
import torchvision.models as models


class SiameseNetwork(nn.Module):
    def __init__(self, backbone="resnet18"):
        super(SiameseNetwork, self).__init__()

        if backbone == "resnet18":
            self.backbone = models.resnet18(pretrained=True)
            self.backbone.fc = nn.Identity()
            embedding_dim = 512

        elif backbone == "mobilenetv3":
            self.backbone = models.mobilenet_v3_large(pretrained=True)
            self.backbone.classifier = nn.Identity()
            embedding_dim = 960

        else:
            raise ValueError("Unsupported backbone")

        self.embedding = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

    def forward_once(self, x):
        x = self.backbone(x)
        x = self.embedding(x)
        return x

    def forward(self, anchor, positive, negative):
        anchor_out = self.forward_once(anchor)
        positive_out = self.forward_once(positive)
        negative_out = self.forward_once(negative)

        return anchor_out, positive_out, negative_out
