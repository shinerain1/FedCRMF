import torch
import torch.nn as nn
import torchvision.models


class ResNet(nn.Module):
    """ResNet featurizer with optional frozen BatchNorm."""

    def __init__(
        self,
        input_shape,
        feature_dimension=None,
        probabilistic=False,
        backbone="resnet50",
        freeze_bn=True,
    ):
        super().__init__()
        self.probabilistic = probabilistic
        self.backbone = str(backbone).lower()
        self.freeze_bn_enabled = bool(freeze_bn)
        if self.backbone == "resnet18":
            self.network = torchvision.models.resnet18(pretrained=True)
            default_outputs = 512
        elif self.backbone == "resnet50":
            self.network = torchvision.models.resnet50(pretrained=True)
            default_outputs = 2048
        else:
            raise ValueError(f"Unsupported ResNet backbone: {backbone}")
        self.n_outputs = int(feature_dimension or default_outputs)

        nc = input_shape[0]
        if nc != 3:
            old_weight = self.network.conv1.weight.data.clone()
            self.network.conv1 = nn.Conv2d(
                nc,
                64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )
            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = old_weight[:, i % 3, :, :]

        self.dropout = nn.Dropout(0)
        out_dim = self.n_outputs * 2 if probabilistic else self.n_outputs
        self.network.fc = nn.Linear(self.network.fc.in_features, out_dim)

    def forward(self, x):
        return self.dropout(self.network(x))

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_bn_enabled:
            self.freeze_bn()
        return self

    def freeze_bn(self):
        for module in self.network.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


def Classifier(in_features, out_features):
    return nn.Linear(in_features, out_features)
