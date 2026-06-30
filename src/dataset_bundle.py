import torch.nn as nn
import torchvision.transforms as transforms
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.loss import ElementwiseLoss

from .models import Classifier, ResNet


class PACS:
    def __init__(
        self,
        dataset,
        probabilistic=False,
        backbone="resnet50",
        freeze_bn=True,
    ):
        self.dataset = dataset
        self.probabilistic = probabilistic
        self.backbone = backbone
        self.freeze_bn = bool(freeze_bn)
        self.input_shape = (3, 224, 224)
        self.groupby_fields = ["domain"]
        self.grouper = CombinatorialGrouper(
            dataset=dataset,
            groupby_fields=self.groupby_fields,
        )
        self.loss = ElementwiseLoss(
            loss_fn=nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        )
        self.train_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                transforms.RandomGrayscale(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.featurizer = ResNet(
            self.input_shape,
            probabilistic=probabilistic,
            backbone=backbone,
            freeze_bn=self.freeze_bn,
        )
        self.classifier = Classifier(self.featurizer.n_outputs, self.n_classes)

    @property
    def n_classes(self):
        return self.dataset.n_classes

    @property
    def is_classification(self):
        return True

    @property
    def key_metric(self):
        return "acc_avg"

    @property
    def name(self):
        return "pacs"


class OfficeHome(PACS):
    @property
    def name(self):
        return "officehome"
