"""The lightning model class for training the object detection."""
from typing import Any
from pytorch_lightning import LightningModule
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from utils.io_utils import visualize_prediction

class ObjectDetectionModel(LightningModule):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.backbone = None
        self.model_head = None

        self.model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.hparams.num_classes)
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        """module forward pass"""
        prediction = self.model(x)
        return prediction

    def configure_optimizers(self) -> dict:
        """set up optimizer and lr-scheduler"""
        optimizer = SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay
        )
        lr_scheduler = StepLR(
            optimizer,
            step_size=self.hparams.step_size,
            gamma=self.hparams.gamma
        )
        return {
           'optimizer': optimizer,
           'lr_scheduler': lr_scheduler,
           'monitor': 'val_loss'
        }

    def training_step(self, train_batch: tuple, batch_idx: int) -> float:
        """perform a training step"""
        imgs, target = train_batch
        imgs = torch.stack(imgs)
        prediction = self.model(imgs, target)
        
        loss = sum(loss for loss in prediction.values())
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss
    
    def validation_step(self, val_batch: tuple, batch_idx: int):
        """perform a validation step"""
        imgs, target = val_batch
        imgs = torch.stack(imgs)
        self.model.train()
        prediction = self.model(imgs, target)
        loss = sum(loss for loss in prediction.values())
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
    
    
    def predict_step(self, predict_batch: tuple, batch_idx: int):
        """perform a validation step"""
        imgs, target = predict_batch
        imgs = torch.stack(imgs)
        self.model.eval()
        prediction = self.model(imgs)
        visualize_prediction(imgs, prediction, "./data/out/viz/", target, denormalize=False, top_n=1)
        # visualize the prediction 