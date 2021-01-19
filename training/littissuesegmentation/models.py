from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
import pytorch_lightning as pl

from littissuesegmentation.losses import (
    DiceLossWithLogits,
    BinaryFocalLossWithLogits,
    binary_mean_iou,
    find_average,
)
import segmentation_models_pytorch as smp


class LitSMPModel(pl.LightningModule):
    def __init__(self, lr, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.losses = [
            ("dice", 0.5, DiceLossWithLogits()),
            ("focal", 2, BinaryFocalLossWithLogits(gamma=2)),
        ]

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        eta = self.lr / 100

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=1, T_mult=2, eta_min=eta, verbose=True
        )

        return [optimizer], [scheduler]

    def _step(self, batch, batch_idx):
        features, mask = batch

        logits = self.forward(features)

        total_loss = 0
        losses = {}
        for loss_name, weight, loss in self.losses:
            ls_mask = loss(logits, mask)
            losses[loss_name] = ls_mask
            total_loss += weight * ls_mask

        return {"loss": total_loss, "logits": logits, "mask": mask, "losses": losses}

    def training_step(self, batch, batch_idx):
        out = self._step(batch, batch_idx)
        self.log_dict(
            {"train_" + k: v for k, v in out["losses"].items()}, prog_bar=True
        )
        self.log_dict({"train_loss": out["loss"]})

        return out["loss"]

    def validation_step(self, batch, batch_id):
        out = self._step(batch, batch_id)

        ret_val = {"val_" + k: v for k, v in out["losses"].items()}
        metrics_log = {
            "val_loss": out["loss"],
            "val_iou": binary_mean_iou(out["logits"], out["mask"]),
            "val_dice": out["losses"]["dice"],
        }

        ret_val.update(metrics_log)
        self.log_dict(ret_val)

        return ret_val

    def validation_epoch_end(self, outputs):
        avg_metrics = {
            "avg_val_iou": find_average(outputs, "val_iou"),
            "avg_val_dice": find_average(outputs, "val_dice"),
        }

        self.log_dict(avg_metrics, prog_bar=True)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(
            parents=[parent_parser],
            add_help=False,
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "--lr",
            default=1e-4,
            type=float,
            help="Learning rate to use",
        )
        parser.add_argument(
            "--encoder_name",
            default="resnet34",
            type=str,
            help="Name of the encoder to use",
        )
        parser.add_argument(
            "--architecture",
            default="UNET",
            type=str,
            help=f"Either of: {', '.join(ARCHITECTURES)}",
        )

        return parser


class LitSMPUnet(LitSMPModel):
    def __init__(self, encoder_name="resnet34", lr=1e-4, *args, **kwargs):
        super().__init__(lr=lr)
        self.model = smp.Unet(
            encoder_name=encoder_name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pretrained weights for encoder initialization
            in_channels=3,  # model input channels (1 for grayscale images, 3 for RGB, etc.)
            classes=1,  # model output channels (number of classes in your dataset)
            *args,
            **kwargs,
        )


class LitSMPFPN(LitSMPModel):
    def __init__(self, encoder_name="resnet34", lr=1e-4, *args, **kwargs):
        super().__init__(lr=lr)
        self.model = smp.FPN(
            encoder_name=encoder_name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pretrained weights for encoder initialization
            in_channels=3,  # model input channels (1 for grayscale images, 3 for RGB, etc.)
            classes=1,  # model output channels (number of classes in your dataset)
            *args,
            **kwargs,
        )


class LitSMPDeepLabV3Plus(LitSMPModel):
    def __init__(self, encoder_name="resnet34", lr=1e-4, *args, **kwargs):
        super().__init__(lr=lr)
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pretrained weights for encoder initialization
            in_channels=3,  # model input channels (1 for grayscale images, 3 for RGB, etc.)
            classes=1,  # model output channels (number of classes in your dataset)
            *args,
            **kwargs,
        )


ARCHITECTURES = {
    "UNET": LitSMPUnet,
    "FPN": LitSMPFPN,
    "DeepLabV3": LitSMPDeepLabV3Plus,
}
