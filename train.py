import argparse
import os

# PyTorch & Pytorch Lightning
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from lightning import Trainer
import torch

# Custom packages
from src.dataset import TinyImageNetDatasetModule
from src.network import SimpleClassifier
import src.config as cfg

torch.set_float32_matmul_precision('medium')


def main():
    parser = argparse.ArgumentParser(description="Train Tiny-ImageNet Classifier")
    parser.add_argument("--model_name", type=str, default=cfg.MODEL_NAME)
    parser.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=cfg.OPTIMIZER_PARAMS["lr"])
    parser.add_argument("--gamma", type=float, default=cfg.SCHEDULER_PARAMS["gamma"])
    parser.add_argument("--milestones", nargs="+", type=int, default=cfg.SCHEDULER_PARAMS["milestones"],
                        help="Milestones for MultiStepLR (e.g., --milestones 20 30)")

    args = parser.parse_args()

    # Apply CLI arguments to config
    cfg.MODEL_NAME = args.model_name
    cfg.BATCH_SIZE = args.batch_size
    cfg.OPTIMIZER_PARAMS["lr"] = args.lr
    cfg.SCHEDULER_PARAMS["gamma"] = args.gamma
    cfg.SCHEDULER_PARAMS["milestones"] = args.milestones

    # Update wandb name
    cfg.WANDB_NAME = f'{cfg.MODEL_NAME}-B{cfg.BATCH_SIZE}-{cfg.OPTIMIZER_PARAMS["type"]}'
    cfg.WANDB_NAME += f'-{cfg.SCHEDULER_PARAMS["type"]}{cfg.OPTIMIZER_PARAMS["lr"]:.1E}'

    # Instantiate model and data
    model = SimpleClassifier(
        model_name=cfg.MODEL_NAME,
        num_classes=cfg.NUM_CLASSES,
        optimizer_params=cfg.OPTIMIZER_PARAMS,
        scheduler_params=cfg.SCHEDULER_PARAMS,
    )

    datamodule = TinyImageNetDatasetModule(
        batch_size=cfg.BATCH_SIZE,
    )

    wandb_logger = WandbLogger(
        project=cfg.WANDB_PROJECT,
        save_dir=cfg.WANDB_SAVE_DIR,
        entity=cfg.WANDB_ENTITY,
        name=cfg.WANDB_NAME,
    )

    trainer = Trainer(
        accelerator=cfg.ACCELERATOR,
        devices=cfg.DEVICES,
        precision=cfg.PRECISION_STR,
        max_epochs=cfg.NUM_EPOCHS,
        check_val_every_n_epoch=cfg.VAL_EVERY_N_EPOCH,
        logger=wandb_logger,
        callbacks=[
            LearningRateMonitor(logging_interval='epoch'),
            ModelCheckpoint(save_top_k=1, monitor='accuracy/val', mode='max'),
            EarlyStopping(monitor='loss/val', mode='min', patience=5, verbose=True),
        ],
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.validate(ckpt_path='best', datamodule=datamodule)


if __name__ == "__main__":
    main()
