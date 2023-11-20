from .data import CIFAR10DataModule
from .nn import LitViT
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

LEARNING_RATE = 0.0001
BATCH_SIZE = 64


def train(path, epochs):
    config = dict(
        num_labels=10,
        batch_size=BATCH_SIZE,
        epochs=epochs,
        architecture="ViT"
    )
    dm = CIFAR10DataModule()
    model = LitViT(img_size=32,
                   patch_size=4,
                   in_chans=3,
                   num_classes=dm.num_classes,
                   len_train=40000,
                   batch_size=BATCH_SIZE,
                   epochs=epochs)
    callbacks = [ModelCheckpoint(dirpath="./checkpoints",
                                 every_n_train_steps=1)]
    wandb_logger = WandbLogger(project="cifar10-vit",
                               config=config)
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        default_root_dir=path,
        logger=wandb_logger,
        callbacks=callbacks
    )
    trainer.fit(model, dm)