from __future__ import annotations

import shutil
import sys
import warnings

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from dataset import create_dataloaders
from lightning import Text2SQLLightningModule

warnings.filterwarnings("ignore")


def main(config: DictConfig):
    _, _, test_dataloader = create_dataloaders(config)
    if config.inference.generate_with_predict:
        checkpoint = ModelCheckpoint(monitor="val/bleu", mode="max", save_weights_only=True)
    else:
        checkpoint = ModelCheckpoint(monitor="val/loss_total", mode="min", save_weights_only=True)

    trainer = Trainer(
        accelerator="gpu",
        devices="auto",
        precision=config.train.precision,
        amp_backend="native",
        strategy="ddp",
        max_steps=config.scheduler.num_training_steps,
        log_every_n_steps=config.train.log_every_n_steps,
        gradient_clip_val=config.train.gradient_clip_val,
        accumulate_grad_batches=config.train.accumulate_grad_batches,
        val_check_interval=config.train.validation_interval,
        logger=WandbLogger(
            config.logging.run_name,
            project=config.logging.project_name,
            entity=config.logging.entity_name,
        ),
        callbacks=[checkpoint, LearningRateMonitor("step")],
    )
    trainer.test(model=Text2SQLLightningModule(config), ckpt_path=config.evaluate.ckpt_path, dataloaders=[test_dataloader])

if __name__ == "__main__":
    main(OmegaConf.merge(OmegaConf.load(sys.argv[1]), OmegaConf.from_cli()))