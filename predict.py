from __future__ import annotations

import os
import sys
import warnings

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from dataset import create_dataloaders
from lightning import Text2SQLLightningModule
from utils import read_json, write_json

warnings.filterwarnings("ignore")


def main(config: DictConfig):
    _, _, test_dataloader = create_dataloaders(config)
    if config.inference.generate_with_predict:
        checkpoint = ModelCheckpoint(monitor="val/RS10", mode="max", save_weights_only=True)
    else:
        checkpoint = ModelCheckpoint(monitor="val/loss_total", mode="min", save_weights_only=True)

    trainer = Trainer(
        accelerator="gpu",
        devices="auto",
        precision=config.train.precision,
        strategy="ddp",
        max_steps=config.scheduler.num_training_steps,
        log_every_n_steps=config.train.log_every_n_steps,
        gradient_clip_val=config.train.gradient_clip_val,
        accumulate_grad_batches=config.train.accumulate_grad_batches,
        val_check_interval=config.train.validation_interval,
        logger=None,
        callbacks=[checkpoint, LearningRateMonitor("step")],
    )
    trainer.test(model=Text2SQLLightningModule(config), ckpt_path=config.predict.ckpt_path, dataloaders=[test_dataloader])

    # gather all predictions and save
    predictions = {}
    RESULT_DIR = f"./{config.logging.run_name}"
    predictions.update(read_json(os.path.join(RESULT_DIR, f"predictions_{i}.json")) for i in range(trainer.world_size))
    write_json(os.path.join(RESULT_DIR, "predictions.json"), predictions)
    os.system(f"cd {RESULT_DIR} && zip -r predictions.zip predictions.json")

if __name__ == "__main__":
    main(OmegaConf.merge(OmegaConf.load(sys.argv[1]), OmegaConf.from_cli()))