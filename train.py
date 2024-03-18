from __future__ import annotations

import shutil
import sys
import warnings

import numpy as np
import random
import torch

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from dataset import create_dataloaders
from lightning import Text2SQLLightningModule

warnings.filterwarnings("ignore")

def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main(config: DictConfig):
    set_seed(config.seed)
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(config)
    checkpoint = ModelCheckpoint(
        monitor="val/bleu", mode="max", save_weights_only=True
    )

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
    trainer.fit(Text2SQLLightningModule(config), train_dataloader, val_dataloader)
    trainer.test(ckpt_path=checkpoint.best_model_path, dataloaders=[test_dataloader])
    shutil.copy(checkpoint.best_model_path, ".")

if __name__ == "__main__":
    main(OmegaConf.merge(OmegaConf.load(sys.argv[1]), OmegaConf.from_cli()))