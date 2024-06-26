from __future__ import annotations

import os
import sys
import warnings
import time

import numpy as np
import random
import torch

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from dataset import create_dataloaders
from lightning import Text2SQLLightningModule
from utils import gather_and_save

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
    if type(config.data.kfold_split) == int:
        config.logging.run_name = f"{config.logging.run_name}_fold{config.data.kfold_split}"
    else:
        set_seed(config.seed)
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(config)
    
    checkpoint_name = config.logging.run_name
    checkpoint_name += "{epoch}-{step}"
    if config.inference.generate_with_predict and config.data.split_ratio != 1.0:
        checkpoint = ModelCheckpoint(
            monitor="val/RS10", 
            mode="max", 
            save_weights_only=True,
            filename=checkpoint_name
        )
    else:
        checkpoint = ModelCheckpoint(
            monitor="val/loss_total", 
            mode="min", 
            save_weights_only=True,
            filename=checkpoint_name
        )

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
        logger=WandbLogger(
            config.logging.run_name,
            project=config.logging.project_name,
            entity=config.logging.entity_name,
        ),
        callbacks=[checkpoint, LearningRateMonitor("step")],
    )
    trainer.fit(Text2SQLLightningModule(config), train_dataloader, val_dataloader)
    trainer.test(model=Text2SQLLightningModule(config), ckpt_path=checkpoint.best_model_path, dataloaders=[test_dataloader])
    time.sleep(10) # wait for all processes to finish
    gather_and_save(config, trainer, filter_error_pred=False) # gather all predictions and save

if __name__ == "__main__":
    main(OmegaConf.merge(OmegaConf.load(sys.argv[1]), OmegaConf.from_cli()))