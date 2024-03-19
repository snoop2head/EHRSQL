from __future__ import annotations

import os
from typing import Any

import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb
from omegaconf import DictConfig
from torch.optim import AdamW, Optimizer
from transformers import T5Tokenizer, T5Config, T5Model, T5ForConditionalGeneration, get_scheduler

# Metrics
from sklearn import metrics
from evaluate import load
from scoring_program.scoring_utils import execute_all, reliability_score, penalize
from scoring_program.postprocessing import post_process_sql

class Text2SQLLightningModule(pl.LightningModule):
    """
    https://github.com/snoop2head/DotNeuralNet/tree/main/src
    https://github.com/ReadingLips/auxiliary-audio-LRS/tree/main/lrs2/src
    """
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        # Tokenizer and Transformer Backbone
        self.tokenizer = T5Tokenizer.from_pretrained(config.model.name_or_path)
        self.t5 = T5ForConditionalGeneration.from_pretrained(config.model.name_or_path)
        
        # Head for is_impossible
        self.is_impossible_head = nn.Sequential(
            nn.Dropout(config.train.is_impossible_dropout),
            nn.Linear(self.t5.config.d_model, 1, bias=False),
        )
        self.lambda_null_classification = config.lambda_null_classification
        self.threshold = config.inference.is_impossible_threshold

    def forward(
        self, id: torch.Tensor, source_ids: torch.Tensor, source_mask: torch.Tensor, target_ids: torch.Tensor, target_is_impossible: torch.LongTensor
    ) -> dict[str, torch.Tensor]:
        """ 
        T5Model: https://github.com/huggingface/transformers/blob/59d5edef34ae0fa56065a2e863736d4f133c558b/src/transformers/models/t5/modeling_t5.py#L1367-L1477
        T5ConditionalGeneration: https://github.com/huggingface/transformers/blob/59d5edef34ae0fa56065a2e863736d4f133c558b/src/transformers/models/t5/modeling_t5.py#L1564-L1613
        """
        target_ids[target_ids[:, :] == self.tokenizer.pad_token_id] = -100 # ignore padding in target for better loss calculation
        # Captioning Loss (LM Loss)
        outputs = self.t5(
            input_ids=source_ids,
            attention_mask=source_mask,
            labels=target_ids,
            output_hidden_states=True,
        )
        loss_captioning = outputs.loss

        # Binary Classification for Answerable vs Unanswerable
        logits_impossible = self.is_impossible_head(outputs.encoder_last_hidden_state.mean(dim=1))
        logits_impossible = logits_impossible.squeeze(-1).float()
        loss_impossible = F.binary_cross_entropy_with_logits(logits_impossible, target_is_impossible.float())
        
        # Composite Loss
        loss_total = loss_captioning + loss_impossible * self.lambda_null_classification

        # Compute Binary Classification metrics
        binary_label = target_is_impossible.detach().cpu().numpy()
        binary_pred = (logits_impossible.sigmoid().detach().cpu().numpy() > self.threshold).astype(int)
        acc = metrics.accuracy_score(binary_label, binary_pred)
        precision, recall, f1, _ = metrics.precision_recall_fscore_support(binary_label, binary_pred, average="binary")

        return {
            "loss_total": loss_total,
            "loss_captioning": loss_captioning,
            "loss_impossible": loss_impossible,
            "acc": acc,
            "f1": f1,
            # "precision": precision, # IDK Why but the results are the same as f1
            # "recall": recall, # IDK Why but the results are the same as f1
        }
    
    def generate_bleu(
        self, id: torch.Tensor, source_ids: torch.Tensor, source_mask: torch.Tensor, target_ids: torch.Tensor, target_is_impossible: torch.LongTensor
    ) -> dict[str, torch.Tensor]:
        """ measures word error rate using huggingface metrics """
        bleu = load("bleu")

        outputs = self.t5.generate(
            input_ids=source_ids,
            max_length=self.config.data.max_target_length,
            num_beams=self.config.inference.num_beams,
            repetition_penalty=self.config.inference.repetition_penalty,
            num_return_sequences=self.config.inference.num_return_sequences,
        )

        # revert -100 to padding token for target_ids
        target_ids[target_ids[:, :] == -100] = self.tokenizer.pad_token_id
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        target_texts = self.tokenizer.batch_decode(target_ids, skip_special_tokens=True)

        # drop samples with target_is_impossible = 1, since they are not answerable
        id = np.array(id)
        id = id[(target_is_impossible == 0).detach().cpu().numpy()]
        generated_texts = [g for g, imp in zip(generated_texts, target_is_impossible) if imp == 0]
        target_texts = [t for t, imp in zip(target_texts, target_is_impossible) if imp == 0]

        # get bleu metrics
        if self.config.inference.post_process:
            generated_texts = [post_process_sql(g) for g in generated_texts]
            target_texts = [post_process_sql(t) for t in target_texts]
        bleu_metric = bleu.compute(predictions=generated_texts, references=target_texts)

        # get sql reliability score
        DB_PATH = os.path.join('data', self.config.data.db_id, f'{self.config.data.db_id}.sqlite')
        real_result = execute_all({i: t for i, t in zip(id, target_texts)}, db_path=DB_PATH, tag='real')
        pred_result = execute_all({i: g for i, g in zip(id, generated_texts)}, db_path=DB_PATH, tag='pred')
        scores, score_dict = reliability_score(real_result, pred_result, return_dict=True)
        accuracy0 = penalize(scores, penalty=0)
        accuracy5 = penalize(scores, penalty=5)
        accuracy10 = penalize(scores, penalty=10)
        accuracyN = penalize(scores, penalty=1000)

        return {
            "bleu": bleu_metric["bleu"],
            "RS0": accuracy0,
            "RS5": accuracy5,
            "RS10": accuracy10,
            "RSN": accuracyN,
            "pred": wandb.Table(columns=["pred", "target"], data=[[g, t] for g, t in zip(generated_texts, target_texts)]),
        }

    
    def training_step(self, batch: dict[str, torch.Tensor], idx: int) -> torch.Tensor:
        metrics = self(**batch)
        self.log_dict({f"train/{k}": v for k, v in metrics.items()})
        return metrics["loss_total"]

    def validation_step(self, batch: dict[str, torch.Tensor], idx: int) -> torch.Tensor:
        metrics = self(**batch)
        if self.config.inference.generate_with_predict:
            metrics.update(self.generate_bleu(**batch)) # bleu
            pred = metrics.pop("pred")
            # self.log_table("val/pred", pred, sync_dist=True) # needs higher version
        self.log_dict({f"val/{k}": v for k, v in metrics.items()}, sync_dist=True)

    def test_step(self, batch: dict[str, torch.Tensor], idx: int) -> torch.Tensor:
        """ NEEDS TO BE IMPLEMENTED"""
        metrics = self(**batch) # loss
        self.log_dict({f"test/{k}": v for k, v in metrics.items()}, sync_dist=True)

    def configure_optimizers(self) -> tuple[list[Optimizer], list[dict[str, Any]]]:
        do_decay = [p for p in self.parameters() if p.requires_grad and p.ndim >= 2]
        no_decay = [p for p in self.parameters() if p.requires_grad and p.ndim < 2]
        param_groups = [{"params": do_decay}, {"params": no_decay, "weight_decay": 0.0}]

        optimizer = AdamW(param_groups, **self.config.optimizer)
        scheduler = get_scheduler(optimizer=optimizer, **self.config.scheduler)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]