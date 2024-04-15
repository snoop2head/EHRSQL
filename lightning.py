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
from utils import read_json, write_json

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
        if "text2sql" in config.model.name_or_path and "t5" in config.model.name_or_path:
            pass
        else:
            self.tokenizer.add_tokens(["<"])
            self.t5.resize_token_embeddings(len(self.tokenizer))
        
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
            output_hidden_states=False,
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
        acc = (binary_label == binary_pred).sum().item() / binary_label.shape[0]
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
        
        if len(generated_texts) != 0 and len(target_texts) != 0:
            bleu_metric = bleu.compute(predictions=generated_texts, references=target_texts)
            bleu_score = bleu_metric["bleu"]
        else:
            bleu_score = 0.0

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
            "bleu": bleu_score,
            "RS0": accuracy0,
            "RS5": accuracy5,
            "RS10": accuracy10,
            "RSN": accuracyN,
            "pred": [[g, t] for g, t in zip(generated_texts, target_texts)],
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
            self.logger.log_text(key="val/pred", columns=["generated", "label"], data=pred)
        self.log_dict({f"val/{k}": v for k, v in metrics.items()}, sync_dist=True)


    def on_test_epoch_start(self):
        """ TEST: https://github.com/ReadingLips/sync_auto_avsr/blob/mainv2/lightning.py """
        self.predictions = {}

    def test_step(self, batch: dict[str, torch.Tensor], idx: int) -> torch.Tensor:
        # generate
        outputs = self.t5.generate(
            input_ids=batch["source_ids"],
            max_length=self.config.data.max_target_length,
            num_beams=self.config.inference.num_beams,
            repetition_penalty=self.config.inference.repetition_penalty,
            num_return_sequences=self.config.inference.num_return_sequences,
        )
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        if self.config.inference.post_process:
            generated_texts = [post_process_sql(g) for g in generated_texts]
        
        # classify
        outputs = self.t5.encoder(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            output_hidden_states=False,
        )

        logits_impossible = self.is_impossible_head(outputs.last_hidden_state.mean(dim=1))
        logits_impossible = logits_impossible.squeeze(-1).float()
        binary_pred = (logits_impossible.sigmoid().detach().cpu().numpy() > self.threshold).astype(int)
        
        for i, p, b in zip(batch["id"], generated_texts, binary_pred):
            if b == 1 or b == True:
                self.predictions[f"{i}"] = "null"
            elif b == 0 or b == False:
                self.predictions[f"{i}"] = p
        
    def on_test_epoch_end(self):
        # for each ddp process, save predictions
        device_id = self.local_rank if self.local_rank != -1 else 0
        RESULT_DIR = f"./RESULTS/{self.config.logging.run_name}"
        os.makedirs(RESULT_DIR, exist_ok=True)
        pred_filename = f"predictions_{device_id}.json" if not self.config.predict.inference_valid else f"predictions_valid_{device_id}.json"
        prediction_path = os.path.join(RESULT_DIR, pred_filename)
        write_json(prediction_path, self.predictions)

    def configure_optimizers(self) -> tuple[list[Optimizer], list[dict[str, Any]]]:
        do_decay = [p for p in self.parameters() if p.requires_grad and p.ndim >= 2]
        no_decay = [p for p in self.parameters() if p.requires_grad and p.ndim < 2]
        param_groups = [{"params": do_decay}, {"params": no_decay, "weight_decay": 0.0}]

        optimizer = AdamW(param_groups, **self.config.optimizer)
        scheduler = get_scheduler(optimizer=optimizer, **self.config.scheduler)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]