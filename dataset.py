from __future__ import annotations

import glob
import os
from dataclasses import dataclass
import multiprocessing
import json
import random

from omegaconf import DictConfig
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

def encode_file(tokenizer, text, max_length, truncation=True, padding=True, return_tensors="pt"):
    """
    Tokenizes the text and returns tensors.
    """
    return tokenizer(
        text,
        max_length=max_length,
        truncation=truncation,
        padding=padding,
        return_tensors=return_tensors,
    )

@dataclass
class T5Dataset(Dataset):
    """
    A dataset class for the T5 model, handling the conversion of natural language questions to SQL queries.
    """
    def __init__(
        self,
        config: DictConfig,
        data_dir: str,
        tables_file: str = None,
        is_test=False,
    ):

        super().__init__()

        self.is_test = is_test # this option does not include target label

        self.tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path)
        if "text2sql" in config.model.name_or_path and "t5" in config.model.name_or_path:
            pass
        else:
            self.tokenizer.add_tokens(["<"])

        self.db_id = config.data.db_id
        self.max_source_length = config.data.max_source_length
        self.max_target_length = config.data.max_target_length
        self.random = random.Random(config.seed) # Schema shuffling

        # Load data from JSON files
        with open(f'{data_dir}/data.json') as json_file:
            data = json.load(json_file)["data"]

        label = {}
        if not self.is_test:
            with open(f'{data_dir}/label.json') as json_file:
                label = json.load(json_file)

        self.db_json = None
        if tables_file:
            with open(tables_file) as f:
                self.db_json = json.load(f)

        # Process and encode the samples from the loaded data
        ids = []
        questions = []
        labels = []
        is_impossibles = []
        for sample in data:

            # id
            if config.data.exclude_unans:
                if sample["id"] in label and label[sample["id"]] == "null":
                    continue
            ids.append(sample['id'])

            # question
            question = self.preprocess_sample(sample, config.data.append_schema_info)
            questions.append(question)

            # label
            if not self.is_test:
                labels.append(label[sample["id"]])
                
                # NEWLY ADDED
                if sample["id"] in label and label[sample["id"]] == "null":
                    is_impossibles.append(True)
                else:
                    is_impossibles.append(False)

        self.ids = ids
        question_encoded = encode_file(self.tokenizer, questions, max_length=self.max_source_length)
        self.source_ids, self.source_mask = question_encoded['input_ids'], question_encoded['attention_mask']
        if not self.is_test:
            label_encoded = encode_file(self.tokenizer, labels, max_length=self.max_target_length)
            self.target_ids = label_encoded['input_ids']
            self.is_impossibles = is_impossibles

    def __len__(self):
        return len(self.source_ids)

    def __getitem__(self, index):
        if self.is_test:
            return {
                "id": self.ids[index],
                "source_ids": self.source_ids[index],
                "source_mask": self.source_mask[index]
            }
        else:
            return {
                "id": self.ids[index],
                "source_ids": self.source_ids[index],
                "source_mask": self.source_mask[index],
                "target_ids": self.target_ids[index],
                "target_is_impossible": self.is_impossibles[index]
            }

    def preprocess_sample(self, sample, append_schema_info=False):
        """
        Processes a single data sample, adding schema description to the question.
        """
        question = sample["question"]

        if append_schema_info:
            if self.db_json:
                tables_json = [db for db in self.db_json if db["db_id"] == self.db_id][0]
                schema_description = self.get_schema_description(tables_json)
                schema_description = ", ".join(schema_description)
                question = f"convert question and table into SQL query. tables: {schema_description}. question: {question}"
            # breakpoint()
            return question
        else:
            return question

    def get_schema_description(self, tables_json, shuffle_schema=False):
        """
        Generates a textual description of the database schema.
        """
        table_names = tables_json["table_names_original"]
        if shuffle_schema:
            self.random.shuffle(table_names)

        columns = [
            (column_name[0], column_name[1].lower(), column_type.lower())
            for column_name, column_type in zip(tables_json["column_names_original"], tables_json["column_types"])
        ]

        schema_description = []
        for table_index, table_name in enumerate(table_names):
            table_columns = [column[1] for column in columns if column[0] == table_index]
            if shuffle_schema:
                self.random.shuffle(table_columns)
            column_desc = ",".join(table_columns)
            schema_description.append(f"{table_name.lower()}({column_desc})")

        return schema_description

    def collate_fn(self, batch, return_tensors='pt', padding=True, truncation=True):
        """
        Collate function for the DataLoader.
        """
        ids = [x["id"] for x in batch]
        input_ids = torch.stack([x["source_ids"] for x in batch]) # BS x SL
        masks = torch.stack([x["source_mask"] for x in batch]) # BS x SL
        pad_token_id = self.tokenizer.pad_token_id
        source_ids, source_mask = trim_batch(input_ids, pad_token_id, attention_mask=masks)

        if self.is_test:
            return {
                "source_ids": source_ids,
                "source_mask": source_mask,
                "id": ids,
            }
        else:
            target_ids = torch.stack([x["target_ids"] for x in batch]) # BS x SL
            target_ids = trim_batch(target_ids, pad_token_id)
            target_is_impossible = torch.tensor([x["target_is_impossible"] for x in batch], dtype=torch.long)
            return {
                "source_ids": source_ids,
                "source_mask": source_mask,
                "target_ids": target_ids,
                "id": ids,
                "target_is_impossible": target_is_impossible,
            }

def trim_batch(input_ids, pad_token_id, attention_mask=None):
    """
    Trims padding from batches of tokenized text.
    """
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])

def create_dataloaders(config: DictConfig) -> tuple[DataLoader, DataLoader, DataLoader]:

    if config.data.split_ratio != 1.0 and config.data.kfold_split is False:
        print("Legacy data split as ehr instruction")
        NEW_TRAIN_DIR = os.path.join(config.data.base_data_dir, '__train')
        NEW_VALID_DIR = os.path.join(config.data.base_data_dir, '__valid')
    elif config.data.split_ratio != 1.0 and type(config.data.kfold_split) == int:
        print("K-Fold data split")
        NEW_TRAIN_DIR = os.path.join(config.data.base_data_dir, f'__train_fold{config.data.kfold_split}')
        NEW_VALID_DIR = os.path.join(config.data.base_data_dir, f'__valid_fold{config.data.kfold_split}')
    if config.data.split_ratio == 1.0:
        NEW_TRAIN_DIR = os.path.join(config.data.base_data_dir, 'train')
        NEW_VALID_DIR = os.path.join(config.data.base_data_dir, 'train')
    NEW_TEST_DIR = os.path.join(config.data.base_data_dir, 'valid')
    TABLES_PATH = os.path.join('data', config.data.db_id, 'tables.json')               # JSON containing database schema
    
    train_dataset = T5Dataset(config, NEW_TRAIN_DIR, tables_file=TABLES_PATH, is_test=False)
    valid_dataset = T5Dataset(config, NEW_VALID_DIR, tables_file=TABLES_PATH, is_test=False)
    test_dataset = T5Dataset(config, NEW_TEST_DIR, tables_file=TABLES_PATH, is_test=True)
    # breakpoint()

    cpu_count = multiprocessing.cpu_count()
    gpu_count = torch.cuda.device_count()
    num_workers = cpu_count // (gpu_count * 2) if gpu_count > 0 else cpu_count // 2

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train.train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_dataloader = DataLoader(
        valid_dataset,
        batch_size=config.train.valid_batch_size,
        shuffle=False if config.data.split_ratio != 1.0 else True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=valid_dataset.collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.train.test_batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=test_dataset.collate_fn,
    )
    return train_dataloader, val_dataloader, test_dataloader