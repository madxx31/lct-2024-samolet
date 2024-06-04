import torch
import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torch.utils.data import Sampler
import math
from transformers import AutoTokenizer
import os
from sklearn.model_selection import StratifiedKFold
import ast
from omegaconf import OmegaConf


def get_strata(x):
    if x == {}:
        return 0
    elif len(x) == 3:
        return 1
    elif "B-discount" in x and "B-value" not in x and "I-value" not in x:
        return 2
    else:
        return 3


label2id = {"B-discount": 1, "B-value": 2, "I-value": 3}


class BucketBatchSampler(Sampler):
    def __init__(self, lengths, batch_size, drop_last=False, length_noise=0.05, seed=17):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.lengths = np.array(lengths)
        self.length_noise = length_noise
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        noise = np.random.uniform(1 - self.length_noise, 1 + self.length_noise, len(self.lengths))
        ids = np.argsort(self.lengths * noise)
        if self.drop_last and len(ids) % self.batch_size > 0:
            drop_cnt = len(ids) % self.batch_size
            drop_idx = self.rng.choice(len(ids), size=drop_cnt, replace=False)
            ids = np.delete(ids, drop_idx)
        self.batches = [ids[i : i + self.batch_size] for i in range(0, len(ids), self.batch_size)]
        self.rng.shuffle(self.batches)
        for i in self.batches:
            yield (i)

    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        else:
            return math.ceil(len(self.lengths) / self.batch_size)


class MyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: AutoTokenizer):
        super().__init__()
        self.enc = tokenizer(
            df.processed_text.to_list(),
            padding=False,
            truncation=False,
            # max_length=max_len,
            return_offsets_mapping=True,
        )
        self.token_labels = []
        self.word_labels = []
        self.words_mapping = []

        for target_dict, offsets, text in zip(
            df.target_labels_positions.values, self.enc["offset_mapping"], df.processed_text
        ):
            words = text.split(" ")
            splits = [i for i, j in enumerate(text) if j == " "]
            starts = np.array([-1] + splits) + 1
            ends = np.array(splits + [len(text)])

            tmp_words_mapping = []

            word_ind = 0
            for b, e in offsets:
                if b == e:
                    tmp_words_mapping.append(-1)
                    continue
                if b >= ends[word_ind]:
                    word_ind += 1
                if b < ends[word_ind] and e > starts[word_ind]:
                    tmp_words_mapping.append(word_ind)
                else:
                    tmp_words_mapping.append(-1)
            assert np.max(tmp_words_mapping) == len(words) - 1
            assert set(tmp_words_mapping).difference([-1]) == set(np.arange(len(words)))
            tmp_words_mapping = np.array(tmp_words_mapping)

            tmp_token_labels = np.zeros(len(offsets))
            tmp_word_labels = np.zeros(len(words))
            for label, word_indexes in target_dict.items():
                for word_index in word_indexes:
                    tmp_token_labels[tmp_words_mapping == word_index] = label2id[label]
                    tmp_word_labels[word_index] = label2id[label]

            self.words_mapping.append(tmp_words_mapping)
            self.token_labels.append(tmp_token_labels)
            self.word_labels.append(tmp_word_labels)

    def __getitem__(self, index):
        return {
            "input_ids": self.enc["input_ids"][index],
            "words_mapping": self.words_mapping[index],
            "token_labels": self.token_labels[index],
            "word_labels": self.word_labels[index],
        }

    def get_lengths(self):
        return [len(i) for i in self.enc["input_ids"]]

    def __len__(self):
        return len(self.enc["input_ids"])


class MyDataModule(pl.LightningDataModule):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg
        self.already_setup = False

    def setup(self, stage=None):
        if self.already_setup:
            return
        tr = pd.read_csv("train_data.csv")
        tr.target_labels_positions = tr.target_labels_positions.apply(lambda x: ast.literal_eval(x))
        tr["strata"] = tr.target_labels_positions.apply(get_strata)
        skf = StratifiedKFold(n_splits=5, random_state=17, shuffle=True)
        tr["fold"] = 0
        for fold, (trx, tex) in enumerate(skf.split(tr.index, tr.strata)):
            tr.loc[tex, "fold"] = fold

        tok = AutoTokenizer.from_pretrained(self.cfg.model.name, use_fast=True)
        self.pad_token_id = tok.pad_token_id
        self.train_ds = MyDataset(tr[tr.fold != self.cfg.data.fold], tok)
        self.val_ds = MyDataset(tr[tr.fold == self.cfg.data.fold], tok)

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.already_setup = True
        self.tok = tok

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_sampler=BucketBatchSampler(
                self.train_ds.get_lengths(), batch_size=self.cfg.data.train_batch_size, drop_last=True
            ),
            collate_fn=self.collator,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            shuffle=False,
            batch_size=self.cfg.data.eval_batch_size,
            collate_fn=self.collator,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
        )

    def collator(self, batch):
        # labels = -100 for word_mapping = -1
        batch_size = len(batch)
        max_text_length = max([len(i["input_ids"]) for i in batch])
        max_word_count = max([len(i["word_labels"]) for i in batch])
        input_ids = np.ones((batch_size, max_text_length)) * self.pad_token_id
        words_mapping = np.ones((batch_size, max_text_length)) * -1
        token_labels = np.ones((batch_size, max_text_length)) * -100
        word_labels = np.ones((batch_size, max_word_count)) * -100
        attention_mask = np.zeros((batch_size, max_text_length))
        for i, b in enumerate(batch):
            i_text_len = len(b["input_ids"])
            input_ids[i, :i_text_len] = b["input_ids"]
            attention_mask[i, :i_text_len] = 1
            words_mapping[i, :i_text_len] = b["words_mapping"]
            token_labels[i, :i_text_len] = b["token_labels"]
            word_labels[i, : len(b["word_labels"])] = b["word_labels"]
        token_labels[words_mapping == -1] = -100
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "words_mapping": torch.tensor(words_mapping, dtype=torch.long),
            "token_labels": torch.tensor(token_labels, dtype=torch.long),
            "word_labels": torch.tensor(word_labels, dtype=torch.long),
        }


if __name__ == "__main__":
    conf = OmegaConf.load("config.yaml")
    dm = MyDataModule(conf)
    dm.setup()
    print(pd.Series(dm.train_ds.get_lengths()).quantile([0.9, 0.95, 0.99, 1]))
    print(pd.Series(dm.val_ds.get_lengths()).quantile([0.9, 0.95, 0.99, 1]))
    # ds = dm.train_ds
    # ex = ds[17]
    # df = pd.DataFrame(
    #     {"w": ex["words_mapping"], "token": dm.tok.convert_ids_to_tokens(ex["input_ids"]), "label": ex["labels"]}
    # )
    # print(df.tail(10))
    # dl = dm.train_dataloader()
    # for b in dl:
    #     break
    # for k in b:
    #     print(k, b[k].shape)
    #     print(b[k])
