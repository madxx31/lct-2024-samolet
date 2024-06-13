#!/usr/bin/python
# coding: utf-8
from transformers import AutoModelForTokenClassification, AutoTokenizer
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from scipy.special import softmax
from flask import Flask, request, jsonify

label2id = {"B-discount": 1, "B-value": 2, "I-value": 3}


class MyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: AutoTokenizer, max_len):
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
        self.max_len = max_len
        self.text_ids = df.index.values

        if "target_labels_positions" not in df.columns:
            df["target_labels_positions"] = [{}] * len(df)

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
            "input_ids": self.enc["input_ids"][index][: self.max_len],
            "words_mapping": self.words_mapping[index][: self.max_len],
            "token_labels": self.token_labels[index][: self.max_len],
            "word_labels": self.word_labels[index][: int(self.words_mapping[index][: self.max_len].max()) + 1],
            "text_id": self.text_ids[index],
        }

    def get_lengths(self):
        return [len(i) for i in self.enc["input_ids"]]

    def __len__(self):
        return len(self.enc["input_ids"])


class Collator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
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
            "text_ids": [i["text_id"] for i in batch],
        }


def average_per_word_logits(token_logits: torch.Tensor, words_mapping: torch.Tensor) -> torch.Tensor:
    """
    Aggregates per-token logits and returns average per-word logits.

    Args:
        token_logits (torch.Tensor): A tensor of shape (batch, token, class).
        words_mapping (torch.Tensor): A tensor of shape (batch, token) with indices indicating to which word a token belongs.

    Returns:
        torch.Tensor: A tensor of shape (batch, word) with average logits per word.
    """
    batch_size, num_tokens, num_classes = token_logits.shape
    max_word_index = words_mapping.max().item()

    # Create a mask for valid tokens (ignore tokens mapped to -1)
    valid_mask = words_mapping != -1

    # Create a one-hot encoding of the words_mapping tensor
    word_one_hot = torch.nn.functional.one_hot(words_mapping * valid_mask, num_classes=max_word_index + 1).float()

    # Use the valid_mask to zero out the invalid tokens in word_one_hot
    word_one_hot = word_one_hot * valid_mask.unsqueeze(-1).float()

    # Sum the logits per word
    word_logits_sum = torch.matmul(word_one_hot.transpose(1, 2), token_logits)

    # Count the tokens per word
    word_counts = word_one_hot.sum(dim=1).unsqueeze(-1)

    # Avoid division by zero
    word_counts[word_counts == 0] = 1

    # Compute the average logits per word
    average_word_logits = word_logits_sum / word_counts

    return average_word_logits


class DisountDetector:
    def __init__(self, batch_size=32, num_workers=4):
        self.deberta_tok = AutoTokenizer.from_pretrained("bin/tok-microsoft-mdeberta-v3-base")
        self.rubert_tok = AutoTokenizer.from_pretrained("bin/tok-ai-forever-rubert-base")
        self.deberta = AutoModelForTokenClassification.from_pretrained("bin/microsoft-mdeberta-v3-base")
        self.rubert = AutoModelForTokenClassification.from_pretrained("bin/ai-forever-rubert-base")
        self.deberta.eval()
        self.rubert.eval()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def predict_with_nn(self, texts, model, tok, max_len=2048):
        texts = pd.DataFrame({"processed_text": texts})
        ds = MyDataset(texts, tok, max_len=max_len)
        dl = DataLoader(
            ds,
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=Collator(tok.pad_token_id),
            num_workers=self.num_workers,
        )
        preds = []
        ids = []
        with torch.no_grad():
            for batch in dl:
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                ).logits
                logits = average_per_word_logits(logits, batch["words_mapping"]).cpu().numpy()
                text_lengths = batch["words_mapping"].cpu().numpy().max(1) + 1
                mask = np.zeros((len(text_lengths), text_lengths.max()))
                for i, l in enumerate(text_lengths):
                    mask[i, :l] = 1
                mask = mask.flatten()
                logits = logits.reshape(-1, 4)
                logits = logits[mask != 0]
                preds.append(logits)
                ids.append(np.repeat(batch["text_ids"], text_lengths))
        ids = np.concatenate(ids)
        preds = np.concatenate(preds)
        preds = softmax(preds, 1)
        res = pd.DataFrame(preds, columns=["p0", "p1", "p2", "p3"])
        res["text_id"] = ids
        res["word_num"] = res.groupby("text_id").p0.cumcount()
        return res

    def predict(self, texts):
        deberta_preds = self.predict_with_nn(texts, self.deberta, self.deberta_tok)
        deberta_preds.columns = [i + "_deberta" if i.startswith("p") else i for i in deberta_preds.columns]
        rubert_preds = self.predict_with_nn(texts, self.rubert, self.rubert_tok, max_len=512)
        rubert_preds.columns = [i + "_rubert" if i.startswith("p") else i for i in rubert_preds.columns]
        res = deberta_preds.merge(rubert_preds, on=["text_id", "word_num"], how="left")
        w = 0.47
        p = (
            w * res[["p%d_rubert" % i for i in range(4)]].values
            + (1 - w) * res[["p%d_deberta" % i for i in range(4)]].values
        )
        res["label"] = p.argmax(1)
        res["label"] = res.label.map({0: "O", 1: "B-discount", 2: "B-value", 3: "I-value"})
        return res.groupby("text_id").label.agg(list).to_list()


detector = DisountDetector()

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    # Ensure 'texts' is present in the JSON data
    if not data or "texts" not in data:
        return jsonify({"error": "Invalid input, 'texts' key is required"}), 400
    texts = data["texts"]
    # Ensure the texts is a list of strings
    if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
        return jsonify({"error": "'texts' must be a list of strings"}), 400
    return jsonify(detector.predict(texts))


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
