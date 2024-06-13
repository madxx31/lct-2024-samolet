import pandas as pd
import numpy as np
from scipy.special import softmax


te = pd.read_csv("data/gt_test.csv")
te["text_id"] = np.arange(len(te))
te["processed_text"] = te["processed_text"].str.split(" ")
te = te.explode("processed_text")
te["word_num"] = te.groupby("text_id").processed_text.cumcount()

preds = pd.read_parquet("preds/microsoft-mdeberta-v3-base_-1.pq")
preds["word_num"] = preds.groupby("text_id").p0.cumcount()
preds[["p0", "p1", "p2", "p3"]] = softmax(preds[["p0", "p1", "p2", "p3"]], 1)
preds.columns = ["text_id", "p0_deberta", "p1_deberta", "p2_deberta", "p3_deberta", "word_num"]
te = te.merge(preds, on=["text_id", "word_num"], how="left")

preds = pd.read_parquet("preds/ai-forever-rubert-base_-1.pq")
preds["word_num"] = preds.groupby("text_id").p0.cumcount()
preds[["p0", "p1", "p2", "p3"]] = softmax(preds[["p0", "p1", "p2", "p3"]], 1)
preds.columns = ["text_id", "p0_rubert", "p1_rubert", "p2_rubert", "p3_rubert", "word_num"]
te = te.merge(preds, on=["text_id", "word_num"], how="left")

w = 0.47
te["p0"] = w * te.p0_rubert + (1 - w) * te.p0_deberta
te["p1"] = w * te.p1_rubert + (1 - w) * te.p1_deberta
te["p2"] = w * te.p2_rubert + (1 - w) * te.p2_deberta
te["p3"] = w * te.p3_rubert + (1 - w) * te.p3_deberta
te["label"] = te[["p0", "p1", "p2", "p3"]].values.argmax(1)
te["label"] = te.label.map({0: "O", 1: "B-discount", 2: "B-value", 3: "I-value"})
subm = te.groupby("text_id").agg({"processed_text": lambda x: " ".join(x), "label": list})
subm.to_csv("submission.csv", index=False)


# print(te[te.label != "O"])
