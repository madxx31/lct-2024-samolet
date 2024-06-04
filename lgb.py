import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import ast
from sklearn.metrics import f1_score


def get_strata(x):
    if x == {}:
        return 0
    elif len(x) == 3:
        return 1
    elif "B-discount" in x and "B-value" not in x and "I-value" not in x:
        return 2
    else:
        return 3


tr = pd.read_csv("train_data.csv")
tr.target_labels_positions = tr.target_labels_positions.apply(lambda x: ast.literal_eval(x))
tr["strata"] = tr.target_labels_positions.apply(get_strata)
skf = StratifiedKFold(n_splits=5, random_state=17, shuffle=True)
tr["fold"] = 0
for fold, (trx, tex) in enumerate(skf.split(tr.index, tr.strata)):
    tr.loc[tex, "fold"] = fold
tr["text_id"] = np.arange(len(tr))

ents = []
for rid, row in tr.iterrows():
    for k, v in row.target_labels_positions.items():
        for ind in v:
            ents.append({"label": k, "text_id": row.text_id, "word_num": ind})
ents = pd.DataFrame(ents)
tr.drop(columns=["target_labels_positions"], inplace=True)

tr["processed_text"] = tr["processed_text"].str.split(" ")
tr = tr.explode("processed_text")
tr["word_num"] = tr.groupby("text_id").processed_text.cumcount()
tr = tr.merge(ents, on=["text_id", "word_num"], how="left")
tr.rename(columns={"processed_text": "word"}, inplace=True)
tr.label.fillna("O", inplace=True)
print(tr)
print(f1_score(tr.label == "B-discount", tr.word.str.startswith("скидк")))
print(tr[tr.word.str.startswith("скидк")].label.value_counts(normalize=True))
print(tr[tr.word.str.startswith("скидк")])
