import hydra
from data_module import MyDataModule
from pytorch_lightning import Trainer, seed_everything
from model import MyModel
import logging
from transformers import AutoTokenizer
import pandas as pd

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")


@hydra.main(config_path=".", config_name="config", version_base="1.2")
def main(cfg) -> None:
    seed_everything(17)
    dm = MyDataModule(cfg)
    dm.setup()
    model = MyModel(cfg)
    trainer = Trainer(**cfg.trainer)
    trainer.fit(model, dm)

    ckpt_name = cfg.model.name.lower().replace("/", "-")
    # save model if training on all data
    if cfg.data.fold == -1:
        model.model.save_pretrained("bin/" + ckpt_name)
        AutoTokenizer.from_pretrained(cfg.model.name).save_pretrained("bin/tok-" + ckpt_name)
        trainer.predict(model, dm.predict_dataloader())
    # save oof or test predictions
    res = pd.DataFrame(model.preds, columns=["p0", "p1", "p2", "p3"])
    res["text_id"] = model.text_ids
    res.to_parquet(f"preds/{ckpt_name}_{cfg.data.fold}.pq")


if __name__ == "__main__":
    main()
