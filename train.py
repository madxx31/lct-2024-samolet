import hydra
from data_module import MyDataModule
from pytorch_lightning import Trainer, seed_everything
from model import MyModel
import logging
import numpy as np
from transformers import AutoTokenizer
from datetime import datetime
import os
from pytorch_lightning.loggers import WandbLogger

os.environ["WANDB_API_KEY"] = "04cc10900943b2d11063303c05b967e980794041"


logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")


@hydra.main(config_path=".", config_name="config", version_base="1.2")
def main(cfg) -> None:
    seed_everything(17)
    wandb_logger = WandbLogger(project="samolet")
    wandb_logger.experiment.config.update(dict(cfg))
    dm = MyDataModule(cfg)
    dm.setup()
    model = MyModel(cfg)
    trainer = Trainer(**cfg.trainer, logger=wandb_logger)
    trainer.fit(model, dm)
    # ckpt_name = cfg.model.name.lower().replace("/", "-") + datetime.now().strftime("_%Y%m%d_%H%M%S")
    # model.model.save_pretrained("bin/" + ckpt_name)
    # AutoTokenizer.from_pretrained(cfg.model.name).save_pretrained("bin/tok-" + ckpt_name)
    # np.save("preds/" + ckpt_name + ".npy", np.concatenate(model.preds[0]))


if __name__ == "__main__":
    main()
