#!/bin/sh
# create virtual env
conda create --solver=libmamba -n lct -c nvidia -c pytorch -c conda-forge python=3.10 pytorch pytorch-cuda=12.1 pandas pytorch-lightning hydra-core transformers scikit-learn scipy sentencepiece protobuf
conda activate lct
# train neural nets
python train_nn/train.py model.name='ai-forever/ruBert-base' data.fold=-1  data.max_len=512 trainer.max_epochs=6   model.learning_rate=6e-6 trainer.accumulate_grad_batches=2 +trainer.limit_val_batches=0 +trainer.num_sanity_val_steps=0 
python train_nn/train.py model.name='microsoft/mdeberta-v3-base' data.fold=-1 trainer.max_epochs=4   model.learning_rate=6e-6 trainer.accumulate_grad_batches=4 +trainer.limit_val_batches=0 +trainer.num_sanity_val_steps=0
# blend neural net predictions and create submission
python blend.py
# build and run docker image with API
docker build -t lct-samolet-ner .
docker run -p 5000:5000 -d lct-samolet-ner
