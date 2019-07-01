#!/bin/bash
dataset="ag_news"

python3 nlp_torch_run.py --GPU=0 --h=1.0 --BN=1 --depth=49 --batch_size=64 --num_epochs=15 --database=${dataset} --noise_level=0.0 --start_lr=0.1 --seed=7
python3 nlp_torch_run.py --GPU=0 --h=0.1 --BN=1 --depth=49 --batch_size=64 --num_epochs=15 --database=${dataset} --noise_level=0.0 --start_lr=0.1 --seed=7

echo "ALL DONE!"