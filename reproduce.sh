#!/bin/bash
# STEK 564 Term Project - Reproducibility Script
# Kullanım: bash reproduce.sh configs/eval_standard.yaml "0,1,2"

CONFIG=${1:-"configs/eval_standard.yaml"}
SEEDS=${2:-"0,1,2"}

python code/run_all.py --config "$CONFIG" --seeds "$SEEDS"