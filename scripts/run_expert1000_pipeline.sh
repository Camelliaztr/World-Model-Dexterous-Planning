#!/usr/bin/env bash
set -e

CONFIG_BC=configs/adroit_relocate_expert1000.yaml
CONFIG_WM=configs/adroit_relocate_expert1000_wm.yaml

echo "Train BC policy"
python -m src.train_policy --config ${CONFIG_BC}

echo "Evaluate BC"
python -m src.evaluate_bc --config ${CONFIG_BC}

echo "Collect BC rollouts"
python -m src.collect_bc_rollouts --config ${CONFIG_BC} --episodes 1000

echo "Split BC rollouts"
python -m src.split_dataset \
  --src_dir data/rollouts_bc_expert1000_raw \
  --out_root data/splits/rollouts_bc_expert1000 \
  --prefix rollouts_bc_expert1000 \
  --seed 42

echo "Train World Model"
python -m src.train_world --config ${CONFIG_WM}

echo "Train Value and Q Models"
python -m src.train_value_q --config ${CONFIG_WM}

echo "Evaluate Planning-WV"
python -m src.evaluate_planning --config ${CONFIG_WM}

echo "Done."
