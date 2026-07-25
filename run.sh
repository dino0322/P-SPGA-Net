#!/usr/bin/env bash

mkdir -p logs

DATA_ROOT="${DATA_ROOT:-./data}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-2}"

echo "NOW TRAINING DOWNY_MILDEW"
python -u main_proposal.py --class_name "downy_mildew" --data_root "${DATA_ROOT}" --batch_size "${BATCH_SIZE}" --num_workers "${NUM_WORKERS}" > "logs/downy_mildew.log" 2>&1


echo "NOW TRAINING POWDERY_MILDEW"
python -u main_proposal.py --class_name "powdery_mildew" --data_root "${DATA_ROOT}" --batch_size "${BATCH_SIZE}" --num_workers "${NUM_WORKERS}" > "logs/powdery_mildew.log" 2>&1


echo "NOW TRAINING SIMILAR_DOWNY_MILDEW"
python -u main_proposal.py --class_name "sim_downy_mildew" --data_root "${DATA_ROOT}" --batch_size "${BATCH_SIZE}" --num_workers "${NUM_WORKERS}" > "logs/simdowny_mildew.log" 2>&1


echo "NOW TRAINING SIMILAR_POWDERY_MILDEW"
python -u main_proposal.py --class_name "sim_powdery_mildew" --data_root "${DATA_ROOT}" --batch_size "${BATCH_SIZE}" --num_workers "${NUM_WORKERS}" > "logs/simpowdery_mildew.log" 2>&1

echo "Finished. Check Your Results"
