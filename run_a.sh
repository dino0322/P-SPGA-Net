#!/usr/bin/env bash

mkdir -p logs

DATA_ROOT="${DATA_ROOT:-./data}"
ABLATIONS="${ABLATIONS:-B0,A0,A1,A2,A3,A5}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-2}"

run_ablation() {
  local class_name="$1"
  echo "NOW ABLATION TRAINING ${class_name} (${ABLATIONS})"
  python -u main_ablation.py \
    --class_name "${class_name}" \
    --data_root "${DATA_ROOT}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --ablation "${ABLATIONS}" > "logs/${class_name}_ablation.log" 2>&1
}

# Active first-half set
run_ablation "downy_mildew"
run_ablation "powdery_mildew"
run_ablation "sim_downy_mildew"
run_ablation "sim_powdery_mildew"
run_ablation "cherry_including_sour_powdery_mildew"
run_ablation "apple_apple_scab"
run_ablation "apple_black_rot"
run_ablation "corn_maize_common_rust"
run_ablation "corn_maize_northern_leaf_blight"
run_ablation "grape_black_rot"
run_ablation "potato_early_blight"
run_ablation "potato_late_blight"
run_ablation "tomato_early_blight"
run_ablation "tomato_late_blight"

# Remaining classes for full run later
# run_ablation "apple_cedar_apple_rust"
# run_ablation "corn_maize_cercospora_leaf_spot_gray_leaf_spot"
# run_ablation "grape_esca_black_measles"
# run_ablation "grape_leaf_blight_isariopsis_leaf_spot"
# run_ablation "peach_bacterial_spot"
# run_ablation "pepper_bell_bacterial_spot"
# run_ablation "strawberry_leaf_scorch"
# run_ablation "tomato_bacterial_spot"
# run_ablation "tomato_leaf_mold"
# run_ablation "tomato_septoria_leaf_spot"
# run_ablation "tomato_spider_mites_two_spotted_spider_mite"
# run_ablation "tomato_target_spot"
# run_ablation "tomato_tomato_mosaic_virus"
# run_ablation "tomato_tomato_yellow_leaf_curl_virus"

echo "Finished. Check Your Ablation Results"
