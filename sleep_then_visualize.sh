#!/bin/bash

echo "===== Starting overnight model runs ====="

CONFIGS=(
  "-GlobalLocal"
  "ConstantQ-Only"
  "Scalogram-Only"
  "-CausalAtt"
)

for cfg in "${CONFIGS[@]}"; do
  echo "===== Running config: $cfg ====="
  python spectro_ablation_experiments_single.py \
    --cfg_name="$cfg" \
    --data_root ./Voice \
    --epochs 15 \
    --seed 0
done

echo "===== Running GradCAM batch visualization ====="
python batch_gradcam_spectro.py

echo "===== Generating summary figure ====="
python gradcam_summary_figure.py

echo "✅ All done. Sweet dreams!"
