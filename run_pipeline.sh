#!/usr/bin/env bash
# Full pipeline: HFValid -> preprocessing -> 5-fold training (attnC, swinC) -> evaluation -> VSD external test.
# Usage: pip install -e . && bash run_pipeline.sh /path/to/HFValid_Collection_v3
# Steps whose outputs exist are skipped; training resumes from the last saved epoch.
set -euo pipefail

HFVALID=${1:?path to HFValid_Collection_v3}
DATA=${DATA:-preprocessed}
CKPT=${CKPT:-checkpoints}
VSD=${VSD:-vsd}
RESULTS=${RESULTS:-results}
R=$(dirname "$0")

[ -f "$DATA/dataset.csv" ] || python "$R/preprocessing/preprocess_hfvalid.py" "$HFVALID" "$DATA"
[ -f "$DATA/split.json" ] || cp "$R/configs/split.json" "$DATA/split.json"

# with two GPUs, run the two models side by side instead:
#   CUDA_VISIBLE_DEVICES=0 python training/train.py --config configs/swinC.yaml ... &
#   CUDA_VISIBLE_DEVICES=1 python training/train.py --config configs/attnC.yaml ...
for m in swinC attnC; do
  python "$R/training/train.py" --config "$R/configs/$m.yaml" --data "$DATA" --out "$CKPT"
done

python "$R/evaluation/evaluate.py"   --data "$DATA" --ckpt "$CKPT" --out "$RESULTS/hfvalid_cv.csv"
python "$R/evaluation/robustness.py" --data "$DATA" --ckpt "$CKPT" --out "$RESULTS/robustness.csv"
python "$R/evaluation/compare.py" "$RESULTS/hfvalid_cv.csv" | tee "$RESULTS/compare_hfvalid.txt"

python "$R/preprocessing/prepare_vsd.py" --out "$VSD"
python "$R/evaluation/evaluate_vsd.py" --vsd "$VSD" --data "$DATA" --ckpt "$CKPT" --out "$RESULTS/vsd.csv"
python "$R/evaluation/compare.py" "$RESULTS/vsd.csv" --key femur \
  --metrics dice hd95 asd nsd2 hip_vs_gtmask_mm angle_vs_gtmask_deg hip_mm angle_deg | tee "$RESULTS/compare_vsd.txt"
