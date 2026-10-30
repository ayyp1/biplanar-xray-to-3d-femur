# biplanar-xray-to-3d-femur

Recovering the femoral mechanical axis (hip centre → knee centre) from two synthetic X-rays (AP + lateral)
via 3D femur reconstruction. Compares AttentionUNet (1.5M params) and SwinUNETR (62M) on HFValid
(101 CTs, 5-fold CV) with an external test on VSD.

**Result:** with the same training recipe, both models recover the axis to about 0.5° on HFValid and 0.6–0.7° on
VSD, so the small CNN matches the transformer. Full method and results: [report](report.pdf).

## Inference

```bash
pip install -e .
python inference/predict.py --ap ap.png --lat lat.png --side L
```

Saves the 3D femur mask, hip/knee centres and axis, and a preview image. Weights download automatically from
[Hugging Face](https://huggingface.co/ayyp/biplanar-xray-to-3d-femur). Inputs must be in the project's DRR format
(AP 256×96, LAT 256×128 pixels at 2 mm, hip at the top).

## Training

```bash
python preprocessing/preprocess_hfvalid.py /path/to/HFValid_Collection_v3 preprocessed
cp configs/split.json preprocessed/
python training/train.py --config configs/attnC.yaml --data preprocessed
python training/train.py --config configs/swinC.yaml --data preprocessed
```

Trains the 5 cross-validation folds of each model; `--fold 0` trains one. All hyperparameters (architecture,
epochs, learning rate, augmentation) are in `configs/*.yaml`. Checkpoints go to `checkpoints/` and an
interrupted run resumes from the last epoch. Both models together take about 15 GPU-hours on T4s.

## Evaluation

```bash
python evaluation/evaluate.py --data preprocessed --ckpt checkpoints
python evaluation/robustness.py --data preprocessed --ckpt checkpoints
python preprocessing/prepare_vsd.py --out vsd
python evaluation/evaluate_vsd.py --vsd vsd --data preprocessed --ckpt checkpoints
python evaluation/compare.py results/hfvalid_cv.csv --a attnC --b swinC
```

Cross-validation, robustness and VSD external test write per-case CSVs to `results/`; `compare.py` runs the
paired statistics. `bash run_pipeline.sh /path/to/HFValid_Collection_v3` runs everything from preprocessing
to statistics.

## Data

- HFValid: [amsacta.unibo.it/8431](https://amsacta.unibo.it/8431/) (CC BY 4.0)
- VSD: [doi.org/10.5281/zenodo.8302448](https://doi.org/10.5281/zenodo.8302448) (CC BY-NC-SA 4.0), with expert
  landmarks from [VSDFullBodyBoneModels](https://github.com/MCM-Fischer/VSDFullBodyBoneModels);
  both downloaded by `preprocessing/prepare_vsd.py`
