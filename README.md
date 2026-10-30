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

## Reproduce

```bash
bash run_pipeline.sh /path/to/HFValid_Collection_v3
```

Or step by step: `preprocessing/` → `training/train.py --config configs/attnC.yaml` → `evaluation/`.
Hyperparameters are in `configs/`.

## Data

- HFValid: amsacta.unibo.it/8431 (CC BY 4.0)
- VSD: doi.org/10.5281/zenodo.8302448 (CC BY-NC-SA 4.0), downloaded by `preprocessing/prepare_vsd.py`
