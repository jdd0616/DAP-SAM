# DAP-SAM

Minimal code for the RETOUCH dual-branch training experiment.

## Files

- `train_dual_branch.py`: training entrypoint.
- `trainer_dual_branch.py`: dual-branch optimization and validation.
- `datasets/dataset_RETOUCH.py`: dataset loading and the required augmentation.
- `segment_anything/`: customized dual-mask SAM definition.

The repository excludes datasets, checkpoints, predictions, figures, and logs.

## Dataset layout

```text
retouch/
├── images/GT_*.png
├── masks/<matching-mask-name>.png
└── reconstructions/Out_*.png
```

## Training

```bash
python train_dual_branch.py \
  --data-root /path/to/retouch \
  --checkpoint /path/to/sam_vit_b.pth
```

Install dependencies with `pip install -r requirements.txt`.
