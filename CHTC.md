# CHTC Training Guide

This branch adds a CHTC-oriented training flow that:

1. Uses the existing training configuration (`GlowModel`, `glow-ckpt.pth`, preloaded CSV).
2. Downloads APOD images directly on the execute node.
3. Trains using local preloaded images.

## Files added for CHTC

- `htc/train.sub`: HTCondor submit description.
- `htc/train_sweep.sub`: HTCondor sweep submit file (multiple configs).
- `htc/run_train_htc.sh`: Job wrapper that downloads data and starts training.
- `scripts/download_preloaded_images.py`: Downloads/normalizes images to `images/<img_index>.png`.

## Run flow

1. Single run submit:

```bash
condor_submit htc/train.sub
```

2. Sweep submit (all requested configs + larger sample runs):

```bash
condor_submit htc/train_sweep.sub
```

3. Each job does:
   - reads `Data/apod_preloaded_dataset.csv`
   - downloads images using `best_url` into `images/`
   - runs `train.py` with current working defaults
   - writes checkpoints to `checkpoints/`:
     - `glow-ckpt.pth`
     - `llm-ckpt.pth`
   - for sweeps, checkpoint names are run-specific:
     - `<run_name>-glow-ckpt.pth`
     - `<run_name>-llm-ckpt.pth`

## Where to configure hyperparameters

- `htc/train_sweep.sub`:
  - edit the table under `queue ... from (...)`.
  - each row is one run with:
    - `run_name, epochs, image_size, n_levels, n_steps, samples, lr, cond_dim, batch_size`
- `htc/train.sub`:
  - single-run baseline args.

## Notes

- `train.py` now supports CLI args for:
  - data locations: `data_csv`, `images_dir`, `checkpoint_dir`
  - model/train params: `epochs`, `image_size`, `n_levels`, `n_steps`, `samples`, `lr`, `cond_dim`, `batch_size`
  - checkpoint names: `flow_ckpt_name`, `llm_ckpt_name`
- `Data/dataset.py` still defaults to local preloaded images and now supports `.png`, `.jpg`, and `.jpeg` fallback.
- If your CHTC environment does not already include dependencies, install them in the job wrapper before running `train.py` (for example with a prebuilt environment or container).
