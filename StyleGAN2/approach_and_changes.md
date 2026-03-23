# StyleGAN2 — CIFAR-10 Integration: Approach & Changes

## Overview

The original StyleGAN2 code loads training images from a **custom LMDB database** (e.g., `dataset/pandalmdb`), which requires running `prepare_data.py` to convert raw images. This has been changed to use **PyTorch's built-in `torchvision.datasets.CIFAR10`**, which automatically downloads and loads the CIFAR-10 dataset (50,000 training images, 32×32 pixels, 10 classes).

---

## What Changed

### 1. `dataset.py` — New `CIFAR10Dataset` class

**Added** a new dataset class that:
- Uses `torchvision.datasets.CIFAR10` to auto-download CIFAR-10 on first run
- Applies `Resize → RandomHorizontalFlip → ToTensor → Normalize` transforms
- Returns **only images** (drops class labels) to match the existing training interface
- Supports `data_ratio` parameter (e.g., `data_ratio=0.1` uses only 10% of data)
- The old `MultiResolutionDataset` class is preserved for backward compatibility

### 2. `train.py` — Updated training pipeline

| Change | Before | After |
|--------|--------|-------|
| Dataset | LMDB (`MultiResolutionDataset`) | CIFAR-10 (`CIFAR10Dataset`) |
| `--dataset` default | _(none)_ | `"cifar10"` |
| `--size` default | `256` | `32` |
| `--batch` default | `16` | `64` |
| LMDB path (`args.path`) | `../dataset/<name>lmdb` | Removed (not needed) |
| FID reference images | `../dataset/<name>/img/` | Auto-saved to `results/<name>/real_images/` |

**New arguments:**
- `--data_root`: where CIFAR-10 is downloaded (default: `./data`)
- `--data_ratio`: fraction of training data to use, 0.0–1.0 (default: `1.0`)

**FID evaluation:** Real CIFAR-10 images are automatically saved as PNGs to `results/<dataset>/real_images/` at the start of training (one-time cost). These are used as the reference for FID score computation.

### 3. `prepare_data.py` — No longer needed

You do **not** need to run `prepare_data.py` for CIFAR-10. The dataset is handled entirely by torchvision.

---

## How to Run on Google Colab

### Step 1: Clone and install dependencies

```python
!git clone <your-repo-url>
%cd gan-test/StyleGAN2

!pip install torch torchvision tqdm pytorch-fid lmdb ninja
```

### Step 2: Train on CIFAR-10

**Basic training (no ReGAN):**
```bash
!python train.py --dataset cifar10 --size 32 --batch 64 --iter 100000 \
    --eva_iter 5000 --eva_size 5000
```

**With ReGAN (sparse/dense reconfiguration):**
```bash
!python train.py --dataset cifar10 --size 32 --batch 64 --iter 100000 \
    --eva_iter 5000 --eva_size 5000 \
    --regan --warmup_iter 5000 --g 5000 --sparsity 0.3
```

**With DiffAugmentation:**
```bash
!python train.py --dataset cifar10 --size 32 --batch 64 --iter 100000 \
    --eva_iter 5000 --eva_size 5000 --diffaug
```

**With limited data (10% of CIFAR-10):**
```bash
!python train.py --dataset cifar10 --size 32 --batch 64 --iter 50000 \
    --eva_iter 2000 --eva_size 5000 --data_ratio 0.1
```

### Step 3: Check results

- **Checkpoints:** `results/cifar10/checkpoint/`
- **Generated evaluation images:** `results/cifar10/eva/`
- **Training logs:** `results/logs/` (CSV files for g_loss, d_loss, fid_score)

---

## Key Design Decisions

1. **Image size = 32:** CIFAR-10 images are 32×32. StyleGAN2's architecture supports this natively through its channel dictionary (`{4: 512, 8: 512, 16: 512, 32: 512, ...}`). The model will have fewer layers than at 256×256, which is expected.

2. **Batch size = 64:** Since 32×32 images are much smaller than 256×256, larger batches fit easily in GPU memory. Adjust based on your Colab GPU (e.g., batch 128 on A100, batch 32 on T4 if needed).

3. **FID cached images:** Instead of pointing to a pre-existing folder of images, we save real CIFAR-10 images as PNGs once at training start. This avoids any dependency on external data directories.

4. **Backward compatibility:** The original `MultiResolutionDataset` class is kept in `dataset.py`, so LMDB-based training still works if needed.
