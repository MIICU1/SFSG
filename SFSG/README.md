# SFSG

**Stepwise Generation Guided by Stress-Field Modeling for Retinal Layer Segmentation**

SFSG generates OCT image–mask pairs for retinal layer segmentation.

## Datasets

| Dataset | Download |
| --- | --- |
| OCTA-500 | [IEEE DataPort](https://ieee-dataport.org/open-access/octa-500) (6 mm subset) |
| AROI | [OSF dataset](https://doi.org/10.17605/OSF.IO/5WYR3) |
| Duke DME | [Duke University dataset](https://people.duke.edu/~sf59/Chiu_BOE_2014_dataset.htm) |
| HCMS | [Johns Hopkins IACL dataset](https://medic.rad.jhmi.edu/index.php?title=OCT_Data) |

## Setup and data preparation

Install dependencies with `pip install -r requirements.txt`. For configuration details, see [`configs/`](configs/) and the script arguments.

**The first three steps require retinal boundary labels in `.npy` format.** For OCTA-6mm, store one `8 × 400` array per B-scan as `<volume_id>/<slice_id>.npy`. The first and last rows store the top (0) and bottom (640) image bounds, respectively.

```bash
python convert_layer_masks_to_npy.py --input-root data/masks --output-root data/layer_npy
```

Use `--encoding grayscale` for grayscale masks or `--input-format mat` for native OCTA-500 annotations.

## Training

Run the steps in order from the repository root.

| Step | Script | Inputs and outputs |
| --- | --- | --- |
| 1. Prepare ILM training data | [`prepare_ilm_train210.py`](prepare_ilm_train210.py) | Use the original boundary NPY files, training split, and condition JSON. Outputs are `GT_Layers_npy/` and `rise_5_train210.json`. |
| 2. Train the ILM generator | [`train_stage1.py`](train_stage1.py) | Use the prepared NPY files and condition JSON from Step 1. |
| 3. Train the retinal layer generator | [`train_stage2.py`](train_stage2.py) or [`train_stage2_sdl.py`](train_stage2_sdl.py) | Use the original boundary NPY files and their condition JSON. |
| 4. Train the OCT image generator | [`train_VQVAE.py`](train_VQVAE.py) → [`train_MLDM.py`](train_MLDM.py) | Train VQ-VAE first and pass its checkpoint to MLDM with `--first_stage_ckpt`. |

## Segmentation and inference

- Segmentation: `train_unet.py` or `train_unet_dual.py`, with data paths set in the YAML.
- Inference: `test_stage1.py`, `test_stage2.py`, `test_VQVAE.py`, and `test_MLDM.py`. For Stage 1, use `--architecture stage1` for current checkpoints or `--architecture legacy` for older checkpoints.
