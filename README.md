# Cross-Scale-Self-Supervised-Blind-Image-Deblurring-via-Implicit-Neural-Representation

This repository contains the official pytorch implementation for "Cross-Scale Self-Supervised Blind Image Deblurring via Implicit Neural Representation", NIPS24

## Overview

The goal of this project is to perform image deblurring. By running `demo_deblur`, you can execute the deblurring process, generating a restored image and an estimated kernel. The deblurred results and the estimated kernel will automatically be saved in the `results` folder for further analysis and verification.

## Prerequisites

- **Python**: Version 3.7
- **PyTorch**: Version 1.20 or higher
- **Requirements**: 
  - `opencv-python`
  - `tqdm`
- **MATLAB**: Required for computing evaluation metrics

## Instructions

1. **Set Up Input Images**: Place the blurred images in the specified input folder.
2. **Run the Demo**: Run the following command to start the deblurring process:

   ```bash
   python demo_deblur.py
   ```

3. **Check Results**: After running, the deblurred images and estimated kernels will be saved in the `results` folder.

## Results Folder

- `results/`: Contains the estimated kernels and the deblurred image results.


## Evaluation Metrics

For calculating PSNR and SSIM, please refer to the implementation in **SelfDeblur**: [https://github.com/csdwren/SelfDeblur](https://github.com/csdwren/SelfDeblur).


