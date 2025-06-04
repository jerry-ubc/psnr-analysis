# PSNR Image Quality Analysis

This project provides tools for analyzing image quality using PSNR (Peak Signal-to-Noise Ratio) metrics across different resolutions.

## Features

- Image resolution verification and standardization
- Batch PSNR calculations
- Support for 2K, 4K, and 8K resolutions
- Performance metrics and statistics

## Setup

1. Create a Python virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate.bat  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create required directories:
```bash
mkdir -p images/{2k,2k-original,2k-uniform,4k,4k-original,4k-uniform,8k,8k-original,8k-uniform,other}
mkdir -p {data,temp,results}
```

## Usage

1. Place source images in their respective resolution directories (2k/, 4k/, 8k/)
2. Run the resolution verification and standardization:
```bash
python resize_and_verify.py
```

3. Run PSNR calculations:
```bash
python calculate_psnr.py
```

## Directory Structure

- `images/`: Contains all image directories
  - `{2k,4k,8k}/`: Source images
  - `{2k,4k,8k}-original/`: Original-sized images that meet minimum resolution
  - `{2k,4k,8k}-uniform/`: Standardized images for PSNR calculation
  - `other/`: For miscellaneous image comparisons
- `results/`: PSNR calculation results
- `temp/`: Temporary files
- `data/`: Additional data files 