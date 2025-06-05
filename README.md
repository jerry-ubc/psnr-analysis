# PSNR Image Quality Analysis

This project provides tools for analyzing image quality using PSNR (Peak Signal-to-Noise Ratio) metrics across different resolutions.

## Setup

1. Create a Python virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```


## Directory Structure

- `images/`: TODO: might disinclude from final, revisit this
- `results/`: PSNR benchmarking calculation results
- `data/`: (jpg) Frames extracted from 2 equivalent videos w/ different resolution
- `videos/`: Downloaded videos to extract frames from
- `verify_frames/`: Frames for dev to verify the extraction app is working as intended