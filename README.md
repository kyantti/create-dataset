# Fig Detection and Hyperspectral Cropping Dataset Creation

This project provides a pipeline for detecting figs in images, segmenting them using state-of-the-art models (Grounding DINO and SAM2), and generating COCO-format annotations and cropped hyperspectral cubes for further analysis.

## Overview

The workflow consists of:

1. **Detection & Segmentation**: Uses [Grounding DINO](https://arxiv.org/abs/2303.05499) and [SAM2](https://arxiv.org/abs/2408.00714) to detect and segment figs in images. Results are saved in COCO format and visualized.
2. **Hyperspectral Cropping**: For each detected fig, extracts a radiometrically corrected hyperspectral subcube using the segmentation mask and bounding box.
3. **Dataset Organization**: Outputs are organized by class and date, with metadata extracted from filenames.

## Main Scripts

- `detect_and_segment_coco_annotations.py`: Runs detection and segmentation, saves annotated images and COCO-format JSONs.
- `create_cropped_cubes.py`: Loads COCO annotations and hyperspectral data, applies radiometric correction, and saves cropped cubes for each fig instance.
- `file_processing.py`: Utilities for parsing filenames, organizing images by class/date, and extracting metadata.

## Requirements

- Python 3.10+
- CUDA-enabled GPU recommended
- [PyTorch](https://pytorch.org/)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [SAM2](https://ai.meta.com/sam2)
- [Supervision](https://github.com/roboflow/supervision)
- [pycocotools](https://github.com/cocodataset/cocoapi)
- [spectral](https://github.com/spectralpython/spectral)
- [Pillow](https://python-pillow.org/)

See `Grounded-SAM-2/README.md` for installation and model checkpoint details.

## Usage

1. **Run detection and segmentation:**
   ```bash
   uv run python detect_and_segment_coco_annotations.py
   ```
   - Outputs COCO JSONs and annotated images in `outputs/segm-masks-.../C*/`

2. **Create cropped hyperspectral cubes:**
   ```bash
   uv run python create_cropped_cubes.py
   ```
   - Outputs cropped cubes in `outputs/cropped-hypercubes/C*/`


## Example Detection & Segmentation

Below is an example of fig detection and segmentation using Grounding DINO and SAM2:

![Example Detection](outputs/segm-masks-2023-july-17-28/C0/Fx10_20230717_Riego_C0_2023-07-17_09-02-21_annotated_with_mask.jpg)

## Data Structure

- `outputs/segm-masks-.../C*/`: COCO annotations and visualizations per class
- `outputs/cropped-hypercubes/C*/`: Cropped hyperspectral cubes per class

## Customization

- Update paths and parameters in the scripts as needed for your dataset.
- Supports multiple classes (C0–C3) and automatic metadata extraction.

## Citation

If you use this code or dataset, please cite:

```
Pablo Setrakian Bearzotti, "Fig detection dataset using SAM2 and Grounding DINO", 2025.
```

## License

See `Grounded-SAM-2/LICENSE` and individual model licenses for details.
