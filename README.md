# ShipPointYOLO: Ship Detection and Description based on Point Coordinates in SAR Images
A point detection (and description) method based on a modified You Only Look Once (YOLO) model, designed to detect vessels in Synthetic Aperture Radar (SAR) satellite images.

## 1. Setup

### 1.1 Data sets
Instructions on how to install datasets used in the experiments are found in the [datasets/README.md](./datasets/README.md) file.

### 1.2 Build wheel module for yolo_lib
* Install wheel-related packages
```bash
pip install wheel
python -m pip install --upgrade pip
pip install check-wheel-contents
```
* Build python wheel file
Run the following command. Output is found in `dist/` 
```bash
python setup.py bdist_wheel 
```
