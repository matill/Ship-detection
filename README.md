# End-to-end vessel detection experiments
Code for capstone and master projects' experiments

## Setup
### Install wheel-related packages
```bash
conda install wheel
python -m pip install --upgrade pip
pip install check-wheel-contents
```

### Build python wheel file
Output found in ./dist/ by running command
```bash
python setup.py bdist_wheel 
```

## Build (and run) inline build app
```bash
python build_lib_file.py
python databricks_composed.py
```
