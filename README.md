## SugarRush

SugarRush extends the work done in GeqShift (arXiv:2311.12657), an E(3) equivariant graph neural network for carbohydrate nuclear magnetic resonance chemical shift prediction. SugarRush expands the dataset to improve generalization, optimizes the memory consumption of the model to enable more accessibility by lowering the need for specialized compute resources, and provides an example web API and frontend to allow for easier use by researchers and show how integration into web services might look like. SugarRush makes use of data from GlycoNMR and CSDB. The example web API and frontend uses 3Dmol.js: https://doi.org/10.1093/bioinformatics/btu829.

## Usage
SugarRush was tested on x86_64 Linux using the following software:
- Python 3.12.1
- PyTorch 2.7.1 with NVIDIA CUDA 12.8 and PyTorch 2.7.1 with AMD ROCm 6.3
- PyTorch Geometric 2.6.1 (CUDA build and ROCm external build)
- PyTorch Scatter 2.1.2 (CUDA build and ROCm external build)
- PyTorch Cluster 1.6.3 (CUDA build and ROCm external build)
- e3nn 0.5.6
- RDKit 2025.3.5
- Flask 3.1.1
- Gunicorn 23.0.0

A Python virtual environment is the easiest method to use the software. Setup depends on operating system and installed software, with a simple example being:
```
python3 -m venv .venv
source .venv/bin/activate
```

`pip`, `conda`, or another preferred tool can be used to install the packages into a virtual environment. PyTorch and PyTorch Geometric usually require specific install methods depending on system (NVIDIA CUDA or AMD ROCm). See their respective documentation for instructions relevant to your specific system.

Running examples:
```
# Generate datasets
python3 data/create_data.py

# Train and test SugarRush model on Mixed dataset fold0
python3 main-opt.py --train_path datasets-mixed-fold0/train_data_13C.pickle --test_path_mo datasets-mixed-fold0/test_data_13C_mo.pickle --test_path_di datasets-mixed-fold0/test_data_13C_di.pickle --test_path_tri datasets-mixed-fold0/test_data_13C_tri.pickle --test_path_tetra datasets-mixed-fold0/test_data_13C_tetra.pickle --test_path_oligo datasets-mixed-fold0/test_data_13C_oligo.pickle --test_path_poly datasets-mixed-fold0/test_data_13C_poly.pickle

# Run example web API and frontend (requires train_data_13C.pickle, train_data_1H.pickle, checkpoint-13C-casper.pkl, checkpoint-1H-casper.pkl, which can be renamed from the files generated in the above steps)
gunicorn -w 1 -b 0.0.0.0:5000 web-example:app
```
