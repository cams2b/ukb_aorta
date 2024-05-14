# Aortic geometric phenotype extraction

* This repository is designed for extracting aortic geometric phenotypes (AGPs) from the UK Biobank.
* We provide pretrained weights to allow other researchers to implement this pipeline.
* Our AGP extraction can be used on any voxel representation of an aorta

Code for reproducing aortic geometric phenotypes in the paper [Three-dimensional aortic geometry: clinical correlates, prognostic value and genetic
architecture ](https://www.biorxiv.org/content/10.1101/2024.05.09.593413v1). If you use the code, please cite our paper.


## Dependencies
This library mainly depends on PyTorch and VMTK. Given that VMTK can be tempermental we encourage you to use a conda environment
- Python 3.9
- Torch 2.1.2
- vmtk 1.5.0
- vtk 9.1.0
- SimpleITK 2.3.1
