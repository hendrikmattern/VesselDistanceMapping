# Vessel Distance Mapping (VDM)

## Requirements
Please install (with pip/venv):
- scikit-image (0.17.2; 0.18.1 problems on win10)
- scikit-learn (for clustering/classification) 
- pandas (for organizing data in the analysis)
- nibabel (nii-file i/o)
- h5py (h5-file i/o)
- Pillow (image i/o)

The remaining requirements (ie numpy) are installed automatically by the above mentioned packages. The full list is saved in requirements.txt (using  $pip freeze > requirements.txt).

Pingouin and Seaborn is optionally and only used in some projects, but not the VDM framework itself.

## Data handling (WIP)
image -> segmentation -> vdm -> analysis (histogram/metrics)

rois in which the analysis should be done, should be a separate nii-file