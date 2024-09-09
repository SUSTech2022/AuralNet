# Description
This repository provides the dataset and demo code for **AuralNet**.

## Files
- `model.py`: Demo code for AuralNet, implemented with TensorFlow 2.5.0.  
  - Requires [mat73](https://pypi.org/project/mat73/) for loading MATLAB v7.3 files.
- `./model_save/MCT_weights.h5`: Pretrained model weights.
- `./dataset`: Directory containing extracted features for training, validation, and testing datasets.  
  - The data is in HDF5 format, converted from MATLAB v7.3 files.  
  - Each file contains four columns: gammatone coefficients for the left ear, right ear, cross-correlation values, and ground truth labels. For more details, please refer to our paper.

## Training and Testing Data
The training, validation, and testing datasets can be [downloaded here](link). Please place the extracted data in the `./dataset` folder.

## Testing
To test the pretrained MCT model, run the following command:

```
python test.py
```

## Train a New Model
To train a new MCT model from scratch, use the following command:

```
python train.py
```



