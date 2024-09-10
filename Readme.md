# Description
This repository provides the dataset and demo code for **AuralNet**.

## Dataset
- Download the dataset [here](https://www.kaggle.com/datasets/liuzhijie2000/auralnet-dataset). Place the downloaded files in the `./dataset` directory. 
- This dataset contains extracted features and ground truth labels of AuralNat, including both MCT and clean training and testing sets.
- The data is in MATLAB v7.3 format.
  - The feature dimension is sample size \* 3, with each column representing the log-Gammatone power spectrogram of the left ear (64 \* 39), the log-Gammatone power spectrogram of the right ear (64 \* 39), and cross-correlation values (33 \* 1).
  - The label dimension is sample size \* 8 \* 3, where 8 represents 8 sectors, and 3 represents labels of sound source presence (1 for present, 0 for absence), normalized azimuth (1 for no source), and normalized elevation (-1 for no source).

## Files
- `model.py`: Demo code for AuralNet, implemented with TensorFlow 2.5.0.  
  - Requires [mat73](https://pypi.org/project/mat73/) for loading MATLAB v7.3 files.
- `./model_save/MCT_weights.h5`: Pretrained model weights.
- `./dataset`: Directory for storing the dataset.

## Testing
To test the pretrained MCT model, run the following command:

```
python test.py
```

## Training a New Model
To train a new MCT model from scratch, use the following command:

```
python train.py
```

## Additional Notes
- Ensure that you have installed all the required dependencies before running the test or training scripts.
- If you encounter any issues, please contact [linya.fu@outlook.com](mailto:linya.fu@outlook.com)


