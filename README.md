# AudioTagging
Working on: https://www.kaggle.com/c/freesound-audio-tagging-2019

## Running

1. Run [00_Preprocess.ipynb](https://github.com/JoshVarty/AudioTagging/blob/master/00_Preprocess.ipynb) (Takes ~2.5 hours 😢)
   - Converts audio files into images and saves them
   - Turns out string labels into binary indicators
   - Perform label smoothing on the noisy dataset
   - Merge the `train_curated.csv` and `train_noisy.csv` into `train_merged.csv`
2. Run [01_BasicModel.ipynb](https://github.com/JoshVarty/AudioTagging/blob/master/01_BasicModel.ipynb)
   - Generates a single (balanced) validation fold based on the curated training set
   - Defines a few simple image transforms
   - Creates an `ImageDataBunch` with batch image normalization `.normalize()`
   - Creates a `vgg16_bn` learner that uses mixup data augmentation

3. Run [src/trainAll.py](https://github.com/JoshVarty/AudioTagging/blob/master/src/trainAll.py)
   - Performs a full training cycle with my best known hyperparameters and network
   - Creates a `/kfolds` folder with validation set predictions
   - Creates a `/model_predictions` folder with test set predictions 
   - Creates a `/model_source` folder with the exact source used to generate a given score

## Optional

- [00_EDA.ipynb](https://github.com/JoshVarty/AudioTagging/blob/master/00_EDA.ipynb) is Exploratory Data Analysis
  - Visualize class balance
  - Visualize audio length 
  - Find incorrect audio file `77b925c2.wav`
  - Example on how to create a validation set that is
     - Only taken from curated dataset
     - Is balanced according to labels (using `MultilabelStratifiedKFold`)
     
- [01_ExploringActivations.ipynb](https://github.com/JoshVarty/AudioTagging/blob/master/01_ExploringActivations.ipynb)
  - Looking at the activations of the network to make sure nothing seems problematic
  
- [03_ImageStats.ipynb](https://github.com/JoshVarty/AudioTagging/blob/master/03_ImageStats.ipynb)
  - My attempt to compute image statistics for normalization (similar to `imagenet_stats` or `mnist_stats`)
  - Unfortunately using the statistics from this doesn't improve performance
    - I have probably made a mistake and am misunderstanding how the statistics should be calculated.
  
