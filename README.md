# AI Tech - Test

# Installation
Setup environment with:

```
 pip install -r requirements.txt
```

# Project description
The project is a joint work between the Marinexplore and Cornell University Whale Detection on the ocean’s environment. The main goal is to transform a code written in Tensorflow in 2013 to PyTorch.

# Usage
## [dataPreparation.py](https://github.com/morphseur/WhaleDetection/blob/main/src/dataPreparation.py)
Following the [Documentation](https://www.kaggle.com/code/diegoasuarezg/transforming-2khz-aiff-whale-audio-to-png), the script allows to transform the data from .aiff to .png files.

```
 python dataPreparation.py --download_path path/to/download/dataset --output_path path/to/prepared/data
```

## [train.py](https://github.com/morphseur/WhaleDetection/blob/main/src/train.py)
The script contains the model architecture, load the dataset and train the model.
```
 python train.py --batch_size 32 --lr 0.001 --num_epochs 5 --data_dir /path/to/data --csv_file /path/to/labelsFile.csv --model_path best_model.pt --weights_path best_model_weights.pt 
```

## [eval.py](https://github.com/morphseur/WhaleDetection/blob/main/src/eval.py)
The script contains the model architecture, load the dataset and evaluate the model.
```
 python eval.py --checkpoint_path path/to/best_model.pt --data_dir /path/to/data --csv_file /path/to/labelsFile.csv --model_path best_model.pt --weights_path best_model_weights.pt 
```

## [loadData.py](https://github.com/morphseur/WhaleDetection/blob/main/src/loadData.py)
The script contains all the required function to have train_dataloader, val_dataloader, test_dataloader, train_dataset, val_dataset, test_dataset.

# TODO
Add arguments to the dataPreparation.py  
Write the predict.py script  
Adapt the code as in the following [Documentation](https://github.com/cs230-stanford/cs230-code-examples/tree/master/pytorch/vision)  
Add to the scripts type-hinting using MyPy and add more documentation  