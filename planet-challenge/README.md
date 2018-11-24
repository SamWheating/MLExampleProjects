# PlanetChallenge

Deep learning to classify satellite images of the Amazon

### Requirements:

This project has a lot of dependencies:
- Pandas
- Torch, TorchVision
- Python Image Library (PIL, pillow)
- Numpy, Scipy
- Jupyter Notebooks

### Instructions to run:

This workbook expects several files which were not included in the repo due to size. Follow these instructions to set up your workspace:

1) Go to the Kaggle page ([Here](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/data)) for this competition and download the files `train-jpg.tar.7z`, `train_v2.csv`, `test-jpg.tar.7z` and `test-jpg-additional.tar.7z`.

2) Extract all the images from the compressed folders and move everything from test-jpg-additional into test-jpg.

This repo also uses Github large file system (LFS) to store the pretrained model (because it's ~80MB).

Once that's all set up you can just open up the workbook and run it from top to bottom to see some cool results. 
