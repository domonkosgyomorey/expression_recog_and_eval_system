# Description

A handwirtten math expression evaluation project for a course at my University

( Currently not working properly )

**There is some advanced mechanisms such as:**
- Automatic source file generator for training models
- Automatic dependency installation via `setup.py`

**Main folders**
- `imgsolver` ( segmentation, and prediction )
- `my_cnn_models` ( utilty for creating, training CNN networks )

## Dataset
https://www.kaggle.com/datasets/xainano/handwrittenmathsymbols<br>
The Dataset contains lots of different math symbols on a 45x45 picture
When reading the dataset I do a data augmentation for better learning

**Final Dataset Structure**
- dataset/digit/`(0..9)`/....png
- dataset/operator/`( +,-,times,div )`/....png
- dataset/paren/`( (,) )`/....png
- dataset/trig_log/`( sin,cos,tan,log )`/....png

**Dataset Transformation**
- With `dataset_trainsform.ps1` scirpt, I limited each class of the dataset to **2000**

**Script Usage**
- The `source` var what we need to change when we limiting the number of images in a class

**Recommendations to install and setup the dataset folder**
- Firstly, I recommend to use the  [7-Zip](https://www.7-zip.org/) for faster unzipping folder (much much faster)
- Secondly, I recommend to try not to copy the file, but just cut and paste it into an another directory
- Thirdly, I recommend to compress the dataset before we move it to another folder

## Current models

- A binary classification model which recognize if a character is a number or a math operator
- A number classification model
- A math operator classification model

Each model have three version.

### Version 1
Simple CNN network, without batch normalization

### Version 2
Simple CNN network, with batch normalization

### Version 3
A Simple restNet based model

### Version 4
VGG, restNet based model

## Current segmentation mechanism
- median filter
- morphology operation
- thresholding
- skeletonizing
- finding contours
- creating a bounding box
- resizeing the ROI image
- sorting the segment based on X coordinate

## Future plans
- Better segmentation mechnism
- Creating some data structure for storing the segments
- Develop an algorithm for reconstructing math symbol from different segments ( like fraction )
 
