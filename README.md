# Description

A handwirtten math expression evaluation project for a course at my University

( Currently not working properly )

## Dataset
https://www.kaggle.com/datasets/xainano/handwrittenmathsymbols
The Dataset contains lots of different math symbols on a 45x45 picture
When reading the dataset I do a data augmentation for better learning

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

## Current segmentation mechanism
- median filter
- morphology operation
- thresholding
- finding contours
- creating a bounding box
- resizeing the ROI image
- sorting the segment based on X coordinate

## Future plans
- Better segmentation mechnism
- Creating some data structure for storing the segments
- Develop an algorithm for reconstructing math symbol from different segments ( like fraction )
 
