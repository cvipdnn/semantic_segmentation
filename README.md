# Semantic_Segmentation
This repo shows a sample code to implement semantic segmenation (people segmentation) using unet + mobilenetv2.
The dataset I used was coco2017. 

## Getting Started
Prerequisites
    Keras 2.4.3
    Tensorflow 2.2.0
    opencv for python


## Usage via command line

### Training the Model 
python unet_semantic_seg_train.py

    ## Training Stats
The training set is train2017 and validation set is val2017. The pretained model under model is getting 96.2% accuracy for training set. 

### Inferences
python unet_semantic_seg_inference.py

The testing set is test2017. Some results are shown below:  
![](result.png)


## References
1)https://towardsdatascience.com/master-the-coco-dataset-for-semantic-image-segmentation-part-1-of-2-732712631047
2)https://www.tensorflow.org/tutorials/images/segmentation 
